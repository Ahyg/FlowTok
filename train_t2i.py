import ml_collections
import torch
from torch import multiprocessing as mp
from torchvision.utils import make_grid, save_image
import flow_utils as utils
import accelerate
from absl import logging
import builtins
import os
import open_clip
from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path

from libs.flowtitok import FlowTiTok, DiagonalGaussianDistribution
from libs.evaluator import TextCondBertEvaluator
from diffusion.flow_matching import FlowMatching, ODEEulerFlowMatchingSolver
from data.webdataset_reader import SimpleImageDataset, PretokenizedWebDataset


def train(config):
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    accelerator = accelerate.Accelerator(split_batches=False)
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    if config.dataset.pretokenized:
        dataset = PretokenizedWebDataset(
            train_shards_path=config.dataset.train_shards_path_or_url,
            eval_shards_path=config.dataset.eval_shards_path_or_url,
            num_train_examples=config.dataset.max_train_examples,
            per_gpu_batch_size=mini_batch_size,
            global_batch_size=config.train.batch_size,
            num_workers_per_gpu=config.dataset.num_workers_per_gpu,
            resize_shorter_edge=config.dataset.resize_shorter_edge,
            crop_size=config.dataset.crop_size,
            random_crop=config.dataset.random_crop,
            random_flip=config.dataset.random_flip,
        )
    else:
        dataset = SimpleImageDataset(
            train_shards_path=config.dataset.train_shards_path_or_url,
            eval_shards_path=config.dataset.eval_shards_path_or_url,
            num_train_examples=config.dataset.max_train_examples,
            per_gpu_batch_size=mini_batch_size,
            global_batch_size=config.train.batch_size,
            num_workers_per_gpu=config.dataset.num_workers_per_gpu,
            resize_shorter_edge=config.dataset.resize_shorter_edge,
            crop_size=config.dataset.crop_size,
            random_crop=config.dataset.random_crop,
            random_flip=config.dataset.random_flip,
            dataset_with_class_label=config.dataset.dataset_with_class_label,
            dataset_with_text_label=config.dataset.dataset_with_text_label,
            res_ratio_filtering=config.dataset.res_ratio_filtering
        )
    train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader

    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer)
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)
    
    # Flow-TiTok
    autoencoder = FlowTiTok(config)
    autoencoder.load_state_dict(torch.load(config.tokenizer_checkpoint, map_location="cpu"))
    autoencoder.eval()
    autoencoder.requires_grad_(False)
    autoencoder.to(device)

    # CLIP
    clip_encoder, _, _ = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
    del clip_encoder.visual
    clip_tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
    clip_encoder.transformer.batch_first = False
    clip_encoder.eval()
    clip_encoder.requires_grad_(False)
    clip_encoder.to(device)

    ClipSocre_model = None

    # Evaluator
    evaluator = TextCondBertEvaluator(
        device=accelerator.device,
        enable_fid=True,
        stat_path="/opt/tiger/ju/data/coco30k_fid_stat.pth",
    )

    _flow_mathcing_model = FlowMatching(noising_type=config.nnet.model_args.noising_type, noising_scale=config.nnet.model_args.noising_scale)

    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()

        if config.dataset.pretokenized:
            _batch_img = None
            _z = _batch["tokens"].to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True)
            bsz = _z.shape[0]
            _z = _z.reshape(bsz, config.vq_model.token_size * 2, 1, -1) # [B, C, 1, L]
            posterior = DiagonalGaussianDistribution(_z)
            _z = posterior.sample().mul_(config.vq_model.scale_factor)
            _z = _z.squeeze(2).permute(0,2,1) # [B, L, C]
        else:
            _batch_img = _batch["image"].to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True)
            with torch.no_grad():
                _z = autoencoder.encode(_batch_img)[0].mul_(config.vq_model.scale_factor) # [B, C, 1, L]
                _z = _z.squeeze(2).permute(0,2,1) # [B, L, C]
        
        _batch_txt = _batch["text"]
        with torch.no_grad():
            text_tokens = clip_tokenizer(_batch_txt).to(accelerator.device)
            cast_dtype = clip_encoder.transformer.get_cast_dtype()
            text_tokens = clip_encoder.token_embedding(text_tokens).to(cast_dtype)  # [batch_size, n_ctx, d_model]
            text_tokens = text_tokens + clip_encoder.positional_embedding.to(cast_dtype)
            text_tokens = text_tokens.permute(1, 0, 2)  # NLD -> LND
            text_tokens = clip_encoder.transformer(text_tokens, attn_mask=clip_encoder.attn_mask)
            text_tokens = text_tokens.permute(1, 0, 2)  # LND -> NLD
            text_tokens = clip_encoder.ln_final(text_tokens)  # [batch_size, n_ctx, transformer.width]
            
        loss, loss_dict = _flow_mathcing_model(_z, nnet, cond=text_tokens, all_config=config, batch_img_clip=_batch_img)

        _metrics['loss'] = accelerator.gather(loss.detach()).mean()
        for key in loss_dict.keys():
            _metrics[key] = accelerator.gather(loss_dict[key].detach()).mean()
        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def ode_fm_solver_sample(nnet_ema, _n_samples, _sample_steps, batch, return_clipScore=False, ClipSocre_model=None):
        with torch.no_grad():
            _batch_txt = batch["text"][:_n_samples]

            text_tokens = clip_tokenizer(_batch_txt).to(accelerator.device)
            cast_dtype = clip_encoder.transformer.get_cast_dtype()
            text_tokens = clip_encoder.token_embedding(text_tokens).to(cast_dtype)  # [batch_size, n_ctx, d_model]
            text_tokens = text_tokens + clip_encoder.positional_embedding.to(cast_dtype)
            text_tokens = text_tokens.permute(1, 0, 2)  # NLD -> LND
            text_tokens = clip_encoder.transformer(text_tokens, attn_mask=clip_encoder.attn_mask)
            text_tokens = text_tokens.permute(1, 0, 2)  # LND -> NLD
            text_tokens = clip_encoder.ln_final(text_tokens)  # [batch_size, n_ctx, transformer.width]

            _z_x0, _, _ = nnet_ema(text_tokens, text_encoder=True)

            if config.nnet.model_args.noising_type != "none":
                _z_x0 = _z_x0 + torch.randn_like(_z_x0) * config.sample.noise_scale
            
            assert config.sample.scale > 1
            _cfg = config.sample.scale

            has_null_indicator = True

            ode_solver = ODEEulerFlowMatchingSolver(nnet_ema, step_size_type="step_in_dsigma", guidance_scale=_cfg)
            _z, _ = ode_solver.sample(x_T=_z_x0, batch_size=_n_samples, sample_steps=_sample_steps, unconditional_guidance_scale=_cfg, has_null_indicator=has_null_indicator)
            _z = _z.permute(0,2,1).unsqueeze(2)

            image_unprocessed = autoencoder.decode_tokens(_z / config.vq_model.scale_factor, text_guidance=text_tokens)

            if return_clipScore:
                clip_score = ClipSocre_model.calculate_clip_score(_batch_txt, image_unprocessed)
                return image_unprocessed, clip_score
            else:
                return image_unprocessed

    def eval_step(evaluate_dataloader, sample_steps):
        logging.info(f'eval_step: sample_steps={sample_steps}, algorithm=ODE_Euler_Flow_Matching_Solver, '
                     f'mini_batch_size={config.sample.mini_batch_size}')

        for i, batch in enumerate(evaluate_dataloader):
            captions = batch['text']
            num_generated_images = len(captions)
            samples = ode_fm_solver_sample(nnet, num_generated_images, sample_steps, batch, return_clipScore=False, ClipSocre_model=None)
            generated = torch.clamp(samples, 0.0, 1.0) * 255.0
            evaluated_image = torch.round(generated) / 255.0
            evaluator.update(evaluated_image, captions)
            if i % 50 == 0:
                print(f"Evaluation step {i}")

        eval_scores = evaluator.result()
        _fid = torch.tensor(eval_scores["FID"], device=device)

        return _fid.item()

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_fid = []
    while train_state.step < config.train.n_steps:
        for batch in train_dataloader:
            nnet.train()
            metrics = train_step(batch)

            nnet.eval()
            if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
                logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
                logging.info(config.workdir)

            ############# save rigid image
            if train_state.step % config.train.eval_interval == 0:
                torch.cuda.empty_cache()
                logging.info('Save a grid of images...')
                samples = ode_fm_solver_sample(nnet_ema, _n_samples=config.train.n_samples_eval, _sample_steps=50, batch=batch)
                samples = make_grid(samples, 5)
                if accelerator.is_main_process:
                    save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
                accelerator.wait_for_everyone()
                torch.cuda.empty_cache()

            ############ save checkpoint and evaluate results
            if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
                torch.cuda.empty_cache()
                logging.info(f'Save and eval checkpoint {train_state.step}...')

                if accelerator.local_process_index == 0:
                    train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
                accelerator.wait_for_everyone()

                fid = eval_step(evaluate_dataloader=eval_dataloader, sample_steps=50)  # calculate fid of the saved checkpoint
                step_fid.append((train_state.step, fid))
                print(f'step: {train_state.step}, fid: {fid}')

                torch.cuda.empty_cache()

            if train_state.step >= config.train.n_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {train_state.step} >= {config.training.max_train_steps}"
                )
                break
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_fid: {step_fid}')
    accelerator.wait_for_everyone()


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join('workdir', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)
