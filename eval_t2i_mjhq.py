import ml_collections
import torch
import utils
import accelerate
from tqdm.auto import tqdm
from absl import logging
import builtins
import os
import open_clip
from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path
import json

from libs.flowtitok import FlowTiTok
from diffusion.flow_matching import ODEEulerFlowMatchingSolver
from cleanfid import fid


import torch
import torchvision.transforms.functional as DF


def categorize_json_data(file_path):
    # Initialize dictionary to store categories
    categories = {}
    
    try:
        # Read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Process each item in the JSON
        for id, item in data.items():
            category = item.get('category')
            prompt = item.get('prompt')
            
            # If category doesn't exist in our categories dict, create a new list
            if category not in categories:
                categories[category] = []
            
            # Append the item to appropriate category list
            categories[category].append({
                'id': id,
                'prompt': prompt
            })
        
        return categories
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return None


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

    nnet = utils.get_nnet(**config.nnet)
    total_params = sum(p.numel() for p in nnet.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    # nnet = accelerator.prepare(nnet)
    flowtok_path = None # put your model path here
    nnet.load_state_dict(torch.load(flowtok_path, map_location='cpu'))
    nnet.eval()
    
    # Flow-TiTok
    autoencoder = FlowTiTok(config)
    autoencoder.load_state_dict(torch.load(config.tokenizer_checkpoint, map_location="cpu"))
    autoencoder.eval()
    autoencoder.requires_grad_(False)
    # autoencoder.to(device)

    # CLIP
    clip_encoder, _, _ = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
    del clip_encoder.visual
    clip_tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
    clip_encoder.transformer.batch_first = False
    clip_encoder.eval()
    clip_encoder.requires_grad_(False)
    # clip_encoder.to(device)
    nnet, autoencoder, clip_encoder = accelerator.prepare(nnet, autoencoder, clip_encoder)

    # ClipSocre_model = ClipSocre(device=device)
    ClipSocre_model = None

    def ode_fm_solver_sample(nnet_ema, _n_samples, _sample_steps, captions, cfg):
        with torch.no_grad():
            text_tokens = clip_tokenizer(captions).to(accelerator.device)
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
            
            _cfg = cfg

            has_null_indicator = True

            ode_solver = ODEEulerFlowMatchingSolver(nnet_ema, step_size_type="step_in_dsigma", guidance_scale=_cfg)
            _z, _ = ode_solver.sample(x_T=_z_x0, batch_size=_n_samples, sample_steps=_sample_steps, unconditional_guidance_scale=_cfg, has_null_indicator=has_null_indicator)
            _z = _z.permute(0,2,1).unsqueeze(2)

            image_unprocessed = autoencoder.decode_tokens(_z / config.vq_model.scale_factor, text_guidance=text_tokens)

            return image_unprocessed

    eval_bsz = 150
    steps = 20
    categories = ["animals", "art", "fashion", "food", "indoor", "landscape", "logo", "people", "plants", "vehicles"]

    # Load jsonl files containing prompts
    json_path = "/opt/tiger/ju/data/MJHQ-30K/meta_data.json"
    prompts_per_category = categorize_json_data(json_path)
    for cfg in [2.0]:
        root = os.path.join('/opt/tiger/ju/vis/FlowTok-H/MJHQ', f'mjhq_cfg_{cfg}_step_{steps}')
        os.makedirs(root, exist_ok=True)
        for category in categories:
            os.makedirs(os.path.join(root, category), exist_ok=True)
            ids_prompts = prompts_per_category[category]
            ids_prompts = sorted(ids_prompts, key=lambda x: x['id'])
            for i in tqdm(range(0, len(ids_prompts), eval_bsz)):
                batch = ids_prompts[i:i+eval_bsz]
                id = map(lambda x: x['id'], batch)
                captions = list(map(lambda x: x['prompt'], batch))
                samples = ode_fm_solver_sample(nnet, eval_bsz, steps, captions, cfg)
                if accelerator.is_main_process:
                    generated = samples.mul(255).add_(0.5).clamp_(0, 255)
                    images_for_saving = [DF.to_pil_image(gen.cpu().byte()) for gen in generated]

                    for i, (id, img) in enumerate(zip(id, images_for_saving)):
                        filename = f"{id}.png"
                        path = os.path.join(os.path.join(root, category), filename)
                        img.save(path)
        
        score = fid.compute_fid(root, dataset_name=f"MJHQ-30K_all", mode="clean", dataset_split="custom")
        with open(os.path.join("/opt/tiger/ju/vis/FlowTok-H/MJHQ/", "fid_scores.txt"), 'a') as file:
            file.write(f'{cfg}_{steps}: {score}\n')

    


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
