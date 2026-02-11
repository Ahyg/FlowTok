import os
import sys
from pathlib import Path

import accelerate
import torch
from absl import app, flags, logging
from absl import flags
from ml_collections import config_flags, ConfigDict

# Ensure project root is on sys.path so we can import local modules.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import flow_utils
from diffusion.flow_matching import FlowMatching, ODEEulerFlowMatchingSolver
from libs.flowtitok import FlowTiTok
from data.dataset import SatelliteRadarNpyDataset, collate_sat2radar_v2v
from torch.utils.data import DataLoader


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    None,
    "Config file path for Sat2Radar-Video training.",
    lock_config=False,
)


def encode_video_with_autoencoder(autoencoder, video, scale_factor: float):
    """
    video: [B, T, C, H, W]
    返回: tokens [B, T*L, C_tok]
    """
    B, T, C, H, W = video.shape
    video = video.view(B * T, C, H, W)
    with torch.no_grad():
        # FlowTiTok.encode 返回 (z, dict)
        z = autoencoder.encode(video)[0].mul_(scale_factor)  # [B*T, C_tok, 1, L]
    z = z.squeeze(2).permute(0, 2, 1)  # [B*T, L, C_tok]
    L = z.shape[1]
    z = z.view(B, T * L, z.shape[2])   # [B, T*L, C_tok]
    return z


def build_dataloader(config, mode: str, accelerator: accelerate.Accelerator):
    """Build dataloader for real-time v2v (aligned sat↔radar, variable T, T=1 => i2i)."""
    assert mode in ["train", "val"], f"mode must be 'train' or 'val', got {mode}"

    split = "train" if mode == "train" else "val"
    use_v2v = config.dataset.get("v2v", True)

    if use_v2v:
        dataset = SatelliteRadarNpyDataset(
            base_dir=None,
            years=None,
            mode="sat2radar_v2v",
            filelist_path=config.dataset.filelist_path,
            filelist_split=split,
            files=None,
            frame_stride=config.dataset.get("frame_stride", 1),
            num_frames=config.dataset.get("num_frames", (1, 8)),  # (min_t, max_t) => i2i + v2v
        )
        collate_fn = collate_sat2radar_v2v
    else:
        dataset = SatelliteRadarNpyDataset(
            base_dir=None,
            years=None,
            mode="sat2radar_video",
            filelist_path=config.dataset.filelist_path,
            filelist_split=split,
            files=None,
            history_frames=config.dataset.get("history_frames", None),
            future_frames=config.dataset.get("future_frames", None),
            frame_stride=config.dataset.get("frame_stride", 1),
        )
        collate_fn = None

    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size // accelerator.num_processes,
        shuffle=True if mode == "train" else False,
        num_workers=config.dataset.num_workers_per_gpu,
        pin_memory=True,
        drop_last=True if mode == "train" else False,
        collate_fn=collate_fn,
    )
    return dataloader


def train(config):
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    accelerator = accelerate.Accelerator(split_batches=False)
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f"Process {accelerator.process_index} using device: {device}")

    config.mixed_precision = accelerator.mixed_precision
    config = config.copy_and_resolve_references()

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        flow_utils.set_logger(
            log_level="info", fname=os.path.join(config.workdir, "output.log")
        )
        logging.info(config)
    else:
        flow_utils.set_logger(log_level="error")
    logging.info(f"Run on {accelerator.num_processes} devices")

    # ========= Data =========
    train_dataloader = build_dataloader(config, mode="train", accelerator=accelerator)
    # 评估时可以单独 build val dataloader；这里简单起见复用 train split 或者你之后自行扩展
    eval_dataloader = None

    # ========= FlowTok backbone & optimizer =========
    train_state = flow_utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer
    )
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    # ========= Pretrained FlowTiTok autoencoders =========
    # I2I/V2V pipeline: 11ch satellite -> sat tokenizer (77 tokens/frame); DiT -> radar tokens -> radar decoder (1ch).
    # Build sat AE config (11 in/out channels) and radar AE config (1 in/out channel).
    def _ae_config(base_config, in_channels, out_channels):
        vq = dict(base_config.vq_model)
        vq["in_channels"] = in_channels
        vq["out_channels"] = out_channels
        cfg = ConfigDict(dict(base_config))
        cfg.vq_model = ConfigDict(vq)
        return cfg

    sat_ae_config = _ae_config(config, 11, 11)
    radar_ae_config = _ae_config(config, 1, 1)

    sat_autoencoder = FlowTiTok(sat_ae_config)
    sat_autoencoder.load_state_dict(
        torch.load(config.sat_tokenizer_checkpoint, map_location="cpu")
    )
    sat_autoencoder.eval()
    sat_autoencoder.requires_grad_(False)
    sat_autoencoder.to(device)

    radar_autoencoder = FlowTiTok(radar_ae_config)
    radar_autoencoder.load_state_dict(
        torch.load(config.radar_tokenizer_checkpoint, map_location="cpu")
    )
    radar_autoencoder.eval()
    radar_autoencoder.requires_grad_(False)
    radar_autoencoder.to(device)

    flow_matching_model = FlowMatching(
        noising_type=config.nnet.model_args.noising_type,
        noising_scale=config.nnet.model_args.noising_scale,
    )

    num_latent_tokens = config.vq_model.num_latent_tokens

    def train_step(batch):
        metrics = dict()
        optimizer.zero_grad()

        sat_video = batch["sat_video"].to(
            accelerator.device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )  # [B, T_max, C_sat, H, W] (may contain padding)
        radar_video = batch["radar_video"].to(
            accelerator.device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )  # [B, T_max, 1, H, W]

        # tokens: [B, T_max*L, C_tok]
        sat_tokens = encode_video_with_autoencoder(
            sat_autoencoder, sat_video, config.vq_model.scale_factor
        )
        radar_tokens = encode_video_with_autoencoder(
            radar_autoencoder, radar_video, config.vq_model.scale_factor
        )

        # Variable-length v2v: mask padded positions in loss (valid_mask [B, T_max] -> token-level)
        valid_mask = batch.get("valid_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(
                accelerator.device,
                non_blocking=True,
            )
            token_mask = valid_mask.repeat_interleave(num_latent_tokens, dim=1)  # [B, T_max*L]
        else:
            token_mask = None

        # x_start = radar tokens, cond = sat tokens
        loss, loss_dict = flow_matching_model(
            x=radar_tokens,
            nnet=nnet,
            cond=sat_tokens,
            all_config=config,
            batch_img_clip=None,
            valid_mask=token_mask,
        )

        metrics["loss"] = accelerator.gather(loss.detach()).mean()
        for key, val in loss_dict.items():
            metrics[key] = accelerator.gather(val.detach()).mean()

        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get("ema_rate", 0.9999))
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]["lr"], **metrics)

    def ode_fm_solver_sample(nnet_ema_local, batch):
        """
        简单的采样示例：给一个 batch 的 sat_video，生成对应的 radar_video。
        """
        with torch.no_grad():
            sat_video = batch["sat_video"].to(
                accelerator.device,
                memory_format=torch.contiguous_format,
                non_blocking=True,
            )
            B = sat_video.shape[0]
            sat_tokens = encode_video_with_autoencoder(
                sat_autoencoder, sat_video, config.vq_model.scale_factor
            )  # [B, L, C]

            x0, _, _ = nnet_ema_local(sat_tokens, text_encoder=True)
            if config.nnet.model_args.noising_type != "none":
                x0 = x0 + torch.randn_like(x0) * config.sample.noise_scale

            guidance_scale = config.sample.scale
            has_null_indicator = True

            ode_solver = ODEEulerFlowMatchingSolver(
                nnet_ema_local,
                step_size_type="step_in_dsigma",
                guidance_scale=guidance_scale,
            )
            z, _ = ode_solver.sample(
                x_T=x0,
                batch_size=B,
                sample_steps=config.sample.sample_steps,
                unconditional_guidance_scale=guidance_scale,
                has_null_indicator=has_null_indicator,
            )
            # z: [B, L, C_tok] -> [B*T, C_tok, 1, L_frame] 再 decode
            # 这里假设 T 与训练时一致，可以由 L 和 num_latent_tokens 反推
            L = z.shape[1]
            num_latent_tokens = config.vq_model.num_latent_tokens
            assert (
                L % num_latent_tokens == 0
            ), "Sequence length must be multiple of num_latent_tokens"
            T = L // num_latent_tokens

            z = z.view(B, T, num_latent_tokens, z.shape[2])  # [B, T, Lf, C]
            z = z.view(B * T, num_latent_tokens, z.shape[3])  # [B*T, Lf, C]
            z = z.permute(0, 2, 1).unsqueeze(2)  # [B*T, C, 1, Lf]

            radar_video_pred = radar_autoencoder.decode_tokens(
                z / config.vq_model.scale_factor, text_guidance=None
            )  # [B*T, 1, H, W]
            radar_video_pred = radar_video_pred.view(B, T, 1, radar_video_pred.shape[-2], radar_video_pred.shape[-1])
            return radar_video_pred

    logging.info(
        f"Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}"
    )

    while train_state.step < config.train.n_steps:
        for batch in train_dataloader:
            nnet.train()
            metrics = train_step(batch)

            nnet.eval()
            if (
                accelerator.is_main_process
                and train_state.step % config.train.log_interval == 0
            ):
                logging.info(flow_utils.dct2str(dict(step=train_state.step, **metrics)))
                logging.info(config.workdir)

            # 保存一些可视化样例
            if train_state.step % config.train.eval_interval == 0:
                torch.cuda.empty_cache()
                logging.info("Save a grid of [sat | radar_gt | radar_pred] (first frame only)...")
                with torch.no_grad():
                    # 预测雷达视频
                    radar_video_pred = ode_fm_solver_sample(nnet_ema, batch)  # [B, T, 1, H, W]
                    B, T_eff, _, H, W = radar_video_pred.shape

                    # 从 batch 中取出对应的 sat / radar GT（只看第 1 帧）
                    sat_video = batch["sat_video"].to(
                        accelerator.device,
                        memory_format=torch.contiguous_format,
                        non_blocking=True,
                    )  # [B, T, C_sat, H, W]
                    radar_video_gt = batch["radar_video"].to(
                        accelerator.device,
                        memory_format=torch.contiguous_format,
                        non_blocking=True,
                    )  # [B, T, 1, H, W]

                    # 只可视化每个样本的第 1 帧，并统一为单通道：
                    # sat: 取第 0 个通道灰度作为 quick-look
                    sat_first = sat_video[:, 0, 0:1]          # [B, 1, H, W]
                    radar_gt_first = radar_video_gt[:, 0]    # [B, 1, H, W]
                    radar_pred_first = radar_video_pred[:, 0]  # [B, 1, H, W]

                    import torchvision.utils as vutils

                    # 对每个样本拼成一行：[sat | radar_gt | radar_pred]
                    max_samples = min(B, 8)
                    rows = []
                    for i in range(max_samples):
                        row = torch.cat(
                            [sat_first[i], radar_gt_first[i], radar_pred_first[i]],
                            dim=-1,
                        )  # [1, H, 3W]
                        rows.append(row)

                    if rows:
                        # 竖直堆叠多行，得到 [1, H*max_samples, 3W]
                        stacked = torch.cat(rows, dim=-2)

                        if accelerator.is_main_process:
                            save_path = os.path.join(
                                config.sample_dir,
                                f"{train_state.step}_sat_radar_gt_pred_first_frames.png",
                            )
                            vutils.save_image(stacked, save_path)
                accelerator.wait_for_everyone()
                torch.cuda.empty_cache()

            # 保存 checkpoint
            if (
                train_state.step % config.train.save_interval == 0
                or train_state.step == config.train.n_steps
            ):
                torch.cuda.empty_cache()
                logging.info(f"Save checkpoint {train_state.step}...")
                if accelerator.local_process_index == 0:
                    train_state.save(
                        os.path.join(config.ckpt_root, f"{train_state.step}.ckpt")
                    )
                accelerator.wait_for_everyone()
                torch.cuda.empty_cache()

            if train_state.step >= config.train.n_steps:
                break
        accelerator.wait_for_everyone()

    logging.info(f"Finish fitting, step={train_state.step}")
    accelerator.wait_for_everyone()


def main(_):
    config = FLAGS.config
    train(config)


if __name__ == "__main__":
    app.run(main)

