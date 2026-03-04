import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import cmweather
import cmcrameri
import open_clip

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml_collections import ConfigDict  # noqa: E402

from data.dataset import SatelliteRadarNpyDataset, collate_sat2radar_v2v  # noqa: E402
from diffusion.flow_matching import ODEEulerFlowMatchingSolver  # noqa: E402
from libs.flowtitok import FlowTiTok  # noqa: E402
import flow_utils  # noqa: E402
from scripts.test_sat2radar_v2v import encode_video_with_autoencoder, load_py_config, _ae_config  # noqa: E402


def _cmap_or_fallback(name, fallback="viridis"):
    """Return a valid colormap name, falling back if needed."""
    try:
        plt.get_cmap(name)
        return name
    except ValueError:
        print(f"[WARN] Colormap '{name}' not found, fallback to '{fallback}'")
        return fallback


def _apply_cmap(img2d, cmap_name, vmin, vmax):
    """Apply matplotlib colormap to a 2D array and return RGB image."""
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return cmap(norm(img2d))[..., :3]


def _save_two_panel(orig_2d, pred_2d, cmap_name, vmin, vmax, out_path):
    """
    Save a 2-panel composite [orig | pred] using the given colormap and value range.
    """
    orig_rgb = _apply_cmap(orig_2d, cmap_name, vmin, vmax)
    pred_rgb = _apply_cmap(pred_2d, cmap_name, vmin, vmax)
    composite = np.concatenate([orig_rgb, pred_rgb], axis=1)
    plt.imsave(out_path, composite)


def _save_four_panel_sat_light_radar(
    sat_ir_2d,
    sat_lgt_2d,
    gt_2d,
    pred_2d,
    cmap_ir,
    cmap_lgt,
    cmap_rad,
    ir_min,
    ir_max,
    l_min,
    l_max,
    z_min,
    z_max,
    out_path,
):
    """
    Save a 4-panel composite [sat_IR | sat_lightning | radar_gt | radar_pred]
    using AE-style colormaps and physical ranges.
    """
    sat_ir_rgb = _apply_cmap(sat_ir_2d, cmap_ir, ir_min, ir_max)
    sat_lgt_rgb = _apply_cmap(sat_lgt_2d, cmap_lgt, l_min, l_max)
    gt_rgb = _apply_cmap(gt_2d, cmap_rad, z_min, z_max)
    pred_rgb = _apply_cmap(pred_2d, cmap_rad, z_min, z_max)
    composite = np.concatenate([sat_ir_rgb, sat_lgt_rgb, gt_rgb, pred_rgb], axis=1)
    plt.imsave(out_path, composite)


def build_eval_dataloader(config, split: str, batch_size: int, mode: str):
    """
    Build dataloader for I2I (T=1) or V2V (T>=1) for visualization only.
    """
    assert mode in ["i2i", "v2v"]
    if mode == "i2i":
        num_frames = 1
    else:
        num_frames = config.dataset.get("num_frames", 16)

    # 与训练 / 测试脚本保持一致：使用 ir_band_indices / use_lightning 选择卫星通道
    ir_band_indices = config.dataset.get("ir_band_indices", None)
    use_lightning = config.dataset.get("use_lightning", True)

    dataset = SatelliteRadarNpyDataset(
        base_dir=None,
        years=None,
        mode="sat2radar_v2v",
        filelist_path=config.dataset.filelist_path,
        filelist_split=split,
        files=None,
        frame_stride=config.dataset.get("frame_stride", 1),
        num_frames=num_frames,
        ir_band_indices=ir_band_indices,
        use_lightning=use_lightning,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers_per_gpu,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_sat2radar_v2v,
    )
    
    # Log dataset info
    num_samples = len(dataset)
    num_batches = len(loader)
    num_frames_cfg = num_frames if mode == "i2i" else config.dataset.get("num_frames", 16)
    print(f"Dataset [{split.upper()}]:")
    print(f"  Split: {split}")
    print(f"  Mode: {mode} (sat2radar_v2v)")
    print(f"  Samples: {num_samples}")
    print(f"  Batches: {num_batches}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num frames: {num_frames_cfg}")
    print(f"  Frame stride: {config.dataset.get('frame_stride', 1)}")
    print(f"  Filelist: {config.dataset.filelist_path}")
    
    return loader


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Visualize sat2radar I2I/V2V predictions (no metrics).")
    parser.add_argument("--config", required=True, help="Python config file, e.g. configs/Sat2Radar-Video-XL.py")
    parser.add_argument("--ckpt", required=True, help="TrainState checkpoint dir, e.g. .../100000.ckpt")
    parser.add_argument("--out_dir", required=True, help="Directory to save visualizations")
    parser.add_argument("--split", default="val", help="Dataset split in filelist (train/val/test)")
    parser.add_argument("--mode", default="i2i", choices=["i2i", "v2v"], help="Visualization mode: i2i or v2v")
    parser.add_argument("--max_batches", type=int, default=10, help="Max number of batches to visualize (-1 for all)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES override, e.g. '0'")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_py_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = build_eval_dataloader(config, split=args.split, batch_size=args.batch_size, mode=args.mode)

    # Load model
    train_state = flow_utils.initialize_train_state(config, device)
    train_state.load(args.ckpt)
    nnet_ema = train_state.nnet_ema.to(device)
    nnet_ema.eval()

    # Load AEs
    # 与 train/test 保持兼容：新配置优先使用 sat_in_channels/radar_in_channels，
    # 旧配置回退到 11ch sat / 1ch radar。
    sat_in_ch = getattr(config, "sat_in_channels", None)
    sat_out_ch = getattr(config, "sat_out_channels", None)
    radar_in_ch = getattr(config, "radar_in_channels", None)
    radar_out_ch = getattr(config, "radar_out_channels", None)
    if sat_in_ch is None or sat_out_ch is None:
        sat_in_ch, sat_out_ch = 11, 11
    if radar_in_ch is None or radar_out_ch is None:
        radar_in_ch, radar_out_ch = 1, 1

    sat_ae_config = _ae_config(config, sat_in_ch, sat_out_ch)
    radar_ae_config = _ae_config(config, radar_in_ch, radar_out_ch)

    sat_autoencoder = FlowTiTok(sat_ae_config).to(device)
    sat_autoencoder.load_state_dict(
        torch.load(config.sat_tokenizer_checkpoint, map_location="cpu")
    )
    sat_autoencoder.eval()
    sat_autoencoder.requires_grad_(False)

    radar_autoencoder = FlowTiTok(radar_ae_config).to(device)
    radar_autoencoder.load_state_dict(
        torch.load(config.radar_tokenizer_checkpoint, map_location="cpu")
    )
    radar_autoencoder.eval()
    radar_autoencoder.requires_grad_(False)

    # Text guidance encoder for FlowTiTok decoder（基于文件名的弱描述）
    try:
        clip_encoder, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14-336", pretrained="openai"
        )
        del clip_encoder.visual
        clip_tokenizer = open_clip.get_tokenizer("ViT-L-14-336")
        clip_encoder.transformer.batch_first = False
        clip_encoder.eval()
        clip_encoder.requires_grad_(False)
        clip_encoder.to(device)
    except Exception as e:
        clip_encoder = None
        clip_tokenizer = None
        print(
            f"[WARN] open_clip not available, FlowTiTok decoder will run without text guidance. Error: {e}"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    num_latent_tokens = config.vq_model.num_latent_tokens

    def infer_and_save(batch, batch_idx: int):
        sat_video = batch["sat_video"].to(
            device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )  # [B, T_max, 11, H, W]
        radar_video_gt = batch["radar_video"].to(
            device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )  # [B, T_max, 1, H, W]
        valid_mask = batch.get("valid_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device)

        B, T_max, C_sat, H, W = sat_video.shape

        # Encode sat video -> tokens [B, T_eff*L, C_tok]
        sat_tokens = encode_video_with_autoencoder(
            sat_autoencoder, sat_video, config.vq_model.scale_factor
        )

        x0, _, _ = nnet_ema(sat_tokens, text_encoder=True)
        if config.nnet.model_args.noising_type != "none":
            x0 = x0 + torch.randn_like(x0) * config.sample.noise_scale

        ode_solver = ODEEulerFlowMatchingSolver(
            nnet_ema,
            step_size_type="step_in_dsigma",
            guidance_scale=config.sample.scale,
        )
        z, _ = ode_solver.sample(
            x_T=x0,
            batch_size=B,
            sample_steps=config.sample.sample_steps,
            unconditional_guidance_scale=config.sample.scale,
            has_null_indicator=True,
        )

        L = z.shape[1]
        assert L % num_latent_tokens == 0, "Sequence length must be multiple of num_latent_tokens"
        T_eff = L // num_latent_tokens

        z = z.view(B, T_eff, num_latent_tokens, z.shape[2])
        z = z.view(B * T_eff, num_latent_tokens, z.shape[3])
        z = z.permute(0, 2, 1).unsqueeze(2)

        # 构造基于文件名的弱文本描述，作为 FlowTiTok decoder 的 text_guidance
        text_guidance = None
        if clip_encoder is not None and clip_tokenizer is not None:
            radar_paths = batch.get("radar_paths")
            texts = []
            for i in range(B):
                for t in range(T_eff):
                    fname = "unknown"
                    if radar_paths and i < len(radar_paths):
                        paths_i = radar_paths[i]
                        if isinstance(paths_i, (list, tuple)) and t < len(paths_i):
                            fname = os.path.basename(str(paths_i[t]))
                    desc = f"A radar reflectivity image from {fname}."
                    texts.append(desc)

            if len(texts) == B * T_eff:
                try:
                    text_tokens = clip_tokenizer(texts).to(device)
                    cast_dtype = clip_encoder.transformer.get_cast_dtype()
                    text_tokens = clip_encoder.token_embedding(text_tokens).to(
                        cast_dtype
                    )  # [B*T, n_ctx, d_model]
                    text_tokens = (
                        text_tokens + clip_encoder.positional_embedding.to(cast_dtype)
                    )
                    text_tokens = text_tokens.permute(1, 0, 2)  # NLD -> LND
                    text_tokens = clip_encoder.transformer(
                        text_tokens, attn_mask=clip_encoder.attn_mask
                    )
                    text_tokens = text_tokens.permute(1, 0, 2)  # LND -> NLD
                    text_tokens = clip_encoder.ln_final(
                        text_tokens
                    )  # [B*T, n_ctx, d_model]
                    text_guidance = text_tokens
                except Exception as e:
                    print(f"[WARN] Failed to build CLIP text guidance: {e}")
                    text_guidance = None

        radar_pred = radar_autoencoder.decode_tokens(
            z / config.vq_model.scale_factor, text_guidance=text_guidance
        )  # [B*T_eff, C_out, H_pred, W_pred]，此处 C_out=3, H_pred/W_pred≈512
        # 雷达物理上是单通道，这里只取第一个通道
        radar_pred = radar_pred[:, 0:1, ...]  # [B*T_eff, 1, H_pred, W_pred]
        radar_pred = torch.clamp(radar_pred, 0.0, 1.0)
        radar_pred = radar_pred.view(B, T_eff, 1, radar_pred.shape[-2], radar_pred.shape[-1])
        radar_gt = torch.clamp(radar_video_gt[:, :T_eff], 0.0, 1.0)  # [B, T_eff, 1, H_gt, W_gt]

        # 如果预测雷达的分辨率与原始 GT 不一致（例如 AE 在 512x512 上预训练，
        # 而数据集裁剪为 128x128），则在可视化前把预测 resize 回 GT 的分辨率，
        # 以便后续拼接 [sat_IR | sat_lightning | radar_gt | radar_pred] 时尺寸一致。
        _, _, _, H_gt, W_gt = radar_gt.shape
        if radar_pred.shape[-2] != H_gt or radar_pred.shape[-1] != W_gt:
            Bv, Tv, _, H_pred, W_pred = radar_pred.shape
            radar_pred_flat = radar_pred.view(Bv * Tv, 1, H_pred, W_pred)
            radar_pred_flat = torch.nn.functional.interpolate(
                radar_pred_flat,
                size=(H_gt, W_gt),
                mode="bilinear",
                align_corners=False,
            )
            radar_pred = radar_pred_flat.view(Bv, Tv, 1, H_gt, W_gt)

        # For each sample, save first few frames (or all for small T) as
        # sat+radar composites only: [sat_IR | sat_lightning | radar_gt | radar_pred].
        max_frames_to_show = 4 if args.mode == "v2v" else 1

        z_min, z_max = 0.0, 60.0
        ir_min, ir_max = 200.0, 320.0
        l_min, l_max = 0.1, 50.0
        cmap_rad = _cmap_or_fallback("HomeyerRainbow", fallback="viridis")
        cmap_ir = _cmap_or_fallback("cmc.batlow_r", fallback="viridis")
        cmap_lgt = _cmap_or_fallback("Reds", fallback="Reds")

        # 单帧 PNG：与原逻辑一致，每个样本前若干帧各保存一张四联图
        for i in range(B):
            frames = []
            for t in range(T_eff):
                if valid_mask is not None and not valid_mask[i, t]:
                    continue
                if len(frames) >= max_frames_to_show:
                    break

                # Scale to physical ranges for visualization
                gt_2d = (radar_gt[i, t, 0] * (z_max - z_min) + z_min).detach().cpu().numpy()
                pred_2d = (radar_pred[i, t, 0] * (z_max - z_min) + z_min).detach().cpu().numpy()

                # Save sat+radar composite [sat_IR | sat_lightning | gt | pred]
                sat_ir = sat_video[i, t, 0] * (ir_max - ir_min) + ir_min  # [H, W]
                sat_ir_2d = sat_ir.detach().cpu().numpy()
                if sat_video.shape[2] > 10:
                    sat_lgt = sat_video[i, t, 10] * (l_max - l_min) + l_min
                    sat_lgt_2d = sat_lgt.detach().cpu().numpy()
                else:
                    sat_lgt_2d = np.zeros_like(sat_ir_2d)
                time_stem = ""
                radar_paths = batch.get("radar_paths")
                if radar_paths and i < len(radar_paths) and t < len(radar_paths[i]):
                    p = radar_paths[i][t]
                    time_stem = os.path.splitext(os.path.basename(str(p)))[0].replace(os.sep, "_").replace(":", "_")
                out_name = f"{time_stem}.png" if time_stem else f"batch{batch_idx}_i{i}_t{t:02d}.png"
                out_path = os.path.join(args.out_dir, out_name)
                _save_four_panel_sat_light_radar(
                    sat_ir_2d,
                    sat_lgt_2d,
                    gt_2d,
                    pred_2d,
                    cmap_ir,
                    cmap_lgt,
                    cmap_rad,
                    ir_min,
                    ir_max,
                    l_min,
                    l_max,
                    z_min,
                    z_max,
                    out_path,
                )

        # v2v 模式下，额外为多个样本保存时间序列 GIF
        if args.mode == "v2v" and HAS_IMAGEIO:
            max_samples = min(B, 8)
            video_frames = []
            for t in range(T_eff):
                rows_t = []
                for i in range(max_samples):
                    if valid_mask is not None and not valid_mask[i, t]:
                        continue

                    gt_2d = (radar_gt[i, t, 0] * (z_max - z_min) + z_min).detach().cpu().numpy()
                    pred_2d = (radar_pred[i, t, 0] * (z_max - z_min) + z_min).detach().cpu().numpy()
                    sat_ir = sat_video[i, t, 0] * (ir_max - ir_min) + ir_min  # [H, W]
                    sat_ir_2d = sat_ir.detach().cpu().numpy()
                    if sat_video.shape[2] > 10:
                        sat_lgt = sat_video[i, t, 10] * (l_max - l_min) + l_min
                        sat_lgt_2d = sat_lgt.detach().cpu().numpy()
                    else:
                        sat_lgt_2d = np.zeros_like(sat_ir_2d)

                    sat_ir_rgb = _apply_cmap(sat_ir_2d, cmap_ir, ir_min, ir_max)
                    sat_lgt_rgb = _apply_cmap(sat_lgt_2d, cmap_lgt, l_min, l_max)
                    gt_rgb = _apply_cmap(gt_2d, cmap_rad, z_min, z_max)
                    pred_rgb = _apply_cmap(pred_2d, cmap_rad, z_min, z_max)

                    row_t = np.concatenate(
                        [sat_ir_rgb, sat_lgt_rgb, gt_rgb, pred_rgb], axis=1
                    )  # [H, 4W, 3]
                    rows_t.append(row_t)

                if not rows_t:
                    continue
                frame = np.concatenate(rows_t, axis=0)  # [H*max_samples, 4W, 3]
                frame_uint8 = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
                video_frames.append(frame_uint8)

            if video_frames:
                gif_name = f"batch{batch_idx}_v2v.gif"
                gif_path = os.path.join(args.out_dir, gif_name)
                try:
                    imageio.mimsave(gif_path, video_frames, fps=4)
                except Exception as e:
                    print(f"[WARN] Failed to save GIF {gif_path}: {e}")

    for b_idx, batch in enumerate(loader):
        if args.max_batches >= 0 and b_idx >= args.max_batches:
            break
        infer_and_save(batch, b_idx)

    print(f"Saved sat2radar {args.mode} visualizations to {args.out_dir}")


if __name__ == "__main__":
    main()

