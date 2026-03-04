import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from tqdm import tqdm
import cmweather
import cmcrameri
import open_clip

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# Import pysteps FSS functions
try:
    from pysteps.verification.spatialscores import (
        fss as pysteps_fss,
        fss_init,
        fss_accum,
        fss_compute,
        fss_merge,
    )
    PYSTEPS_AVAILABLE = True
except ImportError:
    PYSTEPS_AVAILABLE = False
    print("[WARN] pysteps not available. FSS comparison will be skipped.")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml_collections import ConfigDict  # noqa: E402

from data.dataset import SatelliteRadarNpyDataset, collate_sat2radar_v2v  # noqa: E402
from diffusion.flow_matching import ODEEulerFlowMatchingSolver  # noqa: E402
from libs.flowtitok import FlowTiTok  # noqa: E402
import flow_utils  # noqa: E402


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


@torch.no_grad()
def encode_video_with_autoencoder(autoencoder, video, scale_factor: float):
    """
    video: [B, T, C, H, W]
    return: tokens [B, T*L, C_tok]
    """
    B, T, C, H, W = video.shape

    # 将输入 resize 到 AE 预训练时的分辨率（例如 512x512），
    # 避免 FlowTiTok 的 RoPE 位置编码与 checkpoint 中的形状不一致。
    ae_ds_cfg = getattr(getattr(autoencoder, "config", None), "dataset", None)
    if isinstance(ae_ds_cfg, dict):
        target_size = ae_ds_cfg.get("crop_size", H)
    else:
        target_size = getattr(getattr(autoencoder, "config", None), "ae_image_size", H)

    video = video.view(B * T, C, H, W)

    # 通道对齐：FlowTiTok_512 预训练是 3 通道 in / 3 通道 out。
    # 如果 AE 期望 3 通道而当前是 1 通道（雷达），则将单通道复制为 3 通道。
    expected_in_ch = getattr(getattr(autoencoder, "config", None), "vq_model", {}).get(
        "in_channels", C
    )
    if C == 1 and expected_in_ch == 3:
        video = video.repeat(1, 3, 1, 1)  # [B*T, 3, H, W]

    if H != target_size or W != target_size:
        video = F.interpolate(
            video,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )

    z = autoencoder.encode(video)[0].mul_(scale_factor)  # [B*T, C_tok, 1, L]
    z = z.squeeze(2).permute(0, 2, 1)  # [B*T, L, C_tok]
    L = z.shape[1]
    # interpolate / repeat 等操作可能导致非 contiguous，view 会报错，改用 reshape 更安全
    z = z.reshape(B, T * L, z.shape[2])   # [B, T*L, C_tok]
    return z


def load_py_config(config_path: str):
    """
    Load a Python config file (e.g. configs/Sat2Radar-Video-XL.py) that defines get_config().
    """
    import importlib.util

    config_path = os.path.abspath(config_path)
    spec = importlib.util.spec_from_file_location("sat2radar_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_config"):
        raise AttributeError(f"{config_path} must define get_config()")
    return module.get_config()


def build_eval_dataloader(config, split: str, batch_size: int, mode: str):
    """
    Build dataloader for I2I (T=1) or V2V (T>=1).
    Assumes filelist was built with build_dataset.py --v2v --clip-length {1 or >1}.
    """
    assert mode in ["i2i", "v2v"]
    if mode == "i2i":
        num_frames = 1
    else:
        num_frames = config.dataset.get("num_frames", 16)

    # 与训练脚本保持一致：使用 ir_band_indices / use_lightning 选择卫星通道
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


def _ae_config(base_config, in_channels, out_channels):
    """
    Build a FlowTiTok autoencoder config compatible with pretrained FlowTiTok_512.bin.

    与训练脚本 train_sat2radar_v2v.py 中的 _ae_config 行为保持一致：
      - 使用 sat_in_channels / radar_in_channels 指定 in/out 通道；
      - 如果配置中提供 ae_image_size，则将 AE 内部的 dataset.crop_size 覆盖为该分辨率，
        并在 config 上记录 ae_image_size，便于 encode 时读取。
    """
    vq = dict(base_config.vq_model)
    vq["in_channels"] = in_channels
    vq["out_channels"] = out_channels
    cfg = ConfigDict(dict(base_config))
    cfg.vq_model = ConfigDict(vq)

    # 为 AE 单独设置预训练时的图像尺寸（例如 512x512），与 FlowTiTok_512 checkpoint 对齐
    ae_img_size = getattr(base_config, "ae_image_size", None)
    if ae_img_size is not None:
        ds = dict(getattr(base_config, "dataset", {}))
        ds["crop_size"] = ae_img_size
        cfg.dataset = ConfigDict(ds)
        # 也记录到 config 上，encode 时可读取
        cfg.ae_image_size = ae_img_size

    return cfg


def _fss_score(pred2d, target2d, thr, scale):
    # pred2d/target2d: torch.Tensor [H, W] in physical units (e.g. dBZ)
    pred_bin = (pred2d >= thr).float()
    target_bin = (target2d >= thr).float()
    scale = max(int(scale), 1)
    padding = scale // 2
    pred_frac = F.avg_pool2d(pred_bin[None, None], kernel_size=scale, stride=1, padding=padding)[0, 0]
    tgt_frac = F.avg_pool2d(target_bin[None, None], kernel_size=scale, stride=1, padding=padding)[0, 0]
    num = torch.sum((pred_frac - tgt_frac) ** 2)
    den = torch.sum(pred_frac ** 2 + tgt_frac ** 2)
    if den.item() == 0:
        return 1.0
    return 1.0 - (num / den).item()


def _avg_fss(pred2d, target2d, thrs, scales):
    if not thrs:
        thrs = [0.0]
    if not scales:
        scales = [1]
    scores = []
    for thr in thrs:
        for scale in scales:
            scores.append(_fss_score(pred2d, target2d, thr, scale))
    return float(np.mean(scores))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Test sat2radar I2I/V2V model with metrics.")
    parser.add_argument("--config", required=True, help="Python config file, e.g. configs/Sat2Radar-Video-XL.py")
    parser.add_argument("--ckpt", required=True, help="TrainState checkpoint dir, e.g. .../100000.ckpt")
    parser.add_argument("--out_dir", required=True, help="Directory to save predicted radar images")
    parser.add_argument("--split", default="test", help="Dataset split in filelist (train/val/test)")
    parser.add_argument("--mode", default="i2i", choices=["i2i", "v2v"], help="Evaluation mode: i2i (T=1) or v2v (T>=1)")
    parser.add_argument("--max_batches_metrics", type=int, default=10, help="Max batches for metrics (-1 for all)")
    parser.add_argument("--max_batches_images", type=int, default=2, help="Max batches to save images (-1 for all)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES override, e.g. '0'")
    parser.add_argument("--fss_thresholds", default="0,5,10,15,20,25,30,35,40,45,50,55,60")
    parser.add_argument("--fss_scales", default="1,2,3,4,5,6,7,8,9,10")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_py_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader
    loader = build_eval_dataloader(config, split=args.split, batch_size=args.batch_size, mode=args.mode)

    # FlowTok backbone & optimizer (for loading nnet / nnet_ema)
    train_state = flow_utils.initialize_train_state(config, device)
    train_state.load(args.ckpt)  # loads step, optimizer, lr_scheduler, nnet, nnet_ema
    nnet = train_state.nnet.to(device)
    nnet_ema = train_state.nnet_ema.to(device)
    nnet_ema.eval()

    # Pretrained FlowTiTok autoencoders
    # 这里和 train_sat2radar_v2v.py 保持兼容：
    #   - 新配置会通过 config.sat_in_channels / radar_in_channels 指定通道
    #   - 旧配置则回退到 11ch sat / 1ch radar
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

    # FSS / metrics accumulators
    def _parse_csv_numbers(value, cast=float):
        if value is None or value == "":
            return []
        return [cast(v.strip()) for v in value.split(",") if v.strip() != ""]

    thrs = _parse_csv_numbers(args.fss_thresholds, cast=float)
    scales = _parse_csv_numbers(args.fss_scales, cast=int)
    z_min, z_max = 0.0, 60.0  # radar scaling range in dataset

    mse_total = 0.0
    mse_count = 0
    ssim_total = 0.0
    ssim_count = 0
    fss_total = 0.0
    fss_count = 0
    
    # pysteps FSS accumulators (one per threshold-scale combination)
    pysteps_fss_objects = {}
    # Per-pair FSS comparison (for detailed analysis)
    fss_pairwise_comparisons = []  # List of (original_fss, pysteps_fss) tuples
    
    if PYSTEPS_AVAILABLE:
        for thr in thrs:
            for scale in scales:
                key = (thr, scale)
                pysteps_fss_objects[key] = fss_init(thr=thr, scale=float(scale))

    def infer_batch(batch, batch_idx: int):
        nonlocal mse_total, mse_count, ssim_total, ssim_count, fss_total, fss_count
        nonlocal pysteps_fss_objects, fss_pairwise_comparisons

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
        )  # [B, T_max*L, C]

        # FlowTok text encoder branch: 默认使用 textVAE 对 sat tokens 编码得到 x0。
        # 若在配置中显式关闭 textVAE（config.use_text_vae_encoder == False），
        # 则直接使用 sat_tokens 作为 flow 起点，实现“sat tokens -> radar tokens”的显式映射。
        use_text_vae_encoder = getattr(config, "use_text_vae_encoder", True)
        if use_text_vae_encoder:
            x0, _, _ = nnet_ema(sat_tokens, text_encoder=True)
        else:
            x0 = sat_tokens
        if config.nnet.model_args.noising_type != "none":
            x0 = x0 + torch.randn_like(x0) * config.sample.noise_scale

        guidance_scale = config.sample.scale
        ode_solver = ODEEulerFlowMatchingSolver(
            nnet_ema,
            step_size_type="step_in_dsigma",
            guidance_scale=guidance_scale,
        )
        z, _ = ode_solver.sample(
            x_T=x0,
            batch_size=B,
            sample_steps=config.sample.sample_steps,
            unconditional_guidance_scale=guidance_scale,
            has_null_indicator=True,
        )  # [B, L, C_tok]

        # Reshape tokens back to [B, T_eff, C_tok, 1, L_frame] and decode.
        L = z.shape[1]
        assert (
            L % num_latent_tokens == 0
        ), f"Sequence length must be multiple of num_latent_tokens: got {L} vs {num_latent_tokens}"
        T_eff = L // num_latent_tokens

        z = z.view(B, T_eff, num_latent_tokens, z.shape[2])  # [B, T, Lf, C]
        z = z.view(B * T_eff, num_latent_tokens, z.shape[3])  # [B*T, Lf, C]
        z = z.permute(0, 2, 1).unsqueeze(2)  # [B*T, C, 1, Lf]

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
        )  # [B*T_eff, C_out, H_pred, W_pred]
        # 由于雷达物理上是单通道，只取第一个通道
        radar_pred = radar_pred[:, 0:1, ...]  # [B*T_eff, 1, H_pred, W_pred]
        radar_pred = torch.clamp(radar_pred, 0.0, 1.0)
        radar_pred = radar_pred.view(B, T_eff, 1, radar_pred.shape[-2], radar_pred.shape[-1])
        radar_gt = torch.clamp(radar_video_gt[:, :T_eff], 0.0, 1.0)  # [B, T_eff, 1, H_gt, W_gt]

        # 如果预测雷达的分辨率与原始 GT 不一致（例如 AE 在 512x512 上预训练，
        # 而数据集裁剪为 128x128），则在计算指标 / 可视化前把预测 resize 回 GT 的分辨率，
        # 以便 MSE / SSIM / FSS 和 PNG/GIF 的空间尺寸与训练/验证保持一致。
        _, _, _, H_gt, W_gt = radar_gt.shape
        if radar_pred.shape[-2] != H_gt or radar_pred.shape[-1] != W_gt:
            Bv, Tv, _, H_pred, W_pred = radar_pred.shape
            radar_pred_flat = radar_pred.view(Bv * Tv, 1, H_pred, W_pred)
            radar_pred_flat = F.interpolate(
                radar_pred_flat,
                size=(H_gt, W_gt),
                mode="bilinear",
                align_corners=False,
            )
            radar_pred = radar_pred_flat.view(Bv, Tv, 1, H_gt, W_gt)

        # Metrics per (b, t) where valid_mask is True (or all frames if mask is None)
        radar_pred_cpu = radar_pred.detach().cpu()
        radar_gt_cpu = radar_gt.detach().cpu()

        for b in range(B):
            for t in range(T_eff):
                if valid_mask is not None and not valid_mask[b, t]:
                    continue
                gt = radar_gt_cpu[b, t, 0].numpy()
                pred = radar_pred_cpu[b, t, 0].numpy()

                # MSE
                mse_total += float(np.mean((pred - gt) ** 2))
                mse_count += 1

                # SSIM on 0-1 normalized
                ssim_total += ssim(gt, pred, data_range=1.0)
                ssim_count += 1

                # FSS on physical dBZ range
                pred_scaled = torch.from_numpy(pred * (z_max - z_min) + z_min)
                tgt_scaled = torch.from_numpy(gt * (z_max - z_min) + z_min)
                
                # Original FSS calculation (average over all threshold-scale combinations)
                fss_orig = _avg_fss(pred_scaled, tgt_scaled, thrs, scales)
                fss_total += fss_orig
                fss_count += 1
                
                # pysteps FSS calculation
                if PYSTEPS_AVAILABLE:
                    pred_np = pred_scaled.numpy()
                    tgt_np = tgt_scaled.numpy()
                    
                    # Compute per-pair FSS using pysteps (average over all threshold-scale combinations)
                    pysteps_fss_values_per_pair = []
                    for thr in thrs:
                        for scale in scales:
                            # Single pair FSS calculation
                            fss_val = pysteps_fss(pred_np, tgt_np, thr=thr, scale=float(scale))
                            pysteps_fss_values_per_pair.append(fss_val)
                            
                            # Also accumulate for overall average
                            key = (thr, scale)
                            fss_accum(pysteps_fss_objects[key], pred_np, tgt_np)
                    
                    # Average FSS for this pair (comparable to original method)
                    pysteps_avg_per_pair = np.mean(pysteps_fss_values_per_pair)
                    fss_pairwise_comparisons.append((fss_orig, pysteps_avg_per_pair))

        # Save images / videos：
        #   - PNG：每个样本保存第一个有效帧的 [sat_IR | sat_lightning | radar_gt | radar_pred]
        #   - GIF（仅 v2v 且安装了 imageio）：多样本堆叠，沿时间维展示演化
        if args.max_batches_images < 0 or batch_idx < args.max_batches_images:
            cmap_rad = _cmap_or_fallback("HomeyerRainbow", fallback="viridis")
            cmap_ir = _cmap_or_fallback("cmc.batlow_r", fallback="viridis")
            cmap_lgt = _cmap_or_fallback("Reds", fallback="Reds")
            ir_min, ir_max = 200.0, 320.0
            l_min, l_max = 0.1, 50.0

            # PNG：每样本一张，取第一个有效帧
            for i in range(B):
                # choose first t where valid, else t=0
                t0 = 0
                if valid_mask is not None:
                    valid_t = torch.nonzero(valid_mask[i], as_tuple=False)
                    if len(valid_t) > 0:
                        t0 = int(valid_t[0].item())
                # Scale back to physical ranges for visualization
                gt_2d = (radar_gt[i, t0, 0] * (z_max - z_min) + z_min).cpu().numpy()
                pred_2d = (radar_pred[i, t0, 0] * (z_max - z_min) + z_min).cpu().numpy()
                sat_ir = sat_video[i, t0, 0] * (ir_max - ir_min) + ir_min  # [H, W]
                sat_ir_2d = sat_ir.detach().cpu().numpy()
                # Lightning channel assumed at index 10 (as in AE configs)
                if sat_video.shape[2] > 10:
                    sat_lgt = sat_video[i, t0, 10] * (l_max - l_min) + l_min
                    sat_lgt_2d = sat_lgt.detach().cpu().numpy()
                else:
                    # If lightning channel not available, fall back to zeros
                    sat_lgt_2d = np.zeros_like(sat_ir_2d)
                time_stem = ""
                radar_paths = batch.get("radar_paths")
                if radar_paths and i < len(radar_paths) and t0 < len(radar_paths[i]):
                    p = radar_paths[i][t0]
                    time_stem = os.path.splitext(os.path.basename(str(p)))[0].replace(os.sep, "_").replace(":", "_")
                out_name = f"{time_stem}.png" if time_stem else f"batch{batch_idx}_i{i}.png"
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

            # GIF：仅在 v2v 模式下，为多个样本保存时间序列视频
            if args.mode == "v2v" and HAS_IMAGEIO:
                max_samples = min(B, 8)
                video_frames = []
                for t in range(T_eff):
                    rows_t = []
                    for i in range(max_samples):
                        if valid_mask is not None and not valid_mask[i, t]:
                            continue
                        gt_2d = (
                            radar_gt[i, t, 0] * (z_max - z_min) + z_min
                        ).detach().cpu().numpy()
                        pred_2d = (
                            radar_pred[i, t, 0] * (z_max - z_min) + z_min
                        ).detach().cpu().numpy()
                        sat_ir = (
                            sat_video[i, t, 0] * (ir_max - ir_min) + ir_min
                        )  # [H, W]
                        sat_ir_2d = sat_ir.detach().cpu().numpy()
                        if sat_video.shape[2] > 10:
                            sat_lgt = sat_video[i, t, 10] * (l_max - l_min) + l_min
                            sat_lgt_2d = sat_lgt.detach().cpu().numpy()
                        else:
                            sat_lgt_2d = np.zeros_like(sat_ir_2d)

                        sat_ir_rgb = _apply_cmap(sat_ir_2d, cmap_ir, ir_min, ir_max)
                        sat_lgt_rgb = _apply_cmap(
                            sat_lgt_2d, cmap_lgt, l_min, l_max
                        )
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

    n_batches = len(loader)
    total = (
        n_batches
        if (args.max_batches_metrics < 0 or args.max_batches_images < 0)
        else min(n_batches, max(args.max_batches_metrics, args.max_batches_images))
    )
    pbar = tqdm(enumerate(loader), total=total, desc="test_sat2radar", unit="batch")
    for b_idx, batch in pbar:
        do_metrics = args.max_batches_metrics < 0 or b_idx < args.max_batches_metrics
        do_images = args.max_batches_images < 0 or b_idx < args.max_batches_images
        if not do_metrics and not do_images:
            break
        infer_batch(batch, b_idx)
        if mse_count > 0:
            pbar.set_postfix(mse=f"{mse_total / mse_count:.4f}", refresh=False)

    # Print metrics
    if mse_count > 0:
        print(f"MSE: {mse_total / mse_count:.6f}")
    if ssim_count > 0:
        print(f"SSIM: {ssim_total / ssim_count:.6f}")
    if fss_count > 0:
        print(f"FSS (original method): {fss_total / fss_count:.6f}")
    
    # Compute and compare pysteps FSS
    if PYSTEPS_AVAILABLE and pysteps_fss_objects:
        print("\n" + "="*60)
        print("FSS Comparison: Original vs pysteps")
        print("="*60)
        
        # Method 1: Overall average (accumulated)
        pysteps_fss_values = []
        for (thr, scale), fss_obj in pysteps_fss_objects.items():
            fss_val = fss_compute(fss_obj)
            pysteps_fss_values.append(fss_val)
        
        pysteps_avg_fss_accumulated = np.mean(pysteps_fss_values) if pysteps_fss_values else 0.0
        
        # Method 2: Average of per-pair averages
        if fss_pairwise_comparisons:
            orig_per_pair = [x[0] for x in fss_pairwise_comparisons]
            pysteps_per_pair = [x[1] for x in fss_pairwise_comparisons]
            pysteps_avg_fss_pairwise = np.mean(pysteps_per_pair)
            
            print(f"\nOverall Averages:")
            print(f"  Original method:        {fss_total / fss_count:.6f}")
            print(f"  pysteps (accumulated):  {pysteps_avg_fss_accumulated:.6f}")
            print(f"  pysteps (pairwise avg): {pysteps_avg_fss_pairwise:.6f}")
            print(f"  Difference (accum):     {abs(fss_total / fss_count - pysteps_avg_fss_accumulated):.6f}")
            print(f"  Difference (pairwise): {abs(fss_total / fss_count - pysteps_avg_fss_pairwise):.6f}")
            
            # Per-pair statistics
            differences = [abs(o - p) for o, p in fss_pairwise_comparisons]
            print(f"\nPer-pair Statistics ({len(fss_pairwise_comparisons)} pairs):")
            print(f"  Mean absolute difference: {np.mean(differences):.6f}")
            print(f"  Max absolute difference:  {np.max(differences):.6f}")
            print(f"  Min absolute difference:  {np.min(differences):.6f}")
            print(f"  Std of differences:      {np.std(differences):.6f}")
        
        # Per threshold-scale breakdown (using accumulated method)
        print("\nPer threshold-scale FSS breakdown (accumulated method):")
        print(f"{'Threshold':<12} {'Scale':<8} {'pysteps FSS':<15}")
        print("-" * 40)
        for (thr, scale), fss_obj in sorted(pysteps_fss_objects.items()):
            fss_val = fss_compute(fss_obj)
            print(f"{thr:<12.1f} {scale:<8} {fss_val:<15.6f}")
        
        print("="*60)
    elif not PYSTEPS_AVAILABLE:
        print("\n[INFO] pysteps not available. Skipping FSS comparison.")

    print(f"\nSaved sat2radar {args.mode} predictions to {args.out_dir}")


if __name__ == "__main__":
    main()

