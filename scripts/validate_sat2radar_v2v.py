import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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

# Import pysteps FSS helpers (accumulated method), same as test_sat2radar_v2v.py
try:
    from pysteps.verification.spatialscores import (
        fss_init,
        fss_accum,
        fss_compute,
    )

    PYSTEPS_AVAILABLE = True
except ImportError:
    PYSTEPS_AVAILABLE = False
    print("[WARN] pysteps not available. FSS computation will be skipped.")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml_collections import ConfigDict  # noqa: E402

from data.dataset import SatelliteRadarNpyDataset, collate_sat2radar_v2v  # noqa: E402
from diffusion.flow_matching import ODEEulerFlowMatchingSolver  # noqa: E402
from libs.flowtitok import FlowTiTok  # noqa: E402
from libs.adapters import AdapterIn, AdapterOut  # noqa: E402
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


def _ckpt_tag(ckpt_path: str) -> str:
    base = os.path.basename(os.path.normpath(ckpt_path))
    if base.endswith(".ckpt"):
        return base[: -len(".ckpt")]
    return base


def _resolve_ckpt_paths(args) -> List[str]:
    if getattr(args, "ckpts", None):
        return [p.strip() for p in args.ckpts.split(",") if p.strip()]
    if getattr(args, "ckpt_root", None) and getattr(args, "steps", None):
        root = args.ckpt_root.rstrip("/")
        steps = [s.strip() for s in args.steps.split(",") if s.strip()]
        return [os.path.join(root, f"{s}.ckpt") for s in steps]
    if args.ckpt:
        return [args.ckpt]
    raise ValueError(
        "Provide one of: --ckpt, or --ckpts 'path1,path2', or --ckpt_root + --steps."
    )


def _load_clip_text_encoder(device: torch.device):
    """Match train/test: optional OPENCLIP_LOCAL_CKPT for offline Gadi runs."""
    clip_model_name = "ViT-L-14-336"
    local_clip_ckpt = os.environ.get("OPENCLIP_LOCAL_CKPT", None)
    try:
        if local_clip_ckpt and os.path.isfile(local_clip_ckpt):
            print(
                f"[INFO] Loading open_clip '{clip_model_name}' from local checkpoint: {local_clip_ckpt}"
            )
            clip_encoder, _, _ = open_clip.create_model_and_transforms(
                clip_model_name, pretrained=None
            )
            state_dict = torch.load(local_clip_ckpt, map_location="cpu")
            missing, unexpected = clip_encoder.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(
                    f"[INFO] open_clip loaded with missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
                )
        else:
            clip_encoder, _, _ = open_clip.create_model_and_transforms(
                clip_model_name, pretrained="openai"
            )
        del clip_encoder.visual
        clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
        clip_encoder.transformer.batch_first = False
        clip_encoder.eval()
        clip_encoder.requires_grad_(False)
        clip_encoder.to(device)
        return clip_encoder, clip_tokenizer
    except Exception as e:
        print(
            f"[WARN] open_clip not available, FlowTiTok decoder will run without text guidance. Error: {e}"
        )
        return None, None


def _load_optional_adapters(
    config, ckpt_path: str, device: torch.device
) -> Tuple[Optional[Any], Optional[Any]]:
    adapter_in_satellite = None
    adapter_out = None
    adapter_in_sat_cfg = getattr(config, "adapter_in_satellite", None)
    if adapter_in_sat_cfg is None:
        adapter_in_sat_cfg = getattr(config, "adapter_in", None)
    if adapter_in_sat_cfg and adapter_in_sat_cfg.get("enabled", False):
        adapter_in_satellite = AdapterIn(
            in_channels=int(
                adapter_in_sat_cfg.get("in_channels", getattr(config, "sat_in_channels", 3))
            ),
            out_channels=int(getattr(config, "sat_in_channels", 3)),
            mid_channels=int(adapter_in_sat_cfg.get("mid_channels", 32)),
            num_blocks=int(adapter_in_sat_cfg.get("num_blocks", 3)),
        ).to(device)
        adapter_in_satellite_path = os.path.join(ckpt_path, "adapter_in_satellite.pth")
        legacy_adapter_in_path = os.path.join(ckpt_path, "adapter_in.pth")
        if os.path.isfile(adapter_in_satellite_path):
            adapter_in_satellite.load_state_dict(
                torch.load(adapter_in_satellite_path, map_location="cpu")
            )
            adapter_in_satellite.eval()
            print(f"[INFO] Loaded adapter_in_satellite from {adapter_in_satellite_path}")
        elif os.path.isfile(legacy_adapter_in_path):
            adapter_in_satellite.load_state_dict(
                torch.load(legacy_adapter_in_path, map_location="cpu")
            )
            adapter_in_satellite.eval()
            print(f"[INFO] Loaded legacy adapter_in from {legacy_adapter_in_path}")
        else:
            print(
                "[WARN] adapter_in_satellite is enabled in config but checkpoint file not found: "
                f"{adapter_in_satellite_path} (or legacy {legacy_adapter_in_path})"
            )
            adapter_in_satellite = None
    if getattr(config, "adapter_out", None) and config.adapter_out.get("enabled", False):
        adapter_out = AdapterOut(
            in_channels=int(getattr(config, "radar_out_channels", 3)),
            out_channels=1,
            mid_channels=int(config.adapter_out.get("mid_channels", 16)),
            num_blocks=int(config.adapter_out.get("num_blocks", 2)),
        ).to(device)
        adapter_out_path = os.path.join(ckpt_path, "adapter_out.pth")
        if os.path.isfile(adapter_out_path):
            adapter_out.load_state_dict(torch.load(adapter_out_path, map_location="cpu"))
            adapter_out.eval()
            print(f"[INFO] Loaded adapter_out from {adapter_out_path}")
        else:
            print(
                f"[WARN] adapter_out is enabled in config but checkpoint file not found: {adapter_out_path}"
            )
            adapter_out = None
    return adapter_in_satellite, adapter_out


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
    parser = argparse.ArgumentParser(
        description=(
            "Validate / visualize sat2radar I2I or V2V: optional MSE over [0,1] radar, "
            "multi-checkpoint metrics JSON, and PNG/GIF outputs (with CLIP text guidance like train/test)."
        )
    )
    parser.add_argument("--config", required=True, help="Python config file, e.g. configs/Sat2Radar-Video-XL.py")
    ckpt_group = parser.add_mutually_exclusive_group(required=False)
    ckpt_group.add_argument(
        "--ckpt",
        default=None,
        help="Single TrainState checkpoint dir, e.g. .../100000.ckpt",
    )
    ckpt_group.add_argument(
        "--ckpts",
        default=None,
        help="Comma-separated checkpoint dirs, e.g. a/160000.ckpt,b/320000.ckpt",
    )
    parser.add_argument(
        "--ckpt_root",
        default=None,
        help="With --steps: load ckpt_root/{step}.ckpt for each step (alternative to --ckpts).",
    )
    parser.add_argument(
        "--steps",
        default=None,
        help="Comma-separated step ids used with --ckpt_root, e.g. 160000,320000",
    )
    parser.add_argument("--out_dir", required=True, help="Directory to save visualizations and metrics JSON")
    parser.add_argument("--split", default="val", help="Dataset split in filelist (train/val/test)")
    parser.add_argument("--mode", default="i2i", choices=["i2i", "v2v"], help="Visualization mode: i2i or v2v")
    parser.add_argument(
        "--max_batches",
        type=int,
        default=10,
        help="(Deprecated) Max number of batches (-1 for all). Used for both metrics and visualization when enabled.",
    )
    parser.add_argument(
        "--max_batches_metrics",
        type=int,
        default=None,
        help="Max number of batches for metrics (-1 for all). If None, falls back to --max_batches.",
    )
    parser.add_argument(
        "--max_batches_images",
        type=int,
        default=None,
        help="Max number of batches for visualization (PNG/GIF) (-1 for all). If None, falls back to --max_batches.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES override, e.g. '0'")
    parser.add_argument("--filelist_path", default=None, help="Optional override for config.dataset.filelist_path")
    parser.add_argument(
        "--metrics_json",
        default=None,
        help="Path to write per-ckpt MSE summary JSON. Default: <out_dir>/validate_metrics.json",
    )
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="Only compute metrics (MSE); skip PNG/GIF output.",
    )
    # FSS (accumulated method, same as test_sat2radar_v2v.py)
    parser.add_argument(
        "--fss_thresholds",
        default="0,5,10,15,20,25,30,35,40,45,50,55,60",
        help="Comma-separated FSS thresholds (dBZ).",
    )
    parser.add_argument(
        "--fss_scales",
        default="1,2,3,4,5,6,7,8,9,10",
        help="Comma-separated FSS scales (pixels).",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.max_batches_metrics is None:
        args.max_batches_metrics = args.max_batches
    if args.max_batches_images is None:
        args.max_batches_images = args.max_batches

    config = load_py_config(args.config)
    if args.filelist_path:
        config.dataset.filelist_path = args.filelist_path
        print(f"[INFO] Override filelist_path: {args.filelist_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_paths = _resolve_ckpt_paths(args)
    print(f"[INFO] Checkpoint(s) to validate: {len(ckpt_paths)} -> {ckpt_paths}")

    loader = build_eval_dataloader(config, split=args.split, batch_size=args.batch_size, mode=args.mode)

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

    clip_encoder, clip_tokenizer = _load_clip_text_encoder(device)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = args.metrics_json or os.path.join(args.out_dir, "validate_metrics.json")
    results: List[Dict[str, Any]] = []

    num_latent_tokens = config.vq_model.num_latent_tokens

    for ckpt_path in ckpt_paths:
        tag = _ckpt_tag(ckpt_path)
        if len(ckpt_paths) == 1:
            effective_out_dir = args.out_dir
        else:
            effective_out_dir = os.path.join(args.out_dir, tag)
        if not args.no_viz:
            os.makedirs(effective_out_dir, exist_ok=True)

        print(f"\n{'='*60}\n[CKPT] {ckpt_path}  (tag={tag})\n{'='*60}")
        train_state = flow_utils.initialize_train_state(config, device)
        train_state.load(ckpt_path)
        nnet_ema = train_state.nnet_ema.to(device)
        nnet_ema.eval()
        train_step = train_state.step
        if torch.is_tensor(train_step):
            train_step = int(train_step.item())
        else:
            train_step = int(train_step)

        adapter_in_satellite, adapter_out = _load_optional_adapters(config, ckpt_path, device)

        mse_sum = 0.0
        mse_cnt = 0

        # FSS (accumulated method, same as scripts/test_sat2radar_v2v.py)
        z_min, z_max = 0.0, 60.0

        def _parse_csv_numbers(value, cast=float):
            if value is None or value == "":
                return []
            return [cast(v.strip()) for v in value.split(",") if v.strip() != ""]

        thrs = _parse_csv_numbers(args.fss_thresholds, cast=float)
        scales = _parse_csv_numbers(args.fss_scales, cast=int)

        pysteps_fss_objects = {}
        fss_accum_cnt = 0
        if PYSTEPS_AVAILABLE:
            for thr in thrs:
                for scale in scales:
                    pysteps_fss_objects[(thr, scale)] = fss_init(thr=thr, scale=float(scale))

        def build_condition_tokens_from_sat_video(sat_video):
            use_sat_lgt_tokens = getattr(config, "cond_use_sat_lightning_tokens", False)
            if not use_sat_lgt_tokens:
                return encode_video_with_autoencoder(
                    sat_autoencoder,
                    sat_video,
                    config.vq_model.scale_factor,
                    adapter_in=adapter_in_satellite,
                )

            sat_ir_video = sat_video[:, :, :3, :, :]
            lgt_slice = sat_video[:, :, -1:, :, :]
            lgt_video = lgt_slice.repeat(1, 1, 3, 1, 1)

            sat_ir_tokens = encode_video_with_autoencoder(
                sat_autoencoder, sat_ir_video, config.vq_model.scale_factor, adapter_in=None
            )
            lgt_tokens = encode_video_with_autoencoder(
                sat_autoencoder, lgt_video, config.vq_model.scale_factor, adapter_in=None
            )

            fusion = getattr(config, "cond_token_fusion", "mean")
            if fusion == "sum":
                return sat_ir_tokens + lgt_tokens
            return 0.5 * (sat_ir_tokens + lgt_tokens)

        def infer_and_save(batch, batch_idx: int, do_metrics: bool, do_images: bool):
            nonlocal mse_sum, mse_cnt, fss_accum_cnt

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
            sat_tokens = build_condition_tokens_from_sat_video(sat_video)

            # 采样阶段与训练 / 测试保持一致：
            # - 默认：使用 textVAE，对 sat_tokens 做编码得到 x0 作为 flow 起点；
            # - 可选（config.use_text_vae_encoder == False）：直接使用 sat_tokens 作为起点。
            use_text_vae_encoder = getattr(config, "use_text_vae_encoder", True)
            if use_text_vae_encoder:
                x0, _, _ = nnet_ema(sat_tokens, text_encoder=True)
            else:
                x0 = sat_tokens

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
            assert L % num_latent_tokens == 0, (
                "Sequence length must be multiple of num_latent_tokens"
            )
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
                            if isinstance(paths_i, (list, tuple)) and t < len(
                                paths_i
                            ):
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
                        text_tokens = text_tokens + clip_encoder.positional_embedding.to(
                            cast_dtype
                        )
                        text_tokens = text_tokens.permute(1, 0, 2)  # NLD -> LND
                        text_tokens = clip_encoder.transformer(
                            text_tokens, attn_mask=clip_encoder.attn_mask
                        )
                        text_tokens = text_tokens.permute(1, 0, 2)  # LND -> NLD
                        text_tokens = clip_encoder.ln_final(text_tokens)
                        text_guidance = text_tokens
                    except Exception as e:
                        print(
                            f"[WARN] Failed to build CLIP text guidance: {e}"
                        )
                        text_guidance = None

            radar_pred = radar_autoencoder.decode_tokens(
                z / config.vq_model.scale_factor, text_guidance=text_guidance
            )  # [B*T_eff, C_out, H_pred, W_pred]
            if adapter_out is not None:
                _, _, _, H_gt0, W_gt0 = radar_video_gt[:, :T_eff].shape
                radar_pred = adapter_out(radar_pred, out_size=(H_gt0, W_gt0))
            else:
                # 雷达物理上是单通道，这里只取第一个通道
                radar_pred = radar_pred[:, 0:1, ...]

            radar_pred = torch.clamp(radar_pred, 0.0, 1.0)
            radar_pred = radar_pred.view(
                B, T_eff, 1, radar_pred.shape[-2], radar_pred.shape[-1]
            )
            radar_gt = torch.clamp(
                radar_video_gt[:, :T_eff], 0.0, 1.0
            )  # [B, T_eff, 1, H_gt, W_gt]

            # 如果预测雷达的分辨率与原始 GT 不一致，则 resize 回 GT 尺寸。
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

            # MSE + FSS on normalized radar [0,1] (FSS uses physical dBZ range).
            # 定义在可视化之前，确保 do_images=False 时 FSS 仍可用。
            z_min, z_max = 0.0, 60.0
            radar_pred_cpu = radar_pred.detach().cpu()
            radar_gt_cpu = radar_gt.detach().cpu()
            if do_metrics:
                for b in range(B):
                    for t in range(T_eff):
                        if valid_mask is not None and not valid_mask[b, t]:
                            continue
                        gt = radar_gt_cpu[b, t, 0].numpy()
                        pred = radar_pred_cpu[b, t, 0].numpy()

                        mse_sum += float(np.mean((pred - gt) ** 2))
                        mse_cnt += 1

                        if PYSTEPS_AVAILABLE and pysteps_fss_objects:
                            # Convert normalized -> physical dBZ range for FSS
                            pred_np = pred * (z_max - z_min) + z_min
                            tgt_np = gt * (z_max - z_min) + z_min
                            for key in pysteps_fss_objects.keys():
                                fss_accum(
                                    pysteps_fss_objects[key], pred_np, tgt_np
                                )
                            fss_accum_cnt += 1

            if not do_images:
                return

            # For each sample, save first few frames (or all for small T).
            max_frames_to_show = 4 if args.mode == "v2v" else 1

            ir_min, ir_max = 200.0, 320.0
            l_min, l_max = 0.1, 50.0
            cmap_rad = _cmap_or_fallback("HomeyerRainbow", fallback="viridis")
            cmap_ir = _cmap_or_fallback("cmc.batlow_r", fallback="viridis")
            cmap_lgt = _cmap_or_fallback("Reds", fallback="Reds")
            ir_band_indices = getattr(config.dataset, "ir_band_indices", None)
            lgt_channel_idx = (
                len(ir_band_indices)
                if getattr(config.dataset, "use_lightning", False)
                and ir_band_indices is not None
                else None
            )

            # 单帧 PNG：保存 [sat_IR | sat_lightning | radar_gt | radar_pred]
            for i in range(B):
                frames = []
                for t in range(T_eff):
                    if valid_mask is not None and not valid_mask[i, t]:
                        continue
                    if len(frames) >= max_frames_to_show:
                        break

                    gt_2d = (
                        radar_gt[i, t, 0] * (z_max - z_min) + z_min
                    ).detach().cpu().numpy()
                    pred_2d = (
                        radar_pred[i, t, 0] * (z_max - z_min) + z_min
                    ).detach().cpu().numpy()

                    sat_ir = sat_video[i, t, 0] * (ir_max - ir_min) + ir_min
                    sat_ir_2d = sat_ir.detach().cpu().numpy()
                    if (
                        lgt_channel_idx is not None
                        and sat_video.shape[2] > lgt_channel_idx
                    ):
                        sat_lgt = (
                            sat_video[i, t, lgt_channel_idx]
                            * (l_max - l_min)
                            + l_min
                        )
                        sat_lgt_2d = sat_lgt.detach().cpu().numpy()
                    else:
                        sat_lgt_2d = np.zeros_like(sat_ir_2d)

                    time_stem = ""
                    radar_paths = batch.get("radar_paths")
                    if (
                        radar_paths
                        and i < len(radar_paths)
                        and t < len(radar_paths[i])
                    ):
                        p = radar_paths[i][t]
                        time_stem = os.path.splitext(
                            os.path.basename(str(p))
                        )[0].replace(os.sep, "_").replace(":", "_")

                    out_name = (
                        f"{time_stem}.png"
                        if time_stem
                        else f"batch{batch_idx}_i{i}_t{t:02d}.png"
                    )
                    out_path = os.path.join(effective_out_dir, out_name)
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

                        gt_2d = (
                            radar_gt[i, t, 0] * (z_max - z_min) + z_min
                        ).detach().cpu().numpy()
                        pred_2d = (
                            radar_pred[i, t, 0] * (z_max - z_min) + z_min
                        ).detach().cpu().numpy()
                        sat_ir = sat_video[i, t, 0] * (ir_max - ir_min) + ir_min
                        sat_ir_2d = sat_ir.detach().cpu().numpy()
                        if (
                            lgt_channel_idx is not None
                            and sat_video.shape[2] > lgt_channel_idx
                        ):
                            sat_lgt = (
                                sat_video[i, t, lgt_channel_idx]
                                * (l_max - l_min)
                                + l_min
                            )
                            sat_lgt_2d = sat_lgt.detach().cpu().numpy()
                        else:
                            sat_lgt_2d = np.zeros_like(sat_ir_2d)

                        sat_ir_rgb = _apply_cmap(
                            sat_ir_2d, cmap_ir, ir_min, ir_max
                        )
                        sat_lgt_rgb = _apply_cmap(
                            sat_lgt_2d, cmap_lgt, l_min, l_max
                        )
                        gt_rgb = _apply_cmap(gt_2d, cmap_rad, z_min, z_max)
                        pred_rgb = _apply_cmap(
                            pred_2d, cmap_rad, z_min, z_max
                        )

                        row_t = np.concatenate(
                            [sat_ir_rgb, sat_lgt_rgb, gt_rgb, pred_rgb], axis=1
                        )
                        rows_t.append(row_t)

                    if not rows_t:
                        continue
                    frame = np.concatenate(rows_t, axis=0)
                    frame_uint8 = (
                        np.clip(frame, 0.0, 1.0) * 255
                    ).astype(np.uint8)
                    video_frames.append(frame_uint8)

                if video_frames:
                    gif_name = f"batch{batch_idx}_v2v.gif"
                    gif_path = os.path.join(effective_out_dir, gif_name)
                    try:
                        imageio.mimsave(gif_path, video_frames, fps=4)
                    except Exception as e:
                        print(f"[WARN] Failed to save GIF {gif_path}: {e}")

        for b_idx, batch in enumerate(loader):
            do_metrics = args.max_batches_metrics < 0 or b_idx < args.max_batches_metrics
            do_images = (not args.no_viz) and (
                args.max_batches_images < 0 or b_idx < args.max_batches_images
            )
            if not do_metrics and not do_images:
                break
            infer_and_save(batch, b_idx, do_metrics=do_metrics, do_images=do_images)

        mse_mean = (mse_sum / mse_cnt) if mse_cnt > 0 else None

        fss_accum_avg = None
        if PYSTEPS_AVAILABLE and pysteps_fss_objects and fss_accum_cnt > 0:
            pysteps_fss_values = [
                fss_compute(pysteps_fss_objects[key]) for key in pysteps_fss_objects
            ]
            fss_accum_avg = float(np.mean(pysteps_fss_values)) if pysteps_fss_values else None

        print(
            f"[METRICS] ckpt={ckpt_path} train_step={train_step} "
            f"mse_mean_01={mse_mean} (n_frames={mse_cnt}) "
            f"fss_accum_avg={fss_accum_avg}"
        )
        results.append(
            {
                "ckpt_path": ckpt_path,
                "tag": tag,
                "train_step": train_step,
                "out_dir": effective_out_dir,
                "mse_mean_radar_0_1": mse_mean,
                "mse_frame_count": mse_cnt,
                "fss_accumulated_avg": fss_accum_avg,
                "split": args.split,
                "mode": args.mode,
                "max_batches": args.max_batches,
                "max_batches_metrics": args.max_batches_metrics,
                "max_batches_images": args.max_batches_images,
                "batch_size": args.batch_size,
            }
        )
        if not args.no_viz:
            print(f"[OK] Saved sat2radar {args.mode} visualizations under {effective_out_dir}")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Wrote metrics for {len(results)} checkpoint(s) to {metrics_path}")


if __name__ == "__main__":
    main()

