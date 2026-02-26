import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmweather  # noqa: F401  # ensure HomeyerRainbow is registered
import open_clip

try:
    from pysteps.verification.spatialscores import fss_init, fss_accum, fss_compute
    PYSTEPS_AVAILABLE = True
except ImportError:
    PYSTEPS_AVAILABLE = False
    print("[WARN] pysteps not available; FSS will be skipped.")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from libs.flowtitok import FlowTiTok  # noqa: E402
from data.dataset import SatelliteRadarNpyDataset  # noqa: E402


def load_py_config(config_path: str):
    """
    Load a Python config file (e.g. configs/FlowTok-XL-Stage3.py) that defines get_config().
    """
    import importlib.util

    config_path = os.path.abspath(config_path)
    spec = importlib.util.spec_from_file_location("flowtok_config_radar", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_config"):
        raise AttributeError(f"{config_path} must define get_config()")
    return module.get_config()


def _parse_csv_numbers(value, cast=float):
    if value is None or value == "":
        return []
    return [cast(v.strip()) for v in value.split(",") if v.strip() != ""]


def _build_pysteps_fss_objects(thrs, scales):
    if not PYSTEPS_AVAILABLE or not thrs or not scales:
        return {}
    objs = {}
    for thr in thrs:
        for scale in scales:
            objs[(thr, scale)] = fss_init(thr=thr, scale=float(max(int(scale), 1)))
    return objs


def _pysteps_fss_accum(pysteps_fss_objects, pred_2d_np, tgt_2d_np):
    if not pysteps_fss_objects or pred_2d_np is None or tgt_2d_np is None:
        return
    for fss_obj in pysteps_fss_objects.values():
        fss_accum(fss_obj, pred_2d_np, tgt_2d_np)


def _pysteps_fss_compute_avg(pysteps_fss_objects):
    if not pysteps_fss_objects:
        return 0.0, []
    values = []
    for fss_obj in pysteps_fss_objects.values():
        v = fss_compute(fss_obj)
        values.append(float(v) if np.isfinite(v) else np.nan)
    return float(np.nanmean(values)) if values else 0.0, values


def build_radar_dataloader(
    data_root: str,
    filelist: str,
    split: str,
    batch_size: int,
    num_workers: int,
):
    """
    Build a dataloader over fused radar .npy frames (shape: [12, H, W], last channel = radar).

    - Uses SatelliteRadarNpyDataset with:
        * mode="radar"
        * filelist_path = dataset_filelist_i2i.pkl
        * filelist_split = split (e.g. "val")
    - The dataset internally applies `scale_radar_img`, so radar is scaled to [0, 1].
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split {split}, expected 'train', 'val' or 'test'.")

    filelist_path = os.path.abspath(filelist)
    if not os.path.isfile(filelist_path):
        raise FileNotFoundError(f"Filelist not found: {filelist_path}")

    dataset = SatelliteRadarNpyDataset(
        base_dir=os.path.abspath(data_root),
        mode="radar",
        ir_band_indices=None,
        use_lightning=False,
        filelist_path=filelist_path,
        filelist_split=split,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


def _cmap_or_fallback(name, fallback="viridis"):
    try:
        plt.get_cmap(name)
        return name
    except ValueError:
        print(f"[WARN] Colormap '{name}' not found, fallback to '{fallback}'")
        return fallback


def _apply_cmap(img2d, cmap_name, vmin, vmax):
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return cmap(norm(img2d))[..., :3]


def _save_composite(orig_2d, recon_2d, cmap_name, vmin, vmax, out_path):
    diff_2d = np.abs(orig_2d - recon_2d)
    diff_vmax = max(vmax - vmin, 1e-6)
    orig_rgb = _apply_cmap(orig_2d, cmap_name, vmin, vmax)
    recon_rgb = _apply_cmap(recon_2d, cmap_name, vmin, vmax)
    diff_rgb = _apply_cmap(diff_2d, cmap_name, 0.0, diff_vmax)
    composite = np.concatenate([orig_rgb, recon_rgb, diff_rgb], axis=1)
    plt.imsave(out_path, composite)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        "Test FlowTiTok image tokenizer on radar fused npy data (last channel as radar)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Python config file, e.g. configs/FlowTok-XL-Stage3.py (must define get_config()).",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Tokenizer checkpoint path, e.g. FlowTiTok_512.bin.",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Root folder of radar npy data, e.g. /mnt/ssd_1/yghu/Data/71_3m.",
    )
    parser.add_argument(
        "--filelist",
        required=True,
        help="Path to dataset_filelist_i2i.pkl under data_root.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Which split in dataset_filelist_i2i.pkl to evaluate on (default: val).",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to save reconstruction examples and metrics.json.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--max_batches_metrics",
        type=int,
        default=50,
        help="Max number of batches to use for metrics (-1 for all).",
    )
    parser.add_argument(
        "--max_batches_images",
        type=int,
        default=10,
        help="Max number of batches from which to save visualization images (-1 for all).",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="CUDA_VISIBLE_DEVICES override, e.g. '0'.",
    )
    parser.add_argument(
        "--use_text",
        action="store_true",
        help="If set, build simple CLIP text features from filenames as text_guidance.",
    )
    parser.add_argument(
        "--fss_thresholds",
        default="",
        help="Comma-separated thresholds for FSS (in dBZ). Empty string disables FSS.",
    )
    parser.add_argument(
        "--fss_scales",
        default="",
        help="Comma-separated window sizes (in pixels) for FSS. Empty string disables FSS.",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.makedirs(args.out_dir, exist_ok=True)

    config = load_py_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build radar dataloader.
    loader = build_radar_dataloader(
        data_root=args.data_root,
        filelist=args.filelist,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_samples = len(loader.dataset)
    num_batches = len(loader)
    print(f"Tokenizer eval dataset [{args.split.upper()}] (radar npy, last channel, scaled to [0,1]):")
    print(f"  Data root: {args.data_root}")
    print(f"  Filelist : {args.filelist}")
    print(f"  Samples  : {num_samples}")
    print(f"  Batches  : {num_batches}")
    print(f"  Batch size: {args.batch_size}")

    # Build FlowTiTok tokenizer and load weights.
    tokenizer = FlowTiTok(config)
    tokenizer.load_pretrained_weight(args.ckpt)
    tokenizer.eval()
    tokenizer.to(device)

    # Optional CLIP text encoder for text-guided reconstruction.
    clip_encoder = None
    clip_tokenizer = None
    if args.use_text:
        print("[INFO] Using CLIP text features as text_guidance (from filenames).")
        clip_encoder, _, _ = open_clip.create_model_and_transforms("ViT-L-14-336", pretrained="openai")
        # Only keep text branch.
        del clip_encoder.visual
        clip_tokenizer = open_clip.get_tokenizer("ViT-L-14-336")
        clip_encoder.transformer.batch_first = False
        clip_encoder.eval()
        clip_encoder.requires_grad_(False)
        clip_encoder.to(device)

    # Radar value range and colormap (same as validate_flowtitok_ae.py).
    z_min, z_max = 0.0, 60.0
    cmap_rad = _cmap_or_fallback("HomeyerRainbow", fallback="viridis")

    # Metric accumulators.
    mse_total = 0.0
    mse_count = 0
    ssim_total = 0.0
    ssim_count = 0

    # Text guidance shape.
    text_context_length = getattr(config.vq_model, "text_context_length", 77)
    text_embed_dim = getattr(config.vq_model, "text_embed_dim", 768)

    # FSS accumulators on physical dBZ fields.
    thrs = _parse_csv_numbers(args.fss_thresholds, cast=float)
    scales = _parse_csv_numbers(args.fss_scales, cast=int)
    pysteps_fss_objects = _build_pysteps_fss_objects(thrs, scales)

    for b_idx, batch in enumerate(loader):
        do_metrics = args.max_batches_metrics < 0 or b_idx < args.max_batches_metrics
        do_images = args.max_batches_images < 0 or b_idx < args.max_batches_images
        if not do_metrics and not do_images:
            break

        imgs_1ch = batch["image"].to(
            device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )  # [B, 1, H, W] in [0,1] after scaling

        # Ensure spatial size matches tokenizer pre-training (e.g. 512x512).
        crop_size = int(getattr(config.dataset, "crop_size", 512))
        if imgs_1ch.shape[-2] != crop_size or imgs_1ch.shape[-1] != crop_size:
            imgs_1ch = F.interpolate(
                imgs_1ch,
                size=(crop_size, crop_size),
                mode="bilinear",
                align_corners=False,
            )
        paths = batch["path"]  # list of strings

        # FlowTiTok was trained on 3-channel images; replicate radar channel to 3 channels.
        imgs = imgs_1ch.repeat(1, 3, 1, 1)  # [B, 3, H, W]

        # Build text_guidance.
        if args.use_text:
            # Prompt from filenames: "A radar reflectivity image from <filename>."
            texts = []
            for p in paths:
                fname = os.path.basename(p)
                texts.append(f"A radar reflectivity image from {fname}.")
            with torch.no_grad():
                text_tokens = clip_tokenizer(texts).to(device)
                cast_dtype = clip_encoder.transformer.get_cast_dtype()
                text_tokens = clip_encoder.token_embedding(text_tokens).to(cast_dtype)
                text_tokens = text_tokens + clip_encoder.positional_embedding.to(cast_dtype)
                text_tokens = text_tokens.permute(1, 0, 2)  # NLD -> LND
                text_tokens = clip_encoder.transformer(text_tokens, attn_mask=clip_encoder.attn_mask)
                text_tokens = text_tokens.permute(1, 0, 2)  # LND -> NLD
                text_tokens = clip_encoder.ln_final(text_tokens)  # [B, n_ctx, d_model]

                if text_tokens.shape[1] > text_context_length:
                    text_tokens = text_tokens[:, :text_context_length, :]
                elif text_tokens.shape[1] < text_context_length:
                    pad_len = text_context_length - text_tokens.shape[1]
                    pad = torch.zeros(
                        text_tokens.shape[0],
                        pad_len,
                        text_tokens.shape[2],
                        device=text_tokens.device,
                        dtype=text_tokens.dtype,
                    )
                    text_tokens = torch.cat([text_tokens, pad], dim=1)

                if text_tokens.shape[2] != text_embed_dim:
                    proj = torch.nn.Linear(text_tokens.shape[2], text_embed_dim, bias=False).to(device)
                    text_tokens = proj(text_tokens)

                text_guidance = text_tokens.to(imgs.dtype)
        else:
            # Zero text guidance (same as FlowTok AE without text).
            text_guidance = torch.zeros(
                imgs.shape[0],
                text_context_length,
                text_embed_dim,
                device=device,
                dtype=imgs.dtype,
            )

        # Forward through tokenizer.
        print(f"[batch {b_idx+1}/{num_batches}] imgs min/max:", float(imgs.min()), float(imgs.max()))
        recons, _ = tokenizer(imgs, text_guidance=text_guidance)  # [B, 3, H, W]
        print("  recons min/max before clamp:", float(recons.min()), float(recons.max()))
        recons = torch.clamp(recons, 0.0, 1.0)
        print("  recons min/max after clamp:", float(recons.min()), float(recons.max()))

        # For radar, take the first channel as the reconstructed radar field.
        imgs_np = imgs_1ch.detach().cpu().numpy()  # [B,1,H,W]
        recons_np = recons.detach().cpu().numpy()  # [B,3,H,W]

        # Metrics on normalized [0,1] single-channel radar.
        if do_metrics:
            # Use first channel of recon as radar reconstruction.
            recon_1ch = recons[:, 0:1]
            per_pixel_mse = (recon_1ch - imgs_1ch) ** 2  # [B,1,H,W]
            mse_total += per_pixel_mse.mean(dim=(1, 2, 3)).sum().item()
            mse_count += per_pixel_mse.shape[0]

        if do_images:
            for i in range(imgs_np.shape[0]):
                safe_name = os.path.basename(str(paths[i])).replace(os.sep, "_").replace(":", "_")
                orig_ch = imgs_np[i, 0] * (z_max - z_min) + z_min
                recon_ch = recons_np[i, 0] * (z_max - z_min) + z_min
                out_path = os.path.join(
                    args.out_dir, f"{b_idx:03d}_{i:02d}_{safe_name}_radar.png"
                )
                _save_composite(orig_ch, recon_ch, cmap_rad, z_min, z_max, out_path)

                if do_metrics:
                    ssim_total += ssim(imgs_np[i, 0], recons_np[i, 0], data_range=1.0)
                    ssim_count += 1
                    if pysteps_fss_objects:
                        _pysteps_fss_accum(
                            pysteps_fss_objects,
                            recon_ch.astype(np.float64),
                            orig_ch.astype(np.float64),
                        )

        if mse_count > 0 and (b_idx + 1) % 10 == 0:
            batch_mse = mse_total / mse_count
            batch_ssim = ssim_total / max(ssim_count, 1)
            batch_fss = None
            if pysteps_fss_objects:
                _, fss_vals = _pysteps_fss_compute_avg(pysteps_fss_objects)
                if fss_vals:
                    batch_fss = float(np.nanmean(fss_vals))
            if batch_fss is not None:
                print(
                    f"[batch {b_idx+1}/{num_batches}] "
                    f"MSE={batch_mse:.6f}, SSIM={batch_ssim:.6f}, Avg_FSS={batch_fss:.6f}"
                )
            else:
                print(
                    f"[batch {b_idx+1}/{num_batches}] "
                    f"MSE={batch_mse:.6f}, SSIM={batch_ssim:.6f}, Avg_FSS=N/A"
                )

    # Final metrics.
    metrics = {}
    if mse_count > 0:
        metrics["mse"] = mse_total / mse_count
        print(f"Final MSE: {metrics['mse']:.6f}")
    if ssim_count > 0:
        metrics["ssim"] = ssim_total / ssim_count
        print(f"Final SSIM: {metrics['ssim']:.6f}")

    avg_fss = 0.0
    all_fss_vals = []
    if pysteps_fss_objects:
        _, all_fss_vals = _pysteps_fss_compute_avg(pysteps_fss_objects)
        if all_fss_vals:
            avg_fss = float(np.nanmean(all_fss_vals))
        metrics["avg_fss"] = avg_fss
        metrics["fss_thresholds"] = thrs
        metrics["fss_scales"] = scales
        metrics["fss_method"] = "pysteps_accumulated"
        print(f"Final FSS (avg over thresholds/scales): {avg_fss:.6f}")
    else:
        metrics["fss_method"] = "none"

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved tokenizer metrics to {metrics_path}")
    print(f"Saved radar visualization image(s) to {args.out_dir}")


if __name__ == "__main__":
    main()

