import argparse
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmweather
import cmcrameri

try:
    from pysteps.verification.spatialscores import fss_init, fss_accum, fss_compute
    PYSTEPS_AVAILABLE = True
except ImportError:
    PYSTEPS_AVAILABLE = False
    print("[WARN] pysteps not available; FSS will be skipped.")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.dataset import SatelliteRadarNpyDataset
from libs.flowtitok import FlowTiTok
from utils.train_utils import _build_flowtitok_config


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


def _parse_csv_numbers(value, cast=float):
    if value is None or value == "":
        return []
    return [cast(v.strip()) for v in value.split(",") if v.strip() != ""]


def _build_pysteps_fss_objects(thrs, scales):
    """Build one FSS accumulator per (thr, scale) for pysteps accumulated FSS."""
    if not PYSTEPS_AVAILABLE or not thrs or not scales:
        return {}
    objs = {}
    for thr in thrs:
        for scale in scales:
            objs[(thr, scale)] = fss_init(thr=thr, scale=float(max(int(scale), 1)))
    return objs


def _pysteps_fss_accum(pysteps_fss_objects, pred_2d_np, tgt_2d_np):
    """Accumulate one forecast-observation pair into pysteps FSS objects."""
    if not pysteps_fss_objects or pred_2d_np is None or tgt_2d_np is None:
        return
    for fss_obj in pysteps_fss_objects.values():
        fss_accum(fss_obj, pred_2d_np, tgt_2d_np)


def _pysteps_fss_compute_avg(pysteps_fss_objects):
    """Compute FSS for each (thr, scale) and return the mean (and list of values)."""
    if not pysteps_fss_objects:
        return 0.0, []
    values = []
    for fss_obj in pysteps_fss_objects.values():
        v = fss_compute(fss_obj)
        values.append(float(v) if np.isfinite(v) else np.nan)
    return float(np.nanmean(values)) if values else 0.0, values


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)  # pytorch_model.bin
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_batches_metrics", type=int, default=-1,
                        help="Number of batches for metrics; -1 means all.")
    parser.add_argument("--max_batches_images", type=int, default=2,
                        help="Number of batches to save images.")
    parser.add_argument("--split", default="test", help="Override filelist_split, e.g. val/test")
    parser.add_argument("--gpu", default=None, help="GPU id(s), e.g. 0 or 0,1")
    parser.add_argument("--fss_thresholds", default="0", help="Comma-separated thresholds")
    parser.add_argument("--fss_scales", default="1", help="Comma-separated window sizes")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = OmegaConf.load(args.config)

    # Dataset (val split if filelist_path is used).
    ds_cfg = config.dataset.params
    filelist_split = args.split or ds_cfg.get("filelist_split", "test")
    print("data_dir:", ds_cfg.get("data_dir"))
    print("filelist_path:", ds_cfg.get("filelist_path"))
    print("filelist_split:", filelist_split)
    eval_dataset = SatelliteRadarNpyDataset(
        base_dir=ds_cfg.get("data_dir"),
        years=ds_cfg.get("years", "").split(",") if ds_cfg.get("years") else None,
        mode=ds_cfg.get("mode", "satellite"),
        ir_band_indices=ds_cfg.get("ir_band_indices"),
        use_lightning=ds_cfg.get("use_lightning", True),
        filelist_path=ds_cfg.get("filelist_path"),
        filelist_split=filelist_split,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=False,
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model_cfg = _build_flowtitok_config(config)
    model = FlowTiTok(model_cfg).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    # Colormaps / ranges
    ir_min, ir_max = 200.0, 320.0
    l_min, l_max = 0.1, 50.0
    z_min, z_max = 0.0, 60.0
    cmap_ir = _cmap_or_fallback("cmc.batlow_r", fallback="viridis")
    cmap_lgt = _cmap_or_fallback("Reds", fallback="Reds")
    cmap_rad = _cmap_or_fallback("HomeyerRainbow", fallback="viridis")

    os.makedirs(args.out_dir, exist_ok=True)

    thrs = _parse_csv_numbers(args.fss_thresholds, cast=float)
    scales = _parse_csv_numbers(args.fss_scales, cast=int)
    fss_mode = ds_cfg.get("mode", "satellite")
    if fss_mode != "radar" and thrs and max(thrs) > 1.0:
        print("[WARN] fss_thresholds > 1.0 for satellite mode; "
              "auto-scaling by 1/60 to match normalized [0,1].")
        thrs_sat = [t / 60.0 for t in thrs]
    else:
        thrs_sat = thrs

    # pysteps accumulated FSS: one object per (thr, scale)
    if PYSTEPS_AVAILABLE:
        if fss_mode == "radar":
            pysteps_fss_objects = _build_pysteps_fss_objects(thrs, scales)
            pysteps_fss_ir = {}
            pysteps_fss_lightning = {}
        else:
            pysteps_fss_objects = {}
            pysteps_fss_ir = _build_pysteps_fss_objects(thrs_sat, scales)
            pysteps_fss_lightning = _build_pysteps_fss_objects(thrs, scales) if ds_cfg.get("use_lightning", True) else {}
    else:
        pysteps_fss_objects = {}
        pysteps_fss_ir = {}
        pysteps_fss_lightning = {}

    mse_total = 0.0
    mse_count = 0
    ssim_total = 0.0
    ssim_count = 0

    for b_idx, batch in enumerate(eval_loader):
        do_metrics = args.max_batches_metrics < 0 or b_idx < args.max_batches_metrics
        do_images = args.max_batches_images < 0 or b_idx < args.max_batches_images
        if not do_metrics and not do_images:
            break
        images = batch["image"].to(device)
        paths = batch.get("path", [f"sample_{b_idx}_{i}" for i in range(images.shape[0])])

        # FlowTiTok uses zero text guidance.
        text_guidance = torch.zeros(
            images.shape[0],
            config.model.vq_model.get("text_context_length", 77),
            config.model.vq_model.get("text_embed_dim", 768),
            device=device,
            dtype=images.dtype,
        )

        recon, _ = model(images, text_guidance)
        recon = torch.clamp(recon, 0.0, 1.0)

        recon_cpu = recon.detach().cpu()
        images_cpu = images.detach().cpu()

        # Metrics (on normalized 0-1)
        if do_metrics:
            per_channel_mse = torch.mean((recon - images) ** 2, dim=(2, 3))
            mse_total += per_channel_mse.sum().item()
            mse_count += per_channel_mse.numel()

            for i in range(images_cpu.shape[0]):
                img = images_cpu[i].numpy()
                rec = recon_cpu[i].numpy()

                # SSIM + FSS per channel
                for ch in range(img.shape[0]):
                    ssim_total += ssim(img[ch], rec[ch], data_range=1.0)
                    ssim_count += 1

                    # pysteps accumulated FSS (physical units for threshold)
                    if ds_cfg.get("mode", "satellite") == "radar" and pysteps_fss_objects:
                        pred_np = (rec[ch] * (z_max - z_min) + z_min).astype(np.float64)
                        tgt_np = (img[ch] * (z_max - z_min) + z_min).astype(np.float64)
                        _pysteps_fss_accum(pysteps_fss_objects, pred_np, tgt_np)
                    elif ds_cfg.get("mode", "satellite") != "radar":
                        if ds_cfg.get("use_lightning", True) and ch == 10 and pysteps_fss_lightning:
                            pred_np = (rec[ch] * (l_max - l_min) + l_min).astype(np.float64)
                            tgt_np = (img[ch] * (l_max - l_min) + l_min).astype(np.float64)
                            _pysteps_fss_accum(pysteps_fss_lightning, pred_np, tgt_np)
                        elif ch < 10 and pysteps_fss_ir:
                            pred_np = rec[ch].astype(np.float64)
                            tgt_np = img[ch].astype(np.float64)
                            _pysteps_fss_accum(pysteps_fss_ir, pred_np, tgt_np)

        # Save images (can be limited separately)
        if do_images:
            for i in range(images_cpu.shape[0]):
                safe_name = os.path.basename(str(paths[i])).replace(os.sep, "_").replace(":", "_")
                img = images_cpu[i].numpy()
                rec = recon_cpu[i].numpy()

                if ds_cfg.get("mode", "satellite") == "satellite":
                    # IR channels 0-9 -> bands 7-16
                    ir = img[:10] * (ir_max - ir_min) + ir_min
                    for ch in range(10):
                        band = 7 + ch
                        orig_ch = ir[ch]
                        recon_ch = rec[ch] * (ir_max - ir_min) + ir_min
                        out_path = os.path.join(
                            args.out_dir, f"{b_idx:03d}_{i:02d}_{safe_name}_ir{band:02d}.png"
                        )
                        _save_composite(orig_ch, recon_ch, cmap_ir, ir_min, ir_max, out_path)

                    if ds_cfg.get("use_lightning", True) and img.shape[0] > 10:
                        orig_ch = img[10] * (l_max - l_min) + l_min
                        recon_ch = rec[10] * (l_max - l_min) + l_min
                        out_path = os.path.join(
                            args.out_dir, f"{b_idx:03d}_{i:02d}_{safe_name}_lightning.png"
                        )
                        _save_composite(orig_ch, recon_ch, cmap_lgt, 0.0, l_max, out_path)

                else:  # radar
                    orig_ch = img[0] * (z_max - z_min) + z_min
                    recon_ch = rec[0] * (z_max - z_min) + z_min
                    out_path = os.path.join(
                        args.out_dir, f"{b_idx:03d}_{i:02d}_{safe_name}_radar.png"
                    )
                    _save_composite(orig_ch, recon_ch, cmap_rad, z_min, z_max, out_path)

    avg_mse = mse_total / max(mse_count, 1)
    avg_ssim = ssim_total / max(ssim_count, 1)
    # Combine FSS from radar or (IR + lightning) accumulators
    all_fss_vals = []
    if pysteps_fss_objects:
        _, all_fss_vals = _pysteps_fss_compute_avg(pysteps_fss_objects)
    if pysteps_fss_ir:
        _, v = _pysteps_fss_compute_avg(pysteps_fss_ir)
        all_fss_vals.extend(v)
    if pysteps_fss_lightning:
        _, v = _pysteps_fss_compute_avg(pysteps_fss_lightning)
        all_fss_vals.extend(v)
    avg_fss = float(np.nanmean(all_fss_vals)) if all_fss_vals else 0.0

    metrics = {
        "avg_mse": avg_mse,
        "avg_ssim": avg_ssim,
        "avg_fss": avg_fss,
        "num_samples": int(mse_count),
        "fss_thresholds": thrs,
        "fss_scales": scales,
        "fss_method": "pysteps_accumulated" if (pysteps_fss_objects or pysteps_fss_ir or pysteps_fss_lightning) else "none",
    }
    print("Metrics:", metrics)
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
