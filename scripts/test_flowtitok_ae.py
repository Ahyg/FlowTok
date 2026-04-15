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
from utils.train_utils import (
    _build_flowtitok_config,
    create_clip_model,
    _encode_text_with_clip,
)


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


def _add_grid_lines(ax, color="black"):
    for frac in [0.25, 0.5, 0.75]:
        ax.plot([0, 1], [frac, frac], transform=ax.transAxes,
                linestyle='--', linewidth=0.8, color=color, alpha=0.5)
        ax.plot([frac, frac], [0, 1], transform=ax.transAxes,
                linestyle='--', linewidth=0.8, color=color, alpha=0.5)


def _compute_per_image_metrics(orig_2d, recon_2d, thrs, scales, thr_cat):
    """Return (mse, r2, fss, csi, pod, far) for a single [H,W] pair."""
    import itertools
    try:
        import pysteps.verification.spatialscores as _pvs
        import pysteps.verification.detcatscores as _pvdcat
        import pysteps.verification.detcontscores as _pvdcont
        from sklearn.metrics import r2_score as _r2
    except ImportError:
        return 0, 0, 0, 0, 0, 0
    o = np.asarray(orig_2d, dtype=np.float32)
    r = np.asarray(recon_2d, dtype=np.float32)
    mse = float(_pvdcont.det_cont_fct(r, o, scores='MSE')["MSE"].item())
    r2 = float(_r2(o.ravel(), r.ravel()))
    fss_vals = [_pvs.fss(r, o, thr=float(t), scale=int(s))
                for t, s in itertools.product(thrs, scales)]
    fss = float(np.nanmean(fss_vals))
    csi = float(_pvdcat.det_cat_fct(r, o, thr=float(thr_cat), scores='CSI', axis=None)["CSI"].item())
    pod = float(_pvdcat.det_cat_fct(r, o, thr=float(thr_cat), scores='POD', axis=None)["POD"].item())
    far = float(_pvdcat.det_cat_fct(r, o, thr=float(thr_cat), scores='FAR', axis=None)["FAR"].item())
    return mse, r2, fss, csi, pod, far


def _metrics_text(mse, r2, fss, csi, pod, far):
    return (f"MSE:{mse:.2f}, R\u00b2:{r2:.2f}\n"
            f"FSS:{fss:.2f}, CSI:{csi:.2f}\n"
            f"POD:{pod:.2f}, FAR:{far:.2f}")


def _save_composite(orig_2d, recon_2d, cmap_name, vmin, vmax, out_path,
                    title="", row_label="", mask_below=None,
                    fss_thrs=None, fss_scales=None, fss_cat_thr=None):
    """Save [Original | Reconstructed | |Diff|] with title, grid, metrics."""
    diff_2d = np.abs(orig_2d - recon_2d)
    diff_vmax = max(vmax - vmin, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), squeeze=False)
    for c, label in enumerate(["Original", "Reconstructed", "|Diff|"]):
        axes[0, c].set_title(label, fontsize=9)

    if mask_below is not None:
        axes[0, 0].imshow(np.ma.masked_less(orig_2d, mask_below),
                          cmap=cmap_name, vmin=vmin, vmax=vmax)
        axes[0, 1].imshow(np.ma.masked_less(recon_2d, mask_below),
                          cmap=cmap_name, vmin=vmin, vmax=vmax)
    else:
        axes[0, 0].imshow(orig_2d, cmap=cmap_name, vmin=vmin, vmax=vmax)
        axes[0, 1].imshow(recon_2d, cmap=cmap_name, vmin=vmin, vmax=vmax)
    axes[0, 2].imshow(diff_2d, cmap="hot", vmin=0, vmax=diff_vmax)

    if row_label:
        axes[0, 0].set_ylabel(row_label, fontsize=8)

    # Per-image metrics
    if fss_thrs is not None and fss_scales is not None and fss_cat_thr is not None:
        mse, r2, fss, csi, pod, far = _compute_per_image_metrics(
            orig_2d, recon_2d, fss_thrs, fss_scales, fss_cat_thr)
        axes[0, 1].text(
            0.01, 0.97, _metrics_text(mse, r2, fss, csi, pod, far),
            transform=axes[0, 1].transAxes, ha='left', va='top',
            fontsize=7, color='black', fontweight='bold',
        )

    for c in range(3):
        axes[0, c].axis("off")
        _add_grid_lines(axes[0, c])

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


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
    parser.add_argument("--filelist_path", default=None, help="Override filelist_path from config")
    parser.add_argument("--gpu", default=None, help="GPU id(s), e.g. 0 or 0,1")
    parser.add_argument("--fss_thresholds", default="0,5,10,15,20,25,30,35,40,45,50,55,60",
                        help="Comma-separated thresholds (dBZ for radar; auto-scaled for sat/lgt)")
    parser.add_argument("--fss_scales", default="1,2,3,4,5,6,7,8,9,10",
                        help="Comma-separated window sizes")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = OmegaConf.load(args.config)

    # Dataset
    ds_cfg = config.dataset.params
    filelist_path = args.filelist_path or ds_cfg.get("filelist_path")
    filelist_split = args.split or ds_cfg.get("filelist_split", "test")
    print("data_dir:", ds_cfg.get("data_dir"))
    print("filelist_path:", filelist_path)
    print("filelist_split:", filelist_split)
    eval_dataset = SatelliteRadarNpyDataset(
        base_dir=ds_cfg.get("data_dir"),
        years=ds_cfg.get("years", "").split(",") if ds_cfg.get("years") else None,
        mode=ds_cfg.get("mode", "satellite"),
        ir_band_indices=ds_cfg.get("ir_band_indices"),
        use_lightning=ds_cfg.get("use_lightning", True),
        filelist_path=filelist_path,
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
    model.load_state_dict(state, strict=False)
    model.eval()

    # Colormaps / ranges
    ir_min, ir_max = 200.0, 320.0
    l_min, l_max = 0.1, 50.0
    z_min, z_max = 0.0, 60.0
    cmap_ir = _cmap_or_fallback("cmc.batlow_r", fallback="viridis")
    cmap_lgt = _cmap_or_fallback("Reds", fallback="Reds")
    cmap_rad = _cmap_or_fallback("HomeyerRainbow", fallback="viridis")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── CLIP text guidance (must match training) ──────────────────────
    clip_encoder, clip_tokenizer = create_clip_model()
    clip_encoder = clip_encoder.to(device)
    mode = ds_cfg.get("mode", "satellite")

    def _build_text_guidance(paths, batch_size):
        """Build CLIP text guidance from file paths, matching training."""
        texts = []
        for p in paths:
            fname = os.path.basename(str(p))
            if mode == "radar":
                texts.append(f"A radar reflectivity image from {fname}.")
            else:
                texts.append(f"A multispectral satellite infrared and lightning image from {fname}.")
        return _encode_text_with_clip(texts, clip_tokenizer, clip_encoder, device)

    thrs_dbz = _parse_csv_numbers(args.fss_thresholds, cast=float)
    scales = _parse_csv_numbers(args.fss_scales, cast=int)
    fss_mode = ds_cfg.get("mode", "satellite")

    # ── Per-image FSS thresholds (physical units, matching training script) ──
    _rad_thrs = np.arange(0, 61, 5, dtype=np.float32)
    _fss_scales = np.arange(1, 11)
    _rad_thr35 = 35.0
    _ir_thrs = ir_min + (_rad_thrs / 60.0) * (ir_max - ir_min)
    _ir_thr35 = ir_min + (35.0 / 60.0) * (ir_max - ir_min)
    _lgt_thrs = l_min + (_rad_thrs / 60.0) * (l_max - l_min)
    _lgt_thr35 = l_min + (35.0 / 60.0) * (l_max - l_min)

    # ── Accumulated FSS thresholds (for global metrics) ──
    # Radar: physical dBZ; Sat/Lgt: normalized [0,1] (thr/60)
    thrs_ir = [t / 60.0 for t in thrs_dbz]
    thrs_lgt = [t / 60.0 for t in thrs_dbz]

    # pysteps accumulated FSS: one object per (thr, scale)
    if PYSTEPS_AVAILABLE:
        if fss_mode == "radar":
            # Radar: FSS on physical dBZ [0, 60]
            pysteps_fss_objects = _build_pysteps_fss_objects(thrs_dbz, scales)
            pysteps_fss_ir = {}
            pysteps_fss_lightning = {}
        else:
            # Satellite: FSS on normalized [0, 1] for IR and lightning channels
            pysteps_fss_objects = {}
            pysteps_fss_ir = _build_pysteps_fss_objects(thrs_ir, scales)
            pysteps_fss_lightning = _build_pysteps_fss_objects(thrs_lgt, scales) if ds_cfg.get("use_lightning", True) else {}
    else:
        pysteps_fss_objects = {}
        pysteps_fss_ir = {}
        pysteps_fss_lightning = {}

    mse_total = 0.0
    mse_count = 0
    ssim_total = 0.0
    ssim_count = 0
    # Per-channel accumulators (initialized on first batch once we know C)
    per_ch_mse_sum = None     # np.ndarray [C]
    per_ch_mse_count = None   # np.ndarray [C]
    per_ch_ssim_sum = None    # np.ndarray [C]
    per_ch_ssim_count = None  # np.ndarray [C]

    for b_idx, batch in enumerate(eval_loader):
        do_metrics = args.max_batches_metrics < 0 or b_idx < args.max_batches_metrics
        do_images = args.max_batches_images < 0 or b_idx < args.max_batches_images
        if not do_metrics and not do_images:
            break
        images = batch["image"].to(device)
        paths = batch.get("path", [f"sample_{b_idx}_{i}" for i in range(images.shape[0])])

        text_guidance = _build_text_guidance(paths, images.shape[0])

        recon, _ = model(images, text_guidance)
        recon = torch.clamp(recon, 0.0, 1.0)

        recon_cpu = recon.detach().cpu()
        images_cpu = images.detach().cpu()

        # Metrics (on normalized 0-1)
        if do_metrics:
            per_channel_mse = torch.mean((recon - images) ** 2, dim=(2, 3))  # [B, C]
            mse_total += per_channel_mse.sum().item()
            mse_count += per_channel_mse.numel()

            C_now = int(per_channel_mse.shape[1])
            if per_ch_mse_sum is None:
                per_ch_mse_sum = np.zeros(C_now, dtype=np.float64)
                per_ch_mse_count = np.zeros(C_now, dtype=np.int64)
                per_ch_ssim_sum = np.zeros(C_now, dtype=np.float64)
                per_ch_ssim_count = np.zeros(C_now, dtype=np.int64)
            per_ch_mse_sum += per_channel_mse.sum(dim=0).detach().cpu().numpy()
            per_ch_mse_count += int(per_channel_mse.shape[0])

            for i in range(images_cpu.shape[0]):
                img = images_cpu[i].numpy()
                rec = recon_cpu[i].numpy()

                # SSIM + FSS per channel
                for ch in range(img.shape[0]):
                    ssim_val = ssim(img[ch], rec[ch], data_range=1.0)
                    ssim_total += ssim_val
                    ssim_count += 1
                    if per_ch_ssim_sum is not None and ch < per_ch_ssim_sum.shape[0]:
                        per_ch_ssim_sum[ch] += ssim_val
                        per_ch_ssim_count[ch] += 1

                    # pysteps accumulated FSS
                    if fss_mode == "radar" and pysteps_fss_objects:
                        # Radar: convert to physical dBZ [0, 60] to match thrs_dbz
                        pred_np = (rec[ch] * (z_max - z_min) + z_min).astype(np.float64)
                        tgt_np = (img[ch] * (z_max - z_min) + z_min).astype(np.float64)
                        _pysteps_fss_accum(pysteps_fss_objects, pred_np, tgt_np)
                    elif fss_mode != "radar":
                        # Satellite: FSS on normalized [0, 1] with scaled thresholds
                        n_ir = len(ds_cfg.get("ir_band_indices", list(range(10))))
                        if ds_cfg.get("use_lightning", True) and ch == n_ir and pysteps_fss_lightning:
                            pred_np = rec[ch].astype(np.float64)
                            tgt_np = img[ch].astype(np.float64)
                            _pysteps_fss_accum(pysteps_fss_lightning, pred_np, tgt_np)
                        elif ch < n_ir and pysteps_fss_ir:
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
                    # Multi-channel satellite: composite all IR + lightning into one figure
                    n_ir = min(img.shape[0], 10)
                    has_lgt = ds_cfg.get("use_lightning", True) and img.shape[0] > n_ir
                    nrows = n_ir + (1 if has_lgt else 0)
                    fig, axes = plt.subplots(nrows, 3, figsize=(9, 3 * nrows), squeeze=False)
                    for c, label in enumerate(["Original", "Reconstructed", "|Diff|"]):
                        axes[0, c].set_title(label, fontsize=9)
                    for ch in range(n_ir):
                        band = 7 + ch
                        orig_ch = img[ch] * (ir_max - ir_min) + ir_min
                        recon_ch = rec[ch] * (ir_max - ir_min) + ir_min
                        diff_ch = np.abs(orig_ch - recon_ch)
                        axes[ch, 0].imshow(orig_ch, cmap=cmap_ir, vmin=ir_min, vmax=ir_max)
                        axes[ch, 1].imshow(recon_ch, cmap=cmap_ir, vmin=ir_min, vmax=ir_max)
                        axes[ch, 2].imshow(diff_ch, cmap="hot", vmin=0, vmax=ir_max - ir_min)
                        axes[ch, 0].set_ylabel(f"IR Band {band}", fontsize=8)
                        mse, r2, fss_v, csi, pod, far = _compute_per_image_metrics(
                            orig_ch, recon_ch, _ir_thrs, _fss_scales, _ir_thr35)
                        axes[ch, 1].text(
                            0.01, 0.97, _metrics_text(mse, r2, fss_v, csi, pod, far),
                            transform=axes[ch, 1].transAxes, ha='left', va='top',
                            fontsize=6, color='black', fontweight='bold')
                        for c in range(3):
                            axes[ch, c].axis("off")
                            _add_grid_lines(axes[ch, c])
                    if has_lgt:
                        row = n_ir
                        orig_lgt = img[n_ir] * (l_max - l_min) + l_min
                        recon_lgt = rec[n_ir] * (l_max - l_min) + l_min
                        diff_lgt = np.abs(orig_lgt - recon_lgt)
                        axes[row, 0].imshow(orig_lgt, cmap=cmap_lgt, vmin=0, vmax=l_max)
                        axes[row, 1].imshow(recon_lgt, cmap=cmap_lgt, vmin=0, vmax=l_max)
                        axes[row, 2].imshow(diff_lgt, cmap="hot", vmin=0, vmax=l_max)
                        axes[row, 0].set_ylabel("Lightning", fontsize=8)
                        mse, r2, fss_v, csi, pod, far = _compute_per_image_metrics(
                            orig_lgt, recon_lgt, _lgt_thrs, _fss_scales, _lgt_thr35)
                        axes[row, 1].text(
                            0.01, 0.97, _metrics_text(mse, r2, fss_v, csi, pod, far),
                            transform=axes[row, 1].transAxes, ha='left', va='top',
                            fontsize=6, color='black', fontweight='bold')
                        for c in range(3):
                            axes[row, c].axis("off")
                            _add_grid_lines(axes[row, c])
                    fig.suptitle(f"{safe_name}", fontsize=10)
                    fig.tight_layout()
                    out_path = os.path.join(
                        args.out_dir, f"{b_idx:03d}_{i:02d}_{safe_name}_sat.png")
                    fig.savefig(out_path, dpi=100)
                    plt.close(fig)

                else:  # radar
                    orig_ch = img[0] * (z_max - z_min) + z_min
                    recon_ch = rec[0] * (z_max - z_min) + z_min
                    out_path = os.path.join(
                        args.out_dir, f"{b_idx:03d}_{i:02d}_{safe_name}_radar.png"
                    )
                    _save_composite(
                        orig_ch, recon_ch, cmap_rad, z_min, z_max, out_path,
                        title=safe_name, mask_below=1.0,
                        fss_thrs=_rad_thrs, fss_scales=_fss_scales, fss_cat_thr=_rad_thr35,
                    )

    avg_mse = mse_total / max(mse_count, 1)
    avg_ssim = ssim_total / max(ssim_count, 1)
    # Combine FSS from radar or (IR + lightning) accumulators
    all_fss_vals = []
    ir_fss_vals = []
    lgt_fss_vals = []
    if pysteps_fss_objects:
        _, all_fss_vals = _pysteps_fss_compute_avg(pysteps_fss_objects)
    if pysteps_fss_ir:
        _, ir_fss_vals = _pysteps_fss_compute_avg(pysteps_fss_ir)
        all_fss_vals.extend(ir_fss_vals)
    if pysteps_fss_lightning:
        _, lgt_fss_vals = _pysteps_fss_compute_avg(pysteps_fss_lightning)
        all_fss_vals.extend(lgt_fss_vals)
    avg_fss = float(np.nanmean(all_fss_vals)) if all_fss_vals else 0.0

    # ── Per-channel + excl_lgt breakdown (satellite modes with lightning) ──
    has_lgt = (fss_mode != "radar") and bool(ds_cfg.get("use_lightning", True))
    per_channel_mse_avg = []
    per_channel_ssim_avg = []
    if per_ch_mse_sum is not None:
        per_channel_mse_avg = (per_ch_mse_sum / np.maximum(per_ch_mse_count, 1)).tolist()
        per_channel_ssim_avg = (per_ch_ssim_sum / np.maximum(per_ch_ssim_count, 1)).tolist()

    avg_mse_excl_lgt = None
    avg_ssim_excl_lgt = None
    avg_fss_excl_lgt = None
    if has_lgt and len(per_channel_mse_avg) >= 2:
        # Lightning is the last channel in satellite datasets when use_lightning=True
        avg_mse_excl_lgt = float(np.mean(per_channel_mse_avg[:-1]))
        avg_ssim_excl_lgt = float(np.mean(per_channel_ssim_avg[:-1]))
        if ir_fss_vals:
            avg_fss_excl_lgt = float(np.nanmean(ir_fss_vals))

    metrics = {
        "avg_mse": avg_mse,
        "avg_ssim": avg_ssim,
        "avg_fss": avg_fss,
        "avg_mse_excl_lgt": avg_mse_excl_lgt,
        "avg_ssim_excl_lgt": avg_ssim_excl_lgt,
        "avg_fss_excl_lgt": avg_fss_excl_lgt,
        "per_channel_mse": per_channel_mse_avg,
        "per_channel_ssim": per_channel_ssim_avg,
        "num_samples": int(mse_count),
        "fss_thresholds": thrs_dbz,
        "fss_scales": scales,
        "fss_method": "pysteps_accumulated" if (pysteps_fss_objects or pysteps_fss_ir or pysteps_fss_lightning) else "none",
    }
    print("Metrics:", metrics)
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
