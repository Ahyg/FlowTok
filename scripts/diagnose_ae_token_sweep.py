#!/usr/bin/env python3
# scripts/diagnose_ae_token_sweep.py
"""Generalized N-cell diagnostic for FlowTiTok AE checkpoints (token-count sweep).

Ports /tmp/diagnose_run1_vs_run3.py (2-cell, hard-coded) to a general
N-cell comparison driven by a `--cells` JSON list. For each of the N AEs the
script computes the spec §6.2 diagnostics on a small set of evenly-spaced
test samples:

  1. Low / high freq MSE split (Gaussian sigma=2)
  2. Per-patch 8x8 MSE map
  3. Sobel edge IoU @ thr=0.1                <-- HEADLINE metric (spec §7)
  4. Radial power spectrum
  5. High-freq MSE                            <-- HEADLINE metric (spec §7)

Outputs (under --out_dir):
  figs/recon_grid.png       — GT vs each cell (per-sample, IR ch0)
  figs/freq_split.png       — per-channel low/high freq MSE bars across cells
  figs/spatial_heatmap.png  — patch-MSE map per cell
  figs/edge_iou.png         — per-channel Sobel edge IoU + edge count bars
  figs/power_spectrum.png   — radial power spectrum overlay
  diagnose_summary.json     — all numeric results
  diagnose_summary.md       — markdown table aggregated over samples

CLI:
  python scripts/diagnose_ae_token_sweep.py \
    --cells '[{"name":"tok77","config":"...","ckpt":"..."}, ...]' \
    --filelist /path/to/test.pkl \
    --split test \
    --n_samples 8 \
    --out_dir /path/to/output

Loader uses the same FlowTiTok pattern as scripts/latent_utilization.py
(Task 7) — imported directly from there to stay DRY.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.dataset import SatelliteRadarNpyDataset
from libs.flowtitok import FlowTiTok
from utils.train_utils import (
    _build_flowtitok_config,
    _encode_text_with_clip,
    create_clip_model,
)
# Reuse Task 7 helpers to avoid duplicating the loader pattern.
from scripts.latent_utilization import (
    _resolve_ckpt_state_dict_path,
    load_ae,
)


# ----------------------------------------------------------------------
# Metric helpers (ported verbatim from /tmp/diagnose_run1_vs_run3.py).
# ----------------------------------------------------------------------
def gaussian_kernel_2d(size: int = 11, sigma: float = 2.0) -> torch.Tensor:
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    return torch.tensor(k, dtype=torch.float32)


def split_lowhigh(x: torch.Tensor, sigma: float = 2.0, ksize: int = 11):
    """x: [B,C,H,W] -> (low, high) freq components."""
    k = gaussian_kernel_2d(ksize, sigma).to(x.device)
    C = x.shape[1]
    k = k.expand(C, 1, ksize, ksize)
    pad = ksize // 2
    low = F.conv2d(x, k, padding=pad, groups=C)
    high = x - low
    return low, high


def patch_mse_map(x: torch.Tensor, y: torch.Tensor, patch: int = 8) -> torch.Tensor:
    """Per-patch MSE map -> [B, H/p, W/p]."""
    err = (x - y).pow(2).mean(dim=1, keepdim=True)  # [B,1,H,W]
    return F.avg_pool2d(err, patch, patch).squeeze(1)


def sobel_edge(x: torch.Tensor, thr: float = 0.1) -> torch.Tensor:
    """Binary edge map via Sobel magnitude > thr; output [B,C,H,W]."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    C = x.shape[1]
    kx = kx.expand(C, 1, 3, 3)
    ky = ky.expand(C, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-9)
    return (mag > thr).float()


def edge_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (a * b).sum(dim=(-1, -2))
    union = ((a + b).clamp(0, 1)).sum(dim=(-1, -2))
    return (inter + eps) / (union + eps)


def radial_power(x: torch.Tensor) -> np.ndarray:
    """x: [C,H,W] real -> mean radial power spectrum (1-D)."""
    X = torch.fft.fft2(x)
    P = (X.abs() ** 2).mean(0)
    H, W = P.shape
    cy, cx = H // 2, W // 2
    P = torch.fft.fftshift(P)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    rr = torch.sqrt(((yy - cy) ** 2 + (xx - cx) ** 2).float()).to(x.device)
    rr = rr.long()
    rmax = rr.max().item()
    out = torch.zeros(rmax + 1, device=x.device)
    cnt = torch.zeros(rmax + 1, device=x.device)
    out.scatter_add_(0, rr.flatten(), P.flatten())
    cnt.scatter_add_(0, rr.flatten(), torch.ones_like(P.flatten()))
    return (out / cnt.clamp_min(1)).cpu().numpy()


# ----------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------
@torch.no_grad()
def reconstruct(model: FlowTiTok, x: torch.Tensor,
                paths, mode: str,
                clip_encoder, clip_tokenizer, device) -> torch.Tensor:
    """Encode -> decode through `model`, matching test_flowtitok_ae.py text guidance."""
    texts = []
    for p in paths:
        fname = os.path.basename(str(p))
        if mode == "radar":
            texts.append(f"A radar reflectivity image from {fname}.")
        else:
            texts.append(f"A multispectral satellite infrared and lightning image from {fname}.")
    text_guidance = _encode_text_with_clip(texts, clip_tokenizer, clip_encoder, device)
    recon, _ = model(x, text_guidance)
    return torch.clamp(recon, 0.0, 1.0)


# ----------------------------------------------------------------------
# Plotting (each is gated to gracefully handle N>=1 cells)
# ----------------------------------------------------------------------
def _channel_labels(n_channels: int):
    if n_channels == 11:
        return [f"IR{i}" for i in range(10)] + ["lgt"]
    if n_channels == 1:
        return ["rad"]
    return [f"c{i}" for i in range(n_channels)]


def plot_recon_grid(x, recons, names, out_path, n_show=None):
    """Per-sample row: GT, each cell, then a |cell_last - cell_first| diff col."""
    n_show = min(n_show or x.shape[0], x.shape[0])
    n_cells = len(recons)
    n_cols = 1 + n_cells + (1 if n_cells >= 2 else 0)
    fig, axes = plt.subplots(n_show, n_cols, figsize=(3 * n_cols, 2.6 * n_show), squeeze=False)
    for i in range(n_show):
        gt_i = x[i, 0].cpu().numpy()
        axes[i, 0].imshow(gt_i, cmap="gray", vmin=0, vmax=1)
        if i == 0:
            axes[i, 0].set_title("GT (IR ch0)")
        for c, (r, nm) in enumerate(zip(recons, names)):
            axes[i, 1 + c].imshow(r[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            if i == 0:
                axes[i, 1 + c].set_title(nm)
        if n_cells >= 2:
            diff = np.abs(recons[-1][i, 0].cpu().numpy() - recons[0][i, 0].cpu().numpy())
            axes[i, -1].imshow(diff, cmap="hot", vmin=0, vmax=0.3)
            if i == 0:
                axes[i, -1].set_title(f"|{names[-1]}-{names[0]}|")
        for a in axes[i]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_freq_split(mse_low, mse_high, names, n_channels, out_path):
    """Grouped bars: per-channel low/high freq MSE across cells."""
    labels = _channel_labels(n_channels)
    n_cells = len(names)
    bar_w = 0.8 / n_cells
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    centers = np.arange(n_channels)
    for ci in range(n_cells):
        off = (ci - (n_cells - 1) / 2) * bar_w
        axes[0].bar(centers + off, mse_low[ci], bar_w, label=names[ci], color=f"C{ci}")
        axes[1].bar(centers + off, mse_high[ci], bar_w, label=names[ci], color=f"C{ci}")
    for ax, ttl in zip(axes, ["LOW-freq MSE per channel", "HIGH-freq MSE per channel"]):
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_title(ttl)
        ax.set_yscale("log")
        ax.legend()
    fig.suptitle("Frequency-split reconstruction MSE (lower=better). HIGH-freq is HEADLINE metric.")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_spatial_heatmap(x, pm_maps, names, out_path, n_show=None):
    """Per-sample row: GT, each cell's patch-MSE map."""
    n_show = min(n_show or x.shape[0], x.shape[0], 4)
    n_cells = len(pm_maps)
    n_cols = 1 + n_cells
    fig, axes = plt.subplots(n_show, n_cols, figsize=(3.2 * n_cols, 3.2 * n_show), squeeze=False)
    vmax = max(float(pm.max()) for pm in pm_maps)
    for i in range(n_show):
        axes[i, 0].imshow(x[i, 0].cpu().numpy(), cmap="gray")
        if i == 0:
            axes[i, 0].set_title("GT IR0")
        for c, (pm, nm) in enumerate(zip(pm_maps, names)):
            axes[i, 1 + c].imshow(pm[i], cmap="hot", vmin=0, vmax=vmax)
            if i == 0:
                axes[i, 1 + c].set_title(f"{nm} patch-MSE")
        for a in axes[i]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_edge_iou(iou_mean, edge_count_gt_mean, edge_count_mean, names, n_channels, out_path):
    """Per-channel grouped bars for IoU and edge counts."""
    labels = _channel_labels(n_channels)
    n_cells = len(names)
    bar_w = 0.8 / max(n_cells, 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    centers = np.arange(n_channels)
    for ci in range(n_cells):
        off = (ci - (n_cells - 1) / 2) * bar_w
        axes[0].bar(centers + off, iou_mean[ci], bar_w, label=names[ci], color=f"C{ci}")
    axes[0].set_xticks(centers)
    axes[0].set_xticklabels(labels, rotation=45)
    axes[0].set_title("Sobel edge IoU vs GT (higher=better) [HEADLINE]")
    axes[0].legend()
    # Edge count: include GT as a reference column
    bar_w2 = 0.8 / (n_cells + 1)
    axes[1].bar(centers - (n_cells / 2) * bar_w2, edge_count_gt_mean, bar_w2,
                label="GT", color="gray")
    for ci in range(n_cells):
        off = (ci + 1 - (n_cells / 2)) * bar_w2
        axes[1].bar(centers + off, edge_count_mean[ci], bar_w2,
                    label=names[ci], color=f"C{ci}")
    axes[1].set_xticks(centers)
    axes[1].set_xticklabels(labels, rotation=45)
    axes[1].set_title("Edge pixel count (>>GT = hallucinated edges)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_power_spectrum(rps_gt, rps_cells, names, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(rps_gt, label="GT", color="black", linewidth=2)
    for ci, (rps, nm) in enumerate(zip(rps_cells, names)):
        ax.semilogy(rps, label=nm, color=f"C{ci}")
    ax.set_xlabel("Radial frequency (wavenumber)")
    ax.set_ylabel("Power")
    ax.set_title("Radial power spectrum (avg over channels & samples)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="N-cell FlowTiTok AE diagnostic: freq split, patch MSE, "
                    "Sobel edge IoU, power spectrum, reconstruction grid.",
    )
    p.add_argument(
        "--cells", required=True,
        help=("JSON list of {name, config, ckpt} dicts, e.g. "
              "'[{\"name\":\"tok77\",\"config\":\"a.py\",\"ckpt\":\"ckpt/\"},"
              "{\"name\":\"tok128\",\"config\":\"b.py\",\"ckpt\":\"ckpt/\"}]'."),
    )
    p.add_argument("--filelist", required=True, help="Path to test/val pkl filelist.")
    p.add_argument("--split", default="test")
    p.add_argument("--n_samples", type=int, default=8,
                   help="Number of evenly-spaced samples to diagnose.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES override.")
    args = p.parse_args()

    try:
        cells = json.loads(args.cells)
        assert isinstance(cells, list) and len(cells) >= 1
        for c in cells:
            assert {"name", "config", "ckpt"}.issubset(c.keys()), \
                f"each cell must have name/config/ckpt, got {list(c.keys())}"
    except Exception as e:
        raise ValueError(f"--cells must be a JSON list of dicts: {e}")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset (use first cell's config for dataset params, same as Task 7) ----
    cfg0 = OmegaConf.load(cells[0]["config"])
    ds_cfg = cfg0.dataset.params
    mode = ds_cfg.get("mode", "satellite")
    ds = SatelliteRadarNpyDataset(
        base_dir=ds_cfg.get("data_dir"),
        mode=mode,
        ir_band_indices=ds_cfg.get("ir_band_indices"),
        use_lightning=ds_cfg.get("use_lightning", True),
        filelist_path=args.filelist,
        filelist_split=args.split,
    )
    print(f"[data] split={args.split} size={len(ds)}; using {args.n_samples} evenly-spaced samples")

    # ---- Pick N evenly-spaced samples ----
    n_samples = min(args.n_samples, len(ds))
    indices = np.linspace(0, len(ds) - 1, n_samples, dtype=int).tolist()
    xs, paths = [], []
    for i in indices:
        item = ds[int(i)]
        xs.append(item["image"])
        paths.append(item.get("path", f"sample_{i}"))
    x = torch.stack(xs).to(device).float()
    n_channels = int(x.shape[1])
    print(f"[data] sample tensor: {tuple(x.shape)}  channels={n_channels}")

    # ---- CLIP text encoder (shared across all cells) ----
    print("[clip] loading CLIP text encoder...")
    clip_encoder, clip_tokenizer = create_clip_model()
    clip_encoder = clip_encoder.to(device)

    # ---- Reconstruct through every cell ----
    recons = []
    cell_names = [c["name"] for c in cells]
    for cell in cells:
        name = cell["name"]
        print(f"[ae:{name}] loading config={cell['config']} ckpt={cell['ckpt']}")
        ae = load_ae(cell["config"], cell["ckpt"], device=device)
        with torch.no_grad():
            r = reconstruct(ae, x, paths, mode, clip_encoder, clip_tokenizer, device)
        recons.append(r)
        del ae
        torch.cuda.empty_cache()

    # ---- 1. Freq split per channel ----
    print("[metric] freq split MSE")
    low_gt, high_gt = split_lowhigh(x)
    mse_low = []   # list of [C] arrays per cell
    mse_high = []
    high_mse_global = []
    low_mse_global = []
    for r in recons:
        low_r, high_r = split_lowhigh(r)
        ml = (low_r - low_gt).pow(2).mean(dim=(0, 2, 3)).cpu().numpy()
        mh = (high_r - high_gt).pow(2).mean(dim=(0, 2, 3)).cpu().numpy()
        mse_low.append(ml)
        mse_high.append(mh)
        # Per-sample scalars (avg over channels), then mean/std over samples.
        low_mse_global.append((low_r - low_gt).pow(2).mean(dim=(1, 2, 3)).cpu().numpy())
        high_mse_global.append((high_r - high_gt).pow(2).mean(dim=(1, 2, 3)).cpu().numpy())

    # ---- 2. Patch MSE map ----
    print("[metric] patch MSE map")
    pm_maps = [patch_mse_map(r, x, patch=8).cpu().numpy() for r in recons]
    patch_mse_scalar = [pm.mean(axis=(1, 2)) for pm in pm_maps]  # per-sample scalar

    # ---- 3. Sobel edge IoU ----
    print("[metric] Sobel edge IoU")
    e_gt = sobel_edge(x, thr=0.1)
    iou_all = []           # list of [N, C] per cell
    edge_cnt = []          # list of [N, C] per cell
    edge_iou_scalar = []   # per-sample mean over channels
    for r in recons:
        e_r = sobel_edge(r, thr=0.1)
        iou_n_c = edge_iou(e_r, e_gt).cpu().numpy()  # [N, C]
        iou_all.append(iou_n_c)
        edge_cnt.append(e_r.sum(dim=(-1, -2)).cpu().numpy())
        edge_iou_scalar.append(iou_n_c.mean(axis=1))
    edge_count_gt = e_gt.sum(dim=(-1, -2)).cpu().numpy()

    # ---- 4. Radial power spectrum ----
    print("[metric] power spectrum")
    rps_gt = np.mean([radial_power(x[i, :n_channels]) for i in range(n_samples)], axis=0)
    rps_cells = []
    for r in recons:
        rps_cells.append(
            np.mean([radial_power(r[i, :n_channels]) for i in range(n_samples)], axis=0)
        )

    # ---- Figures ----
    print("[plot] writing figures")
    plot_recon_grid(x, recons, cell_names, figs_dir / "recon_grid.png", n_show=n_samples)
    plot_freq_split(mse_low, mse_high, cell_names, n_channels, figs_dir / "freq_split.png")
    plot_spatial_heatmap(x, pm_maps, cell_names, figs_dir / "spatial_heatmap.png")
    iou_mean_per_cell = [iou.mean(axis=0) for iou in iou_all]
    edge_cnt_mean_per_cell = [ec.mean(axis=0) for ec in edge_cnt]
    edge_cnt_gt_mean = edge_count_gt.mean(axis=0)
    plot_edge_iou(iou_mean_per_cell, edge_cnt_gt_mean, edge_cnt_mean_per_cell,
                  cell_names, n_channels, figs_dir / "edge_iou.png")
    plot_power_spectrum(rps_gt, rps_cells, cell_names, figs_dir / "power_spectrum.png")

    # ---- JSON dump of full numerics ----
    summary = {
        "cells": cells,
        "filelist": args.filelist,
        "split": args.split,
        "n_samples": n_samples,
        "sample_indices": indices,
        "n_channels": n_channels,
        "channel_labels": _channel_labels(n_channels),
        "per_cell": [],
    }
    for ci, name in enumerate(cell_names):
        summary["per_cell"].append({
            "name": name,
            "low_freq_mse_per_channel": mse_low[ci].tolist(),
            "high_freq_mse_per_channel": mse_high[ci].tolist(),
            "low_freq_mse_per_sample": low_mse_global[ci].tolist(),
            "high_freq_mse_per_sample": high_mse_global[ci].tolist(),
            "patch_mse_per_sample": patch_mse_scalar[ci].tolist(),
            "edge_iou_per_sample": edge_iou_scalar[ci].tolist(),
            "edge_iou_per_channel_mean": iou_mean_per_cell[ci].tolist(),
            "edge_count_per_channel_mean": edge_cnt_mean_per_cell[ci].tolist(),
            "radial_power": rps_cells[ci].tolist(),
        })
    summary["gt_radial_power"] = rps_gt.tolist()
    summary["gt_edge_count_per_channel_mean"] = edge_cnt_gt_mean.tolist()
    (out_dir / "diagnose_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # ---- Markdown table: rows=metric, columns=cells, values=mean±std ----
    def fmt_meanstd(arr):
        return f"{np.mean(arr):.4e} ± {np.std(arr):.2e}"

    def fmt_iou(arr):
        return f"{np.mean(arr):.4f} ± {np.std(arr):.4f}"

    md = [
        "# AE token-count diagnostic — multi-cell summary",
        "",
        f"- filelist: `{args.filelist}` (split=`{args.split}`)",
        f"- N samples: {n_samples}  (indices: {indices})",
        f"- N channels: {n_channels}  (labels: {_channel_labels(n_channels)})",
        f"- N cells: {len(cells)}  ({', '.join(cell_names)})",
        "",
        "## Headline metrics (per spec §7)",
        "",
        "Rows = metric, columns = cells; values are mean ± std over the N samples "
        "(averaged across channels). LOWER is better for MSE; HIGHER is better for edge IoU.",
        "",
        "| metric | " + " | ".join(cell_names) + " |",
        "|---|" + "---|" * len(cell_names),
        "| HIGH-freq MSE (HEADLINE) | "
        + " | ".join(fmt_meanstd(high_mse_global[ci]) for ci in range(len(cells)))
        + " |",
        "| Sobel edge IoU (HEADLINE) | "
        + " | ".join(fmt_iou(edge_iou_scalar[ci]) for ci in range(len(cells)))
        + " |",
        "| LOW-freq MSE | "
        + " | ".join(fmt_meanstd(low_mse_global[ci]) for ci in range(len(cells)))
        + " |",
        "| Patch-8 MSE | "
        + " | ".join(fmt_meanstd(patch_mse_scalar[ci]) for ci in range(len(cells)))
        + " |",
        "",
        "## Per-channel HIGH-freq MSE (mean over samples)",
        "",
        "| ch | " + " | ".join(cell_names) + " |",
        "|---|" + "---|" * len(cell_names),
    ]
    labels = _channel_labels(n_channels)
    for c in range(n_channels):
        md.append(
            f"| {labels[c]} | "
            + " | ".join(f"{mse_high[ci][c]:.4e}" for ci in range(len(cells)))
            + " |"
        )
    md += [
        "",
        "## Per-channel Sobel edge IoU (mean over samples)",
        "",
        "| ch | GT count | " + " | ".join(f"{nm}" for nm in cell_names) + " |",
        "|---|---|" + "---|" * len(cell_names),
    ]
    for c in range(n_channels):
        row = f"| {labels[c]} | {edge_cnt_gt_mean[c]:.0f} | "
        row += " | ".join(
            f"{iou_mean_per_cell[ci][c]:.3f} (cnt={edge_cnt_mean_per_cell[ci][c]:.0f})"
            for ci in range(len(cells))
        )
        row += " |"
        md.append(row)
    md += [
        "",
        "## Interpretation cheat-sheet",
        "",
        "- Lower HIGH-freq MSE = recon recovers fine-scale structure better.",
        "- Higher Sobel edge IoU = edges placed in the right pixels (not hallucinated).",
        "- If a cell has low LOW-freq MSE but high HIGH-freq MSE + low edge IoU →",
        "  it captures the gross pattern but hallucinates wrong-place details.",
        "- See `figs/recon_grid.png` for qualitative side-by-side.",
        "",
        "Full numerics in `diagnose_summary.json`.",
    ]
    (out_dir / "diagnose_summary.md").write_text("\n".join(md) + "\n")

    print("=" * 60)
    print(f"DONE — outputs in {out_dir}")
    for pth in sorted(out_dir.rglob("*")):
        if pth.is_file():
            print(f"  {pth}")


if __name__ == "__main__":
    main()
