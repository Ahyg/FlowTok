#!/usr/bin/env python3
# scripts/latent_utilization.py
"""Measure how 'used' the latent tokens are by a FlowTiTok AE.

Given a FlowTiTok AE ckpt + config + filelist pkl, encode N random samples
and report:
- Per-token activation variance (across batch) and count of near-dead tokens
- Per-token mean KL divergence from prior
- PCA effective rank of the flattened [B, N*D] latents

Outputs JSON + markdown to --out_dir.

Note: production training uses model_type="flowtitok" -> libs/flowtitok.py::FlowTiTok
(not modeling.tatitok.TATiTok). FlowTiTok.encode() returns
    (z_quantized, posteriors_DiagonalGaussianDistribution)
with posteriors.mean / posteriors.logvar of shape [B, D, 1, N].
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.dataset import SatelliteRadarNpyDataset
from libs.flowtitok import FlowTiTok
from utils.train_utils import _build_flowtitok_config


def _resolve_ckpt_state_dict_path(ckpt_path: str) -> Path:
    """Resolve a ckpt argument to the underlying pytorch_model.bin path.

    Accepts:
      * a directory `checkpoint-<slot>/` (uses `unwrapped_model/pytorch_model.bin`)
      * a directory containing `pytorch_model.bin` directly
      * a path to a `.bin` file
    """
    p = Path(ckpt_path)
    if p.is_file():
        return p
    cand1 = p / "unwrapped_model" / "pytorch_model.bin"
    if cand1.exists():
        return cand1
    cand2 = p / "pytorch_model.bin"
    if cand2.exists():
        return cand2
    raise FileNotFoundError(
        f"Could not locate pytorch_model.bin under {ckpt_path}. "
        f"Tried: {cand1}, {cand2}"
    )


def load_ae(config_path: str, ckpt_path: str, device: str = "cuda:0") -> FlowTiTok:
    cfg = OmegaConf.load(config_path)
    model_cfg = _build_flowtitok_config(cfg)
    model = FlowTiTok(model_cfg)
    sd_path = _resolve_ckpt_state_dict_path(ckpt_path)
    state = torch.load(sd_path, map_location="cpu")
    # Use strict=False to tolerate auxiliary keys (e.g. EMA-only stuff), matching
    # the loading pattern in scripts/test_flowtitok_ae.py.
    msg = model.load_state_dict(state, strict=False)
    if msg.missing_keys:
        print(f"[load_ae] missing keys: {len(msg.missing_keys)}")
    if msg.unexpected_keys:
        print(f"[load_ae] unexpected keys: {len(msg.unexpected_keys)}")
    model = model.to(device).eval()
    return model


@torch.no_grad()
def gather_latents(model: FlowTiTok, dataset, n_samples: int = 256,
                   batch_size: int = 16, device: str = "cuda:0"):
    """Encode random samples and return numpy arrays of mean, logvar, sampled latent.

    Output shapes (all [N_samp, D, N_tok]):
      - means
      - logvars
      - latents (one sample from each posterior)
    """
    n_samples = min(n_samples, len(dataset))
    indices = np.random.RandomState(42).choice(
        len(dataset), size=n_samples, replace=False)
    means, logvars, latents = [], [], []
    crop = getattr(model.encoder, "image_size", None)
    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start:start + batch_size]
        imgs = torch.stack([dataset[int(i)]["image"] for i in batch_idx]).to(device).float()
        # Crop / pad to the encoder's expected resolution if necessary.
        if crop is not None and imgs.shape[-1] != crop:
            imgs = imgs[..., :crop, :crop]
        z_q, posteriors = model.encode(imgs)
        # posteriors is a DiagonalGaussianDistribution; .mean / .logvar are
        # shape [B, D, 1, N]. Squeeze the singleton spatial dim.
        m = posteriors.mean.squeeze(2)        # [B, D, N]
        lv = posteriors.logvar.squeeze(2)     # [B, D, N]
        z = z_q.squeeze(2)                    # [B, D, N]
        means.append(m.detach().cpu())
        logvars.append(lv.detach().cpu())
        latents.append(z.detach().cpu())
    means = torch.cat(means, 0).numpy()
    logvars = torch.cat(logvars, 0).numpy()
    latents = torch.cat(latents, 0).numpy()
    return means, logvars, latents


def per_token_stats(means, logvars, latents):
    """Compute the spec §6.3 metrics on arrays of shape [B, D, N]."""
    B, D, N = means.shape
    # Per-token activation variance across batch, averaged over channel dim D.
    var_per_token = latents.var(axis=0).mean(axis=0)  # [N]
    mean_var = float(var_per_token.mean())
    near_dead = int((var_per_token < 0.01 * mean_var).sum()) if mean_var > 0 else int(N)
    # Per-token KL of N(mean, var) || N(0, 1) summed over D, averaged over batch.
    var = np.exp(logvars)
    kl = 0.5 * (means ** 2 + var - 1.0 - logvars)     # [B, D, N]
    kl_per_token = kl.sum(axis=1).mean(axis=0)        # [N]
    # PCA effective rank: entropy of normalized squared-singular-value distribution.
    flat = latents.reshape(B, -1)
    s = np.linalg.svd(flat - flat.mean(0, keepdims=True), compute_uv=False)
    s2 = s ** 2
    total = s2.sum()
    if total > 0:
        p = s2 / total
        eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    else:
        eff_rank = 0.0
    return {
        "num_tokens": int(N),
        "token_dim": int(D),
        "var_per_token_mean": float(mean_var),
        "var_per_token_min": float(var_per_token.min()),
        "var_per_token_max": float(var_per_token.max()),
        "near_dead_count": int(near_dead),
        "near_dead_frac": float(near_dead / max(N, 1)),
        "kl_per_token_mean": float(kl_per_token.mean()),
        "kl_per_token_min": float(kl_per_token.min()),
        "kl_per_token_max": float(kl_per_token.max()),
        "pca_effective_rank": float(eff_rank),
        "pca_max_possible": float(min(B, N * D)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--ckpt", required=True,
        help=("Path to a checkpoint-<slot>/ directory (uses "
              "unwrapped_model/pytorch_model.bin) or a pytorch_model.bin file."),
    )
    p.add_argument("--filelist", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--gpu", default=None, help="GPU id, e.g. 0")
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    ds_cfg = cfg.dataset.params
    ds = SatelliteRadarNpyDataset(
        base_dir=ds_cfg.get("data_dir"),
        mode=ds_cfg.get("mode", "satellite"),
        ir_band_indices=ds_cfg.get("ir_band_indices"),
        use_lightning=ds_cfg.get("use_lightning", True),
        filelist_path=args.filelist,
        filelist_split=args.split,
    )

    model = load_ae(args.config, args.ckpt, device=device)
    means, logvars, latents = gather_latents(
        model, ds, n_samples=args.n_samples,
        batch_size=args.batch_size, device=device)
    stats = per_token_stats(means, logvars, latents)
    stats["ckpt"] = str(args.ckpt)
    stats["config"] = str(args.config)
    stats["filelist"] = str(args.filelist)
    stats["split"] = str(args.split)
    stats["n_samples_used"] = int(means.shape[0])

    json_path = Path(args.out_dir) / "latent_utilization.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    md_lines = [
        f"# Latent utilization for {Path(args.ckpt).name}",
        "",
        f"- config: `{args.config}`",
        f"- filelist: `{args.filelist}` (split={args.split})",
        f"- num_tokens: {stats['num_tokens']}",
        f"- token_dim: {stats['token_dim']}",
        f"- n_samples used: {stats['n_samples_used']}",
        f"- mean activation variance: {stats['var_per_token_mean']:.4g}",
        (f"- near-dead tokens (variance < 1% of mean): "
         f"{stats['near_dead_count']} / {stats['num_tokens']} "
         f"({stats['near_dead_frac']:.1%})"),
        (f"- KL per token: mean {stats['kl_per_token_mean']:.4g}, "
         f"min {stats['kl_per_token_min']:.4g}, "
         f"max {stats['kl_per_token_max']:.4g}"),
        (f"- PCA effective rank: {stats['pca_effective_rank']:.1f} "
         f"(max possible: {stats['pca_max_possible']:.0f})"),
    ]
    md_path = Path(args.out_dir) / "latent_utilization.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Wrote {json_path} + {md_path}")


if __name__ == "__main__":
    main()
