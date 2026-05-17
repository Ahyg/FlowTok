"""Latent-alignment + reconstruction eval for one joint-tokenizer group.

Loads a (sat AE, radar AE) pair, encodes time-paired test frames, and reports
the metrics that decide the experiment (design §5):

  cross-modal alignment   : mean index-wise cosine(S_k,R_k); linear CKA;
                            per-index linear-fit normalized error (16->16)
  reconstruction guardrail: per-modality recon MSE / PSNR
  capacity                : near-dead token count, PCA effective rank

Run per group; the orchestrator diffs Group A vs Group B JSON.
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.dataset import SatelliteRadarNpyDataset, collate_sat2radar_v2v
from libs.flowtitok import FlowTiTok
from utils.train_utils import (
    _build_flowtitok_config, create_clip_model, _encode_text_with_clip,
)
from scripts.latent_utilization import _resolve_ckpt_state_dict_path


def _build_ae(joint_cfg, in_ch, out_ch, ckpt, device):
    cfg = copy.deepcopy(joint_cfg)
    cfg.model.vq_model.in_channels = in_ch
    cfg.model.vq_model.out_channels = out_ch
    model = FlowTiTok(_build_flowtitok_config(cfg))
    sd = torch.load(_resolve_ckpt_state_dict_path(ckpt), map_location="cpu")
    msg = model.load_state_dict(sd, strict=False)
    if len(msg.missing_keys) > 20 or len(msg.unexpected_keys) > 20:
        raise RuntimeError(
            f"AE load looks wrong (missing={len(msg.missing_keys)}, "
            f"unexpected={len(msg.unexpected_keys)}) — arch/ckpt mismatch?")
    return model.to(device).eval()


def _linear_cka(X, Y):
    """Linear CKA between two [n, f] centered matrices."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    return float(hsic / (np.linalg.norm(X.T @ X, ord="fro")
                         * np.linalg.norm(Y.T @ Y, ord="fro") + 1e-12))


def _per_index_linfit_err(S, R):
    """Mean normalized residual of a per-token-index linear fit S_k->R_k.

    S,R: [n, D, N]. For each token index k fit D->D least squares (n>>D, well
    posed). Lower = the two spaces are linearly close per index = the v2v flow
    target sits closer to its conditioning.
    """
    n, D, N = S.shape
    errs = []
    for k in range(N):
        x, y = S[:, :, k], R[:, :, k]                  # [n, D]
        w, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = ((y - x @ w) ** 2).sum()
        denom = ((y - y.mean(0, keepdims=True)) ** 2).sum() + 1e-12
        errs.append(resid / denom)
    return float(np.mean(errs))


def _capacity(latents):
    """near-dead token count + PCA effective rank. latents: [n, D, N]."""
    var_per_tok = latents.var(0).mean(0)               # [N]
    mv = float(var_per_tok.mean())
    near_dead = int((var_per_tok < 0.01 * mv).sum())
    flat = latents.reshape(latents.shape[0], -1)
    s = np.linalg.svd(flat - flat.mean(0, keepdims=True), compute_uv=False)
    p = (s ** 2) / (s ** 2).sum()
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    return {"near_dead_tokens": near_dead, "n_tokens": int(latents.shape[2]),
            "pca_effective_rank": eff_rank}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint_config", required=True)
    ap.add_argument("--sat_ckpt", required=True)
    ap.add_argument("--radar_ckpt", required=True)
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_samples", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    jc = OmegaConf.load(args.joint_config)
    crop = int(jc.dataset.preprocessing.crop_size)

    sat_ae = _build_ae(jc, 11, 11, args.sat_ckpt, device)
    radar_ae = _build_ae(jc, 1, 1, args.radar_ckpt, device)
    clip_enc, clip_tok = create_clip_model()
    clip_enc = clip_enc.to(device).eval()

    ds = SatelliteRadarNpyDataset(
        filelist_path=args.filelist, filelist_split=args.split,
        mode="sat2radar_v2v", num_frames=1, frame_stride=1,
        use_lightning=True, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_sat2radar_v2v)

    S, R = [], []
    sat_se = radar_se = 0.0
    nb = 0
    seen = 0
    for batch in loader:
        if seen >= args.n_samples:
            break
        sat = batch["sat_video"][:, 0].to(device).float()
        radar = batch["radar_video"][:, 0].to(device).float()
        if sat.shape[-1] != crop:
            sat = F.interpolate(sat, (crop, crop), mode="bilinear",
                                align_corners=False)
            radar = F.interpolate(radar, (crop, crop), mode="bilinear",
                                  align_corners=False)
        sp = [os.path.basename(str(p[0] if isinstance(p, (list, tuple)) else p))
              for p in batch.get("sat_paths", [])]
        rp = [os.path.basename(str(p[0] if isinstance(p, (list, tuple)) else p))
              for p in batch.get("radar_paths", [])]
        tg_s = _encode_text_with_clip(
            [f"A multispectral satellite infrared and lightning image from {x}."
             for x in sp], clip_tok, clip_enc, device)
        tg_r = _encode_text_with_clip(
            [f"A radar reflectivity image from {x}." for x in rp],
            clip_tok, clip_enc, device)
        sat_rec, sp_post = sat_ae(sat, tg_s)
        radar_rec, rp_post = radar_ae(radar, tg_r)
        sat_se += F.mse_loss(sat_rec, sat).item()
        radar_se += F.mse_loss(radar_rec, radar).item()
        nb += 1
        S.append(sp_post.mean.squeeze(2).cpu().numpy())   # [B, D, N]
        R.append(rp_post.mean.squeeze(2).cpu().numpy())
        seen += sat.shape[0]

    S = np.concatenate(S, 0)[:args.n_samples]
    R = np.concatenate(R, 0)[:args.n_samples]
    cos = (np.sum(S * R, 1)
           / (np.linalg.norm(S, axis=1) * np.linalg.norm(R, axis=1) + 1e-12))
    res = {
        "tag": args.tag,
        "n_samples": int(S.shape[0]),
        "alignment": {
            "index_wise_cosine_mean": float(cos.mean()),
            "index_wise_cosine_std": float(cos.std()),
            "linear_cka": _linear_cka(S.reshape(S.shape[0], -1),
                                      R.reshape(R.shape[0], -1)),
            "per_index_linfit_norm_err": _per_index_linfit_err(S, R),
        },
        "reconstruction": {
            "sat_mse": sat_se / max(nb, 1),
            "radar_mse": radar_se / max(nb, 1),
            "sat_psnr": float(10 * np.log10(1.0 / (sat_se / max(nb, 1) + 1e-12))),
            "radar_psnr": float(10 * np.log10(1.0 / (radar_se / max(nb, 1) + 1e-12))),
        },
        "capacity": {"sat": _capacity(S), "radar": _capacity(R)},
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    jp = Path(args.out_dir) / f"align_{args.tag}.json"
    jp.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"[eval_joint_tokenizer] wrote {jp}")


if __name__ == "__main__":
    main()
