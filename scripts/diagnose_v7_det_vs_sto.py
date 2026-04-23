"""
Diagnose train/inference mismatch in v7 tokenconcat: cond_projector trained with
reparameterization (z = mu + eps*std) but inference uses mu only.

For a fixed batch of sat/lgt inputs, run the full FlowTok pipeline with:
  - mode A: deterministic=True  (inference-as-shipped, x0 = mu)
  - mode B: deterministic=False (train-matched, x0 = reparameterized z, 4 seeds)
Then decode to radar and compare MSE / MAE vs GT.

Also reports mu/logvar statistics so we can tell whether std is meaningfully
non-zero (otherwise A and B are equivalent regardless).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml_collections import ConfigDict  # noqa: E402

from diffusion.flow_matching import ODEEulerFlowMatchingSolver  # noqa: E402
from libs.flowtitok import FlowTiTok  # noqa: E402
from libs.cond_projector import CondTokenProjector  # noqa: E402
import flow_utils  # noqa: E402

from scripts.test_sat2radar_v2v import (  # noqa: E402
    load_py_config, _ae_config, encode_video_with_autoencoder,
    build_eval_dataloader,
)


@torch.no_grad()
def run_once(sat_tokens_ir, lgt_tokens, nnet_ema, cond_projector, radar_ae,
             config, deterministic, num_latent_tokens, device):
    cond_projector.eval()
    z_cond = cond_projector(sat_tokens_ir, lgt_tokens, deterministic=deterministic)
    mu = cond_projector.last_mu
    logvar = cond_projector.last_logvar

    x0 = z_cond
    if config.nnet.model_args.noising_type != "none":
        x0 = x0 + torch.randn_like(x0) * config.sample.noise_scale

    guidance_scale = config.sample.scale
    solver = ODEEulerFlowMatchingSolver(
        nnet_ema, step_size_type="step_in_dsigma", guidance_scale=guidance_scale,
    )
    z, _ = solver.sample(
        x_T=x0, batch_size=x0.shape[0],
        sample_steps=config.sample.sample_steps,
        unconditional_guidance_scale=guidance_scale,
        has_null_indicator=guidance_scale > 1.0,
    )  # [B, T*L, C]

    B_, L_tot, C_tok = z.shape
    T_ = L_tot // num_latent_tokens
    z = z.view(B_, T_, num_latent_tokens, C_tok)
    z = z.view(B_ * T_, num_latent_tokens, C_tok).permute(0, 2, 1).unsqueeze(2)
    radar = radar_ae.decode_tokens(z / config.vq_model.scale_factor, text_guidance=None)
    radar = radar[:, 0:1, ...]  # single channel
    _, _, H, W = radar.shape
    radar = radar.view(B_, T_, 1, H, W)
    return torch.clamp(radar, 0.0, 1.0), mu, logvar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--n_stochastic", type=int, default=4)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_py_config(args.config)
    assert getattr(config, "cond_token_fusion", "mean") == "concat", (
        "This diagnostic only applies to fusion='concat' runs."
    )
    # Force fixed seed so the stochastic samples differ only by eps inside projector.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Dataloader on the (tiny) train set, no augmentation. build_eval_dataloader
    # ignores augment flags (eval mode) and picks num_frames from config for v2v
    # or forces T=1 for i2i. v7 tiny is i2i (num_frames=1).
    mode = "i2i" if int(config.dataset.get("num_frames", 16)) == 1 else "v2v"
    loader = build_eval_dataloader(
        config, split=args.split, batch_size=args.batch_size, mode=mode,
    )

    # Build nnet / nnet_ema + cond_projector. Avoid train_state.load() — it
    # also restores optimizer/lr_scheduler whose param groups no longer match
    # (the v7 training added cond_projector params; we don't). Only the nnet
    # weights are needed for inference.
    train_state = flow_utils.initialize_train_state(config, device)
    nnet_ema = train_state.nnet_ema.to(device).eval()
    nnet_ema_weights = torch.load(os.path.join(args.ckpt, "nnet_ema.pth"), map_location="cpu")
    nnet_ema.load_state_dict(nnet_ema_weights)
    nnet_ema.requires_grad_(False)
    print(f"[INFO] Loaded nnet_ema from {args.ckpt}/nnet_ema.pth")

    _tvae = config.nnet.model_args.textVAE
    cond_projector = CondTokenProjector(
        token_dim=config.nnet.model_args.channels,
        num_latent_tokens=config.vq_model.num_latent_tokens,
        num_blocks=getattr(_tvae, "num_blocks", 6),
        num_attention_heads=getattr(_tvae, "num_attention_heads", 4),
        d_ff=getattr(_tvae, "hidden_dim", 256),
        dropout=getattr(_tvae, "dropout_prob", 0.1),
    ).to(device)
    cp_path = os.path.join(args.ckpt, "cond_projector.pth")
    cond_projector.load_state_dict(torch.load(cp_path, map_location="cpu"))
    cond_projector.eval()
    cond_projector.requires_grad_(False)
    print(f"[INFO] Loaded cond_projector from {cp_path}")

    # Autoencoders.
    sat_in = getattr(config, "sat_in_channels", 3)
    sat_out = getattr(config, "sat_out_channels", 3)
    radar_in = getattr(config, "radar_in_channels", 3)
    radar_out = getattr(config, "radar_out_channels", 3)
    sat_ae = FlowTiTok(_ae_config(config, sat_in, sat_out)).to(device).eval()
    sat_ae.load_state_dict(torch.load(config.sat_tokenizer_checkpoint, map_location="cpu"), strict=False)
    sat_ae.requires_grad_(False)
    radar_ae = FlowTiTok(_ae_config(config, radar_in, radar_out)).to(device).eval()
    radar_ae.load_state_dict(torch.load(config.radar_tokenizer_checkpoint, map_location="cpu"), strict=False)
    radar_ae.requires_grad_(False)

    num_latent_tokens = int(config.vq_model.num_latent_tokens)

    # Single batch diagnostic — enough for this comparison.
    batch = next(iter(loader))
    sat_video = batch["sat_video"].to(device)         # [B, T, C_sat, H, W]
    radar_gt = batch["radar_video"].to(device)        # [B, T, 1, H, W]
    B, T, _, H_gt, W_gt = radar_gt.shape
    print(f"[INFO] batch: sat_video={tuple(sat_video.shape)} radar_gt={tuple(radar_gt.shape)}")

    sat_ir_video = sat_video[:, :, :3, :, :]
    lgt_video = sat_video[:, :, -1:, :, :].repeat(1, 1, 3, 1, 1)
    sat_ir_tokens = encode_video_with_autoencoder(
        sat_ae, sat_ir_video, config.vq_model.scale_factor, adapter_in=None,
    )
    lgt_tokens = encode_video_with_autoencoder(
        sat_ae, lgt_video, config.vq_model.scale_factor, adapter_in=None,
    )

    # ---- Mode A: deterministic = True (= mu) ----
    torch.manual_seed(args.seed)
    radar_det, mu, logvar = run_once(
        sat_ir_tokens, lgt_tokens, nnet_ema, cond_projector, radar_ae,
        config, deterministic=True, num_latent_tokens=num_latent_tokens,
        device=device,
    )
    std = torch.exp(0.5 * logvar)
    mu_abs = mu.abs()
    print(f"\n[STATS] mu: mean={mu.mean():.4f}  std(across)={mu.std():.4f}  "
          f"|mu|_median={mu_abs.median():.4f}  |mu|_max={mu_abs.max():.4f}")
    print(f"[STATS] logvar: mean={logvar.mean():.4f}  min={logvar.min():.4f}  "
          f"max={logvar.max():.4f}")
    print(f"[STATS] std=exp(0.5*logvar): mean={std.mean():.4f}  median={std.median():.4f}  "
          f"max={std.max():.4f}")

    # Resize AE output (e.g., 512x512) back to GT resolution.
    def _to_gt(r):
        if r.shape[-2] != H_gt or r.shape[-1] != W_gt:
            r = F.interpolate(r.view(B * T, 1, r.shape[-2], r.shape[-1]),
                              size=(H_gt, W_gt), mode="bilinear", align_corners=False)
            r = r.view(B, T, 1, H_gt, W_gt)
        return r

    radar_det = _to_gt(radar_det)
    err = lambda a, b: ((a - b) ** 2).mean(dim=[1, 2, 3, 4]).cpu().numpy()
    mae = lambda a, b: (a - b).abs().mean(dim=[1, 2, 3, 4]).cpu().numpy()
    mse_det = err(radar_det, radar_gt)
    mae_det = mae(radar_det, radar_gt)

    # ---- Mode B: deterministic = False (= z), repeated N times ----
    radar_stos = []
    mse_stos = []
    mae_stos = []
    for k in range(args.n_stochastic):
        torch.manual_seed(args.seed + 1000 + k)
        radar_sto, _, _ = run_once(
            sat_ir_tokens, lgt_tokens, nnet_ema, cond_projector, radar_ae,
            config, deterministic=False, num_latent_tokens=num_latent_tokens,
            device=device,
        )
        radar_sto = _to_gt(radar_sto)
        radar_stos.append(radar_sto)
        mse_stos.append(err(radar_sto, radar_gt))
        mae_stos.append(mae(radar_sto, radar_gt))

    radar_stos_t = torch.stack(radar_stos)                    # [K, B, T, 1, H, W]
    mse_stos_np = np.stack(mse_stos)                          # [K, B]
    mae_stos_np = np.stack(mae_stos)

    # Pairwise variance among stochastic samples: measures how much randomness affects output.
    mean_sto = radar_stos_t.mean(dim=0)
    diff_to_mean = (radar_stos_t - mean_sto.unsqueeze(0)).pow(2).mean(dim=[2, 3, 4, 5])  # [K, B]
    pairwise_var_per_sample = diff_to_mean.mean(dim=0).cpu().numpy()  # [B]

    # Det vs avg-of-sto MSE gap
    mean_sto_metric = ((mean_sto - radar_gt) ** 2).mean(dim=[1, 2, 3, 4]).cpu().numpy()

    summary = {
        "ckpt": args.ckpt,
        "config": args.config,
        "batch_size": int(B),
        "T": int(T),
        "mu_stats": {
            "mean": float(mu.mean()), "std_across": float(mu.std()),
            "abs_median": float(mu_abs.median()), "abs_max": float(mu_abs.max()),
        },
        "logvar_stats": {
            "mean": float(logvar.mean()), "min": float(logvar.min()), "max": float(logvar.max()),
        },
        "std_eps_stats": {
            "mean": float(std.mean()), "median": float(std.median()), "max": float(std.max()),
        },
        "mse_deterministic": mse_det.tolist(),
        "mae_deterministic": mae_det.tolist(),
        "mse_stochastic_per_run": mse_stos_np.tolist(),
        "mae_stochastic_per_run": mae_stos_np.tolist(),
        "mse_mean_of_stochastic": mse_stos_np.mean(axis=0).tolist(),
        "mae_mean_of_stochastic": mae_stos_np.mean(axis=0).tolist(),
        "mse_of_mean_sto_sample": mean_sto_metric.tolist(),
        "pairwise_var_per_sample": pairwise_var_per_sample.tolist(),
    }

    # Summary numbers across batch.
    print("\n=== Summary across the batch ===")
    print(f"MSE det            (mean±std):  {mse_det.mean():.6f} ± {mse_det.std():.6f}")
    print(f"MSE sto per-run    (mean±std):  {mse_stos_np.mean():.6f} ± {mse_stos_np.std():.6f}")
    print(f"MSE of avg-of-sto  (mean±std):  {mean_sto_metric.mean():.6f} ± {mean_sto_metric.std():.6f}")
    print(f"MAE det            (mean±std):  {mae_det.mean():.6f} ± {mae_det.std():.6f}")
    print(f"MAE sto per-run    (mean±std):  {mae_stos_np.mean():.6f} ± {mae_stos_np.std():.6f}")
    print(f"Pairwise-var among sto samples (mean±std): "
          f"{pairwise_var_per_sample.mean():.6e} ± {pairwise_var_per_sample.std():.6e}")

    json_path = os.path.join(args.out_dir, "det_vs_sto_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Wrote {json_path}")

    # Save visual panel for the first frame of each sample: GT | det | sto[0] | sto[1]
    from matplotlib import pyplot as plt
    from matplotlib import colors as mcolors
    try:
        import cmcrameri  # noqa: F401
        cmap_name = "cmc.batlow"
    except Exception:
        cmap_name = "viridis"

    def _panel(a, b, c, d, out):
        cmap = plt.get_cmap(cmap_name)
        norm = mcolors.Normalize(0.0, 1.0, clip=True)
        imgs = [cmap(norm(x))[..., :3] for x in (a, b, c, d)]
        composite = np.concatenate(imgs, axis=1)
        plt.imsave(out, composite)

    for i in range(B):
        gt2d = radar_gt[i, 0, 0].cpu().numpy()
        det2d = radar_det[i, 0, 0].cpu().numpy()
        s0 = radar_stos[0][i, 0, 0].cpu().numpy()
        s1 = radar_stos[min(1, args.n_stochastic - 1)][i, 0, 0].cpu().numpy()
        _panel(gt2d, det2d, s0, s1,
               os.path.join(args.out_dir, f"sample_{i:02d}_gt_det_sto0_sto1.png"))
    print(f"[INFO] Saved {B} visual panels to {args.out_dir}")


if __name__ == "__main__":
    main()
