"""Pre-qsub CPU smoke: exercise the real integrated training paths with
synthetic tokens (no AE, no dataset). Gate before submitting GPU jobs.

Run: python tests/smoke_paths.py   (exit 0 = all paths OK)
"""
import importlib.util
import os
import sys
import traceback

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from flow_utils import get_nnet  # noqa: E402
from diffusion.flow_matching import FlowMatching  # noqa: E402
from diffusion.token_diffusion import TokenDiffusion  # noqa: E402


def load_cfg(name):
    f = os.path.join(ROOT, "configs", name)
    s = importlib.util.spec_from_file_location(name[:-3], f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m.get_config()


def _finite(t):
    return torch.isfinite(t).all().item()


def smoke_flow(cfg_name, tag):
    cfg = load_cfg(cfg_name)
    nnet = get_nnet(**cfg.nnet)
    B, L, C = 2, cfg.vq_model.num_latent_tokens, cfg.nnet.model_args.channels
    sat = torch.randn(B, L, C)
    radar = torch.randn(B, L, C)
    fm = FlowMatching(
        noising_type=cfg.nnet.model_args.noising_type,
        noising_scale=cfg.nnet.model_args.noising_scale,
        flow_prediction_target=cfg.get("flow_prediction_target", "velocity"),
    )
    loss, d = fm(x=radar, nnet=nnet, cond=sat, all_config=cfg, valid_mask=None)
    # mos() returns per-sample [B]; real train loop reduces via .mean() (≈L831).
    loss.mean().backward()
    g = [p.grad for p in nnet.parameters() if p.grad is not None]
    assert _finite(loss) and len(g) > 0 and _finite(g[0]), tag
    print(f"OK  {tag:18s} loss={loss.mean().item():.4f} "
          f"fpt={cfg.get('flow_prediction_target','velocity')} "
          f"utve={cfg.use_text_vae_encoder} diff={d['diff_loss'].mean().item():.4f}")


def smoke_diffusion(cfg_name, tag):
    cfg = load_cfg(cfg_name)
    nnet = get_nnet(**cfg.nnet)
    assert nnet.x_embedder.in_features == 2 * cfg.nnet.model_args.channels, \
        f"{tag}: expected concat in_features"
    B, L, C = 2, cfg.vq_model.num_latent_tokens, cfg.nnet.model_args.channels
    sat = torch.randn(B, L, C)
    radar = torch.randn(B, L, C)
    dcfg = cfg.diffusion
    td = TokenDiffusion(
        train_timesteps=int(dcfg.get("train_timesteps", 1000)),
        schedule=dcfg.get("schedule", "linear"),
        target=dcfg.get("target", "pred_x0"),
        gamma=dcfg.get("gamma", "ddim"),
    )
    loss, d = td.loss(nnet, radar, sat)
    loss.backward()
    g = [p.grad for p in nnet.parameters() if p.grad is not None]
    assert _finite(loss) and len(g) > 0 and _finite(g[0]), tag
    with torch.no_grad():
        z = td.ddim_sample(nnet, cond=sat, sample_steps=4)
    assert tuple(z.shape) == (B, L, C) and _finite(z), f"{tag}: ddim shape/finite"
    print(f"OK  {tag:18s} loss={loss.item():.4f} ddim_out={tuple(z.shape)}")


def main():
    jobs = [
        ("flow",  smoke_flow,      "Sat2Radar-i2i-b-m1-2021summer_gadi.py", "M1 baseline"),
        ("flow",  smoke_flow,      "Sat2Radar-i2i-b-m2-2021summer_gadi.py", "M2 textVAE"),
        ("flow",  smoke_flow,      "Sat2Radar-i2i-b-m4-2021summer_gadi.py", "M4 x1-pred"),
        ("diff",  smoke_diffusion, "Sat2Radar-i2i-b-m3-2021summer_gadi.py", "M3 diffusion"),
    ]
    fails = 0
    for _, fn, cfg, tag in jobs:
        try:
            fn(cfg, tag)
        except Exception:
            fails += 1
            print(f"FAIL {tag}")
            traceback.print_exc()
    print(f"\n{len(jobs) - fails}/{len(jobs)} paths OK")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
