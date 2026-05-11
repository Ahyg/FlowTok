"""Quick per-channel eval for the overfit-lgt tiny experiment pair.

Loads each EMA ckpt-5000, forwards the 16 tiny samples, reports per-channel MSE/SSIM.
"""
import os, sys, json
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SatelliteRadarNpyDataset
from libs.flowtitok import FlowTiTok


def _build_flowtitok_config(config):
    """Inlined from utils.train_utils to avoid pulling in the loss-module imports
    (which require pyiqa even when we only do inference)."""
    vq_model = OmegaConf.to_container(config.model.vq_model, resolve=True)
    vq_model.setdefault("use_rmsnorm", False)
    vq_model.setdefault("use_swiglu", True)
    vq_model.setdefault("nan_debug", bool(config.training.get("nan_check", False)))
    dataset = {"crop_size": config.dataset.preprocessing.crop_size}
    return OmegaConf.create({"vq_model": vq_model, "dataset": dataset})

EXPS = {
    "baseline": "/scratch/kl02/yh0308/Projv2v/Experiments/sat10ch_flowtitok_ae_bl77_vae_scratch_overfit_lgt_baseline",
    "weighted": "/scratch/kl02/yh0308/Projv2v/Experiments/sat10ch_flowtitok_ae_bl77_vae_scratch_overfit_lgt_weighted",
}
CFGS = {
    "baseline": "/scratch/kl02/yh0308/Projv2v/FlowTok/configs/sat10ch_flowtitok_ae_bl77_vae_scratch_overfit_lgt_baseline_gadi.yaml",
    "weighted": "/scratch/kl02/yh0308/Projv2v/FlowTok/configs/sat10ch_flowtitok_ae_bl77_vae_scratch_overfit_lgt_weighted_gadi.yaml",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = {}
for name, exp_dir in EXPS.items():
    cfg = OmegaConf.load(CFGS[name])
    ds_cfg = cfg.dataset.params
    ds = SatelliteRadarNpyDataset(
        base_dir=ds_cfg.get("data_dir"),
        years=ds_cfg.get("years", "").split(",") if ds_cfg.get("years") else None,
        mode=ds_cfg.get("mode", "satellite"),
        ir_band_indices=ds_cfg.get("ir_band_indices"),
        use_lightning=ds_cfg.get("use_lightning", True),
        filelist_path=ds_cfg.get("filelist_path"),
        filelist_split=ds_cfg.get("filelist_split", "train"),
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    model_cfg = _build_flowtitok_config(cfg)
    model = FlowTiTok(model_cfg).to(device)
    ckpt_path = os.path.join(exp_dir, "checkpoint-5000", "ema_model", "pytorch_model.bin")
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[{name}] loaded {ckpt_path}; missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    per_ch_mse = None
    per_ch_ssim = None
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            tg = torch.zeros(x.shape[0], cfg.model.vq_model.get("text_context_length", 77),
                             cfg.model.vq_model.get("text_embed_dim", 768),
                             device=device, dtype=x.dtype)
            recon, _ = model(x, tg)
            recon = torch.clamp(recon, 0.0, 1.0)
            err = (recon - x) ** 2
            ch_mse = err.mean(dim=(2, 3)).sum(dim=0).cpu().numpy()  # [C]
            if per_ch_mse is None:
                per_ch_mse = np.zeros(ch_mse.shape, dtype=np.float64)
                per_ch_ssim = np.zeros(ch_mse.shape, dtype=np.float64)
            per_ch_mse += ch_mse
            xc = x.cpu().numpy(); rc = recon.cpu().numpy()
            for b in range(xc.shape[0]):
                for c in range(xc.shape[1]):
                    per_ch_ssim[c] += ssim(xc[b, c], rc[b, c], data_range=1.0)
            n += x.shape[0]
    per_ch_mse /= n
    per_ch_ssim /= n
    results[name] = dict(
        per_ch_mse=per_ch_mse.tolist(),
        per_ch_ssim=per_ch_ssim.tolist(),
        n=int(n),
    )

C = len(results["baseline"]["per_ch_mse"])
print(f"\nN samples = {results['baseline']['n']}\n")
print(f"{'ch':>3}  {'baseline MSE':>14}  {'weighted MSE':>14}  {'Δ MSE':>10}    {'baseline SSIM':>14}  {'weighted SSIM':>14}  {'Δ SSIM':>10}")
for c in range(C):
    bm = results['baseline']['per_ch_mse'][c]
    wm = results['weighted']['per_ch_mse'][c]
    bs = results['baseline']['per_ch_ssim'][c]
    ws = results['weighted']['per_ch_ssim'][c]
    label = "lgt" if c == 10 else f"IR{c}"
    print(f"{label:>3}  {bm:>14.6e}  {wm:>14.6e}  {(wm-bm)/bm*100:>+9.1f}%    {bs:>14.4f}  {ws:>14.4f}  {(ws-bs):>+10.4f}")

ir_b_mse = np.mean(results['baseline']['per_ch_mse'][:10])
ir_w_mse = np.mean(results['weighted']['per_ch_mse'][:10])
ir_b_ssim = np.mean(results['baseline']['per_ch_ssim'][:10])
ir_w_ssim = np.mean(results['weighted']['per_ch_ssim'][:10])
print(f"\n{'IR avg':>3}  {ir_b_mse:>14.6e}  {ir_w_mse:>14.6e}  {(ir_w_mse-ir_b_mse)/ir_b_mse*100:>+9.1f}%    {ir_b_ssim:>14.4f}  {ir_w_ssim:>14.4f}  {(ir_w_ssim-ir_b_ssim):>+10.4f}")

with open("/tmp/overfit_lgt_per_ch.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved /tmp/overfit_lgt_per_ch.json")
