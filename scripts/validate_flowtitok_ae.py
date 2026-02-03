import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmweather
import cmcrameri

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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)  # pytorch_model.bin
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_batches", type=int, default=2)
    parser.add_argument("--split", default="val", help="Override filelist_split, e.g. val/test")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    # Dataset (val split if filelist_path is used).
    ds_cfg = config.dataset.params
    filelist_split = args.split or ds_cfg.get("filelist_split", "val")
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

    for b_idx, batch in enumerate(eval_loader):
        if b_idx >= args.max_batches:
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
        print("recon:", recon.min(), recon.max())
        recon = torch.clamp(recon, 0.0, 1.0).cpu().numpy()
        images_np = images.detach().cpu().numpy()

        for i in range(images.shape[0]):
            safe_name = os.path.basename(str(paths[i])).replace(os.sep, "_").replace(":", "_")
            img = images_np[i]

            if ds_cfg.get("mode", "satellite") == "satellite":
                # IR channels 0-9 -> bands 7-16
                ir = img[:10] * (ir_max - ir_min) + ir_min
                for ch in range(10):
                    band = 7 + ch
                    orig_ch = ir[ch]
                    recon_ch = recon[i, ch] * (ir_max - ir_min) + ir_min
                    out_path = os.path.join(
                        args.out_dir, f"{b_idx:03d}_{i:02d}_{safe_name}_ir{band:02d}.png"
                    )
                    _save_composite(orig_ch, recon_ch, cmap_ir, ir_min, ir_max, out_path)

                if ds_cfg.get("use_lightning", True) and img.shape[0] > 10:
                    orig_ch = img[10] * (l_max - l_min) + l_min
                    recon_ch = recon[i, 10] * (l_max - l_min) + l_min
                    out_path = os.path.join(
                        args.out_dir, f"{b_idx:03d}_{i:02d}_{safe_name}_lightning.png"
                    )
                    _save_composite(orig_ch, recon_ch, cmap_lgt, 0.0, l_max, out_path)

            else:  # radar
                orig_ch = img[0] * (z_max - z_min) + z_min
                recon_ch = recon[i, 0] * (z_max - z_min) + z_min
                out_path = os.path.join(
                    args.out_dir, f"{b_idx:03d}_{i:02d}_{safe_name}_radar.png"
                )
                _save_composite(orig_ch, recon_ch, cmap_rad, z_min, z_max, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
