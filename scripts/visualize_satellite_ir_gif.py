#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import cmcrameri

# Ensure project root is on sys.path so we can import local modules.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml_collections import ConfigDict  # noqa: E402
from data.dataset import SatelliteRadarNpyDataset, collate_sat2radar_v2v  # noqa: E402
from scripts.test_sat2radar_v2v import load_py_config  # noqa: E402


def _cmap_or_fallback(name, fallback="viridis"):
    """Return a valid colormap name, falling back if needed."""
    try:
        plt.get_cmap(name)
        return name
    except ValueError:
        print(f"[WARN] Colormap '{name}' not found, fallback to '{fallback}'")
        return fallback


def _apply_cmap(img2d, cmap_name, vmin, vmax):
    """Apply matplotlib colormap to a 2D array and return RGB image."""
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return cmap(norm(img2d))[..., :3]


def build_satellite_dataloader(config, split: str, mode: str, batch_size: int):
    """
    Build dataloader over satellite videos only, using the same filelist/config
    as Sat2Radar i2i/v2v 流程。
    """
    assert mode in ["i2i", "v2v"]
    if mode == "i2i":
        num_frames = 1
    else:
        num_frames = config.dataset.get("num_frames", 16)

    ir_band_indices = config.dataset.get("ir_band_indices", [0, 2, 6])
    use_lightning = config.dataset.get("use_lightning", False)

    dataset = SatelliteRadarNpyDataset(
        base_dir=None,
        years=None,
        mode="sat2radar_v2v",
        filelist_path=config.dataset.filelist_path,
        filelist_split=split,
        files=None,
        frame_stride=config.dataset.get("frame_stride", 1),
        num_frames=num_frames,
        ir_band_indices=ir_band_indices,
        use_lightning=use_lightning,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers_per_gpu,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_sat2radar_v2v,
    )

    num_samples = len(dataset)
    num_batches = len(loader)
    num_frames_cfg = num_frames if mode == "i2i" else config.dataset.get("num_frames", 16)
    print(f"Dataset [{split.upper()}] (satellite only):")
    print(f"  Split: {split}")
    print(f"  Mode: {mode} (sat2radar_v2v)")
    print(f"  Samples: {num_samples}")
    print(f"  Batches: {num_batches}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num frames: {num_frames_cfg}")
    print(f"  Frame stride: {config.dataset.get('frame_stride', 1)}")
    print(f"  Filelist: {config.dataset.filelist_path}")
    print(f"  IR band indices: {ir_band_indices}")
    print(f"  Use lightning: {use_lightning}")

    return loader


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Visualize satellite IR 7/9/13 channels as GIFs.")
    parser.add_argument(
        "--config",
        required=True,
        help="Python config file, e.g. configs/Sat2Radar-v2v-pretrained-FlowTiTok-XL.py",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to save GIFs",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split in filelist (train/val/test)",
    )
    parser.add_argument(
        "--mode",
        default="v2v",
        choices=["i2i", "v2v"],
        help="Visualization mode: i2i (T=1) or v2v (T>=1)",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=10,
        help="Max number of batches to visualize (-1 for all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="CUDA_VISIBLE_DEVICES override, e.g. '0'",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frames per second for GIFs",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_py_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = build_satellite_dataloader(
        config=config,
        split=args.split,
        mode=args.mode,
        batch_size=args.batch_size,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # 物理范围和 colormap，与训练/测试脚本保持一致
    ir_min, ir_max = 200.0, 320.0
    cmap_ir = _cmap_or_fallback("cmc.batlow_r", fallback="viridis")

    ir_names = ["IR7", "IR9", "IR13"]  # 对应原始 0,2,6 三个通道

    max_batches = args.max_batches
    for b_idx, batch in enumerate(loader):
        if max_batches >= 0 and b_idx >= max_batches:
            break

        sat_video = batch["sat_video"].to(
            device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )  # [B, T_max, C_sat(=3), H, W]
        sat_paths = batch.get("sat_paths", None)

        B, T_max, C_sat, H, W = sat_video.shape
        print(f"[batch {b_idx}] B={B}, T={T_max}, C_sat={C_sat}, H={H}, W={W}")

        for i in range(B):
            frames = []
            for t in range(T_max):
                # 取当前样本的 t 帧三个 IR 通道
                # sat_video 已经过 scale_sat_lgt_img 归一化到 [0,1]，这里再还原物理范围便于 colormap。
                ir_panels = []
                for ch in range(min(3, C_sat)):
                    ir_ch = sat_video[i, t, ch] * (ir_max - ir_min) + ir_min  # [H, W]
                    ir_ch_np = ir_ch.detach().cpu().numpy()
                    ir_rgb = _apply_cmap(ir_ch_np, cmap_ir, ir_min, ir_max)  # [H, W, 3]
                    ir_panels.append(ir_rgb)

                if not ir_panels:
                    continue

                # 横向拼接 IR7/IR9/IR13，得到一帧 [H, 3W, 3]
                frame = np.concatenate(ir_panels, axis=1)
                frame_uint8 = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
                frames.append(frame_uint8)

            if not frames:
                continue

            # 根据第一个时间步的路径生成文件名 stem
            time_stem = f"batch{b_idx}_i{i}"
            if sat_paths and i < len(sat_paths) and len(sat_paths[i]) > 0:
                p0 = sat_paths[i][0]
                time_stem = (
                    os.path.splitext(os.path.basename(str(p0)))[0]
                    .replace(os.sep, "_")
                    .replace(":", "_")
                )

            gif_path = os.path.join(args.out_dir, f"{time_stem}_sat_ir7_9_13.gif")
            try:
                imageio.mimsave(gif_path, frames, fps=args.fps)
                print(f"Saved GIF: {gif_path}")
            except Exception as e:
                print(f"[WARN] Failed to save GIF {gif_path}: {e}")


if __name__ == "__main__":
    main()

