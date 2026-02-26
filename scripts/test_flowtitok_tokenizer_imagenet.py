import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from libs.flowtitok import FlowTiTok  # noqa: E402


def load_py_config(config_path: str):
    """
    Load a Python config file (e.g. configs/FlowTok-XL-Stage1.py) that defines get_config().
    """
    import importlib.util

    config_path = os.path.abspath(config_path)
    spec = importlib.util.spec_from_file_location("flowtok_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_config"):
        raise AttributeError(f"{config_path} must define get_config()")
    return module.get_config()


def build_imagenet_like_dataloader(config, data_root: str, split: str, batch_size: int, num_workers: int):
    """
    Build a dataloader over natural images arranged like ImageNet:
        data_root/
          train/cls_x/xxx.jpg
          val/cls_y/yyy.jpg
          ...

    We only care about images; class labels are ignored.

    Uses:
      - resize_shorter_edge -> Resize
      - crop_size -> CenterCrop
      - ToTensor in [0, 1]
    """
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split {split}, expected 'train' or 'val'.")

    resize_shorter = int(getattr(config.dataset, "resize_shorter_edge", 256))
    crop_size = int(getattr(config.dataset, "crop_size", 256))

    t_list = [
        transforms.Resize(resize_shorter, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),  # [0,1]
    ]

    transform = transforms.Compose(t_list)

    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    dataset = datasets.ImageFolder(root=split_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Test FlowTiTok image tokenizer on natural images (e.g. ImageNet).")
    parser.add_argument(
        "--config",
        required=True,
        help="Python config file, e.g. configs/FlowTok-XL-Stage1.py (must define get_config()).",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Tokenizer checkpoint path, e.g. flowtok_image_tokenizer.bin.",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Root folder of natural image dataset (ImageFolder-style, containing train/ and/or val/).",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
        help="Which split under data_root to evaluate on (default: val).",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to save reconstruction examples and metrics.json.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--max_batches_metrics",
        type=int,
        default=50,
        help="Max number of batches to use for metrics (-1 for all).",
    )
    parser.add_argument(
        "--max_batches_images",
        type=int,
        default=10,
        help="Max number of batches from which to save visualization images (-1 for all).",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="CUDA_VISIBLE_DEVICES override, e.g. '0'.",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.makedirs(args.out_dir, exist_ok=True)

    config = load_py_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataloader over natural images.
    loader = build_imagenet_like_dataloader(
        config=config,
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Log dataset info.
    num_samples = len(loader.dataset)
    num_batches = len(loader)
    print(f"Tokenizer eval dataset [{args.split.upper()}]:")
    print(f"  Root: {args.data_root}")
    print(f"  Samples: {num_samples}")
    print(f"  Batches: {num_batches}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Resize shorter edge: {getattr(config.dataset, 'resize_shorter_edge', 256)}")
    print(f"  Crop size: {getattr(config.dataset, 'crop_size', 256)}")

    # Build FlowTiTok tokenizer and load weights.
    tokenizer = FlowTiTok(config)
    tokenizer.load_pretrained_weight(args.ckpt)
    tokenizer.eval()
    tokenizer.to(device)

    # Metric accumulators.
    mse_total = 0.0
    mse_count = 0
    ssim_total = 0.0
    ssim_count = 0

    # For saving visualization grids.
    saved_images = 0

    def _save_batch_images(batch_idx, images, recons):
        nonlocal saved_images
        # images/recons: [B, 3, H, W] in [0,1] (we clamp before calling)
        b = images.size(0)
        grid_list = []
        for i in range(b):
            x = images[i]
            y = recons[i]
            diff = torch.clamp(torch.abs(y - x) * 4.0, 0.0, 1.0)  # amplify diff for visualization
            triplet = torch.cat([x, y, diff], dim=2)  # [3, H, 3W]
            grid_list.append(triplet)
        grid = torch.cat(grid_list, dim=1)  # [3, B*H, 3W]
        # Save as a single image.
        out_path = os.path.join(args.out_dir, f"recon_batch{batch_idx:04d}.png")
        vutils.save_image(grid, out_path)
        saved_images += 1

    for b_idx, (imgs, _labels) in enumerate(loader):
        do_metrics = args.max_batches_metrics < 0 or b_idx < args.max_batches_metrics
        do_images = args.max_batches_images < 0 or b_idx < args.max_batches_images
        if not do_metrics and not do_images:
            break

        imgs = imgs.to(
            device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )  # [B, 3, H, W] in [0,1]

        # Forward through tokenizer (no text guidance).
        print("imgs min max:", imgs.min(), imgs.max())
        recons, _ = tokenizer(imgs, text_guidance=None)  # [B, 3, H, W]
        print("recons min max before clamp:", recons.min(), recons.max())
        recons = torch.clamp(recons, 0.0, 1.0)
        print("recons min max after clamp:", recons.min(), recons.max())

        if do_metrics:
            # Compute MSE and SSIM per image (0-1 space).
            imgs_cpu = imgs.detach().cpu()
            recons_cpu = recons.detach().cpu()
            b = imgs_cpu.size(0)
            for i in range(b):
                x = imgs_cpu[i].numpy()  # [3,H,W]
                y = recons_cpu[i].numpy()
                x_chw = np.transpose(x, (1, 2, 0))  # [H,W,3]
                y_chw = np.transpose(y, (1, 2, 0))
                mse_total += float(np.mean((x_chw - y_chw) ** 2))
                mse_count += 1
                # SSIM on luminance or average over channels.
                # Convert to grayscale by averaging channels.
                x_gray = np.mean(x_chw, axis=2)
                y_gray = np.mean(y_chw, axis=2)
                ssim_total += ssim(x_gray, y_gray, data_range=1.0)
                ssim_count += 1

        if do_images:
            _save_batch_images(b_idx, imgs, recons)

        if mse_count > 0 and (b_idx + 1) % 10 == 0:
            print(
                f"[batch {b_idx+1}/{num_batches}] "
                f"MSE={mse_total / mse_count:.6f}, SSIM={ssim_total / ssim_count:.6f}"
            )

    # Final metrics.
    metrics = {}
    if mse_count > 0:
        metrics["mse"] = mse_total / mse_count
        print(f"Final MSE: {metrics['mse']:.6f}")
    if ssim_count > 0:
        metrics["ssim"] = ssim_total / ssim_count
        print(f"Final SSIM: {metrics['ssim']:.6f}")

    # Save metrics.json
    import json

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved tokenizer metrics to {metrics_path}")
    print(f"Saved {saved_images} visualization image(s) to {args.out_dir}")


if __name__ == "__main__":
    main()

