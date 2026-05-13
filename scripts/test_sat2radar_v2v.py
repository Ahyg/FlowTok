import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import r2_score as sk_r2
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors as mcolors
from tqdm import tqdm
import cmweather
import cmcrameri
import open_clip

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARN] cv2 not available. Sobel gradient metrics will be skipped.")

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# Import pysteps verification functions
try:
    from pysteps.verification.spatialscores import (
        fss_init,
        fss_accum,
        fss_compute,
    )
    import pysteps.verification.spatialscores as pvs
    import pysteps.verification.detcatscores as pvdcat
    PYSTEPS_AVAILABLE = True
except ImportError:
    PYSTEPS_AVAILABLE = False
    pvs = None
    pvdcat = None
    print("[WARN] pysteps not available. FSS / categorical metrics will be skipped.")

# Match Diffi2i: radar pixels < 1 dBZ are masked in visualisations.
_RADAR_THR_01 = 1.0 / 60.0  # 1 dBZ in [0,1] normalized scale

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml_collections import ConfigDict  # noqa: E402

from data.dataset import SatelliteRadarNpyDataset, collate_sat2radar_v2v  # noqa: E402
from diffusion.flow_matching import ODEEulerFlowMatchingSolver  # noqa: E402
from libs.flowtitok import FlowTiTok  # noqa: E402
from libs.adapters import AdapterIn, AdapterOut  # noqa: E402
import flow_utils  # noqa: E402

# Generation-quality metrics: i2i (FID/sFID/KID via torchmetrics) and
# v2v (FVD/KVD via I3D-Kinetics + Temporal Consistency via RAFT).
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _generation_metrics import (
        make_i2i_metrics,
        make_v2v_metrics,
        TemporalConsistency,
        radar_to_3ch,
    )
    GEN_METRICS_AVAILABLE = True
except ImportError as _e:
    GEN_METRICS_AVAILABLE = False
    print(f"[WARN] generation metrics module failed to load: {_e}")


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


def _save_two_panel(orig_2d, pred_2d, cmap_name, vmin, vmax, out_path):
    """
    Save a 2-panel composite [orig | pred] using the given colormap and value range.
    """
    orig_rgb = _apply_cmap(orig_2d, cmap_name, vmin, vmax)
    pred_rgb = _apply_cmap(pred_2d, cmap_name, vmin, vmax)
    composite = np.concatenate([orig_rgb, pred_rgb], axis=1)
    plt.imsave(out_path, composite)


def _add_grid_lines(ax, color="black"):
    """Match Diffi2i: dashed gridlines at 25/50/75% of the axes."""
    for frac in [0.25, 0.5, 0.75]:
        ax.plot([0, 1], [frac, frac], transform=ax.transAxes,
                linestyle='--', linewidth=0.8, color=color, alpha=0.5)
        ax.plot([frac, frac], [0, 1], transform=ax.transAxes,
                linestyle='--', linewidth=0.8, color=color, alpha=0.5)


def _compute_radar_metrics_dbz(gt_01, pred_01):
    """Match Diffi2i `_compute_radar_metrics`. Per-sample text annotation only.

    Args:
        gt_01, pred_01: [H, W] arrays in [0,1] normalized radar scale.
    Returns:
        (mse, r2, fss, csi35, pod35, far35) computed on the dBZ-rescaled frame.
    """
    gt_dbz = (gt_01 * 60.0).astype(np.float32)
    pred_dbz = (pred_01 * 60.0).astype(np.float32)
    mse = float(np.mean((pred_dbz - gt_dbz) ** 2))
    try:
        r2 = float(sk_r2(gt_dbz.ravel(), pred_dbz.ravel()))
    except Exception:
        r2 = float("nan")
    if PYSTEPS_AVAILABLE:
        thrs = np.arange(0, 61, 5)
        scales = np.arange(1, 11)
        fss_vals = [pvs.fss(pred_dbz, gt_dbz, thr=int(t), scale=int(s))
                    for t, s in itertools.product(thrs, scales)]
        fss = float(np.nanmean(fss_vals)) if fss_vals else float("nan")
        try:
            csi35 = float(pvdcat.det_cat_fct(pred_dbz, gt_dbz, thr=35,
                                             scores='CSI', axis=None)["CSI"].item())
            pod35 = float(pvdcat.det_cat_fct(pred_dbz, gt_dbz, thr=35,
                                             scores='POD', axis=None)["POD"].item())
            far35 = float(pvdcat.det_cat_fct(pred_dbz, gt_dbz, thr=35,
                                             scores='FAR', axis=None)["FAR"].item())
        except Exception:
            csi35 = pod35 = far35 = float("nan")
    else:
        fss = csi35 = pod35 = far35 = float("nan")
    return mse, r2, fss, csi35, pod35, far35


def _sobel_mean(im, k):
    """Mean Sobel-gradient magnitude over a 2D image (dBZ scale). Match Diffi2i."""
    if not HAS_CV2:
        return float("nan")
    gx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=k)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def _save_four_panel_sat_light_radar(
    sat_ir_01,
    sat_lgt_01,
    gt_01,
    pred_01,
    cmap_ir,
    cmap_lgt,
    cmap_rad,
    out_path,
    metrics_text=None,
):
    """Render a 1×4 panel PNG: [Sat IR ch0 | Lightning | GT Radar | Pred Radar].

    Match Diffi2i `save_vis_images`:
      - inputs in [0,1]; vmin=0, vmax=1 across all panels
      - radar panels mask pixels < 1 dBZ (1/60 in [0,1])
      - dashed gridlines at 25/50/75%
      - per-sample metrics text overlaid on Pred panel (computed on this frame)
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), squeeze=False)
    ax = axes[0]
    ax[0].imshow(sat_ir_01, cmap=cmap_ir, vmin=0, vmax=1)
    ax[1].imshow(sat_lgt_01, cmap=cmap_lgt, vmin=0, vmax=1)
    ax[2].imshow(np.ma.masked_less(gt_01, _RADAR_THR_01),
                 cmap=cmap_rad, vmin=0, vmax=1)
    ax[3].imshow(np.ma.masked_less(pred_01, _RADAR_THR_01),
                 cmap=cmap_rad, vmin=0, vmax=1)
    titles = ["Sat IR ch0", "Lightning", "GT Radar", "Pred Radar"]
    for j, ttl in enumerate(titles):
        ax[j].set_title(ttl)
    for a in ax:
        a.axis("off")
        _add_grid_lines(a)

    if metrics_text is None:
        mse, r2, fss, csi35, pod35, far35 = _compute_radar_metrics_dbz(gt_01, pred_01)
        metrics_text = (
            f"MSE:{mse:.2f}, R²:{r2:.2f}\n"
            f"FSS:{fss:.2f}, CSI35:{csi35:.2f}\n"
            f"POD35:{pod35:.2f}, FAR35:{far35:.2f}"
        )
    ax[3].text(0.01, 0.97, metrics_text,
               transform=ax[3].transAxes, ha='left', va='top',
               fontsize=7, color='black', fontweight='bold')

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_v2v_gif_mpl(
    sat_video_01,    # [B, T, H, W]
    lgt_video_01,    # [B, T, H, W]
    gt_video_01,     # [B, T, H, W]
    pred_video_01,   # [B, T, H, W]
    valid_mask,      # [B, T] bool tensor or None
    out_path,
    cmap_ir,
    cmap_lgt,
    cmap_rad,
    fps=4,
):
    """Render a B×4 panel GIF animating T frames. Matches Diffi2i `save_vis_video`.

    All input arrays in [0,1]. Radar panels mask < 1 dBZ; gridlines at 25/50/75%;
    per-sample metrics text overlaid on Pred panel (recomputed each frame).
    """
    B, T, H, W = sat_video_01.shape
    fig, axes = plt.subplots(B, 4, figsize=(16, 4 * B), squeeze=False)
    titles = ["Sat IR ch0", "Lightning", "GT Radar", "Pred Radar"]
    for j, ttl in enumerate(titles):
        axes[0, j].set_title(ttl)
    for i in range(B):
        for j in range(4):
            axes[i, j].axis("off")
            _add_grid_lines(axes[i, j])
        axes[i, 0].set_ylabel(f"sample {i}", rotation=90, fontsize=10)

    im_sat = [axes[i, 0].imshow(sat_video_01[i, 0], cmap=cmap_ir,
                                vmin=0, vmax=1, animated=True) for i in range(B)]
    im_lgt = [axes[i, 1].imshow(lgt_video_01[i, 0], cmap=cmap_lgt,
                                vmin=0, vmax=1, animated=True) for i in range(B)]
    im_gt = [axes[i, 2].imshow(np.ma.masked_less(gt_video_01[i, 0], _RADAR_THR_01),
                               cmap=cmap_rad, vmin=0, vmax=1, animated=True) for i in range(B)]
    im_pred = [axes[i, 3].imshow(np.ma.masked_less(pred_video_01[i, 0], _RADAR_THR_01),
                                 cmap=cmap_rad, vmin=0, vmax=1, animated=True) for i in range(B)]

    txt_metrics = []
    for i in range(B):
        mse, r2, fss, csi35, pod35, far35 = _compute_radar_metrics_dbz(
            gt_video_01[i, 0], pred_video_01[i, 0])
        txt = axes[i, 3].text(
            0.01, 0.97,
            f"MSE:{mse:.2f}, R²:{r2:.2f}\n"
            f"FSS:{fss:.2f}, CSI35:{csi35:.2f}\n"
            f"POD35:{pod35:.2f}, FAR35:{far35:.2f}",
            transform=axes[i, 3].transAxes, ha='left', va='top',
            fontsize=7, color='black', fontweight='bold', animated=True,
        )
        txt_metrics.append(txt)

    title_obj = fig.suptitle(f"frame 0/{T}", fontsize=12)
    fig.tight_layout()

    def _update(t):
        for i in range(B):
            if valid_mask is not None and not bool(valid_mask[i, t]):
                continue
            im_sat[i].set_data(sat_video_01[i, t])
            im_lgt[i].set_data(lgt_video_01[i, t])
            im_gt[i].set_data(np.ma.masked_less(gt_video_01[i, t], _RADAR_THR_01))
            im_pred[i].set_data(np.ma.masked_less(pred_video_01[i, t], _RADAR_THR_01))
            mse, r2, fss, csi35, pod35, far35 = _compute_radar_metrics_dbz(
                gt_video_01[i, t], pred_video_01[i, t])
            txt_metrics[i].set_text(
                f"MSE:{mse:.2f}, R²:{r2:.2f}\n"
                f"FSS:{fss:.2f}, CSI35:{csi35:.2f}\n"
                f"POD35:{pod35:.2f}, FAR35:{far35:.2f}"
            )
        title_obj.set_text(f"frame {t}/{T}")
        return im_sat + im_lgt + im_gt + im_pred + txt_metrics + [title_obj]

    ani = animation.FuncAnimation(fig, _update, frames=T, interval=300, blit=False)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)


@torch.no_grad()
def encode_video_with_autoencoder(autoencoder, video, scale_factor: float, adapter_in=None):
    """
    video: [B, T, C, H, W]
    return: tokens [B, T*L, C_tok]
    """
    B, T, C, H, W = video.shape

    # 将输入 resize 到 AE 预训练时的分辨率（例如 512x512），
    # 避免 FlowTiTok 的 RoPE 位置编码与 checkpoint 中的形状不一致。
    ae_ds_cfg = getattr(getattr(autoencoder, "config", None), "dataset", None)
    if isinstance(ae_ds_cfg, dict):
        target_size = ae_ds_cfg.get("crop_size", H)
    else:
        target_size = getattr(getattr(autoencoder, "config", None), "ae_image_size", H)

    video = video.view(B * T, C, H, W)

    if adapter_in is not None:
        video = adapter_in(video, target_size)
    else:
        # 通道对齐：FlowTiTok_512 预训练是 3 通道 in / 3 通道 out。
        # 如果 AE 期望 3 通道而当前是 1 通道（雷达），则将单通道复制为 3 通道。
        expected_in_ch = getattr(getattr(autoencoder, "config", None), "vq_model", {}).get(
            "in_channels", C
        )
        if C == 1 and expected_in_ch == 3:
            video = video.repeat(1, 3, 1, 1)  # [B*T, 3, H, W]

        if H != target_size or W != target_size:
            video = F.interpolate(
                video,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )

    z = autoencoder.encode(video)[0].mul_(scale_factor)  # [B*T, C_tok, 1, L]
    z = z.squeeze(2).permute(0, 2, 1)  # [B*T, L, C_tok]
    L = z.shape[1]
    # interpolate / repeat 等操作可能导致非 contiguous，view 会报错，改用 reshape 更安全
    z = z.reshape(B, T * L, z.shape[2])   # [B, T*L, C_tok]
    return z


def load_py_config(config_path: str):
    """
    Load a Python config file (e.g. configs/Sat2Radar-Video-XL.py) that defines get_config().
    """
    import importlib.util

    config_path = os.path.abspath(config_path)
    spec = importlib.util.spec_from_file_location("sat2radar_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_config"):
        raise AttributeError(f"{config_path} must define get_config()")
    return module.get_config()


def build_eval_dataloader(config, split: str, batch_size: int, mode: str):
    """
    Build dataloader for I2I (T=1) or V2V (T>=1).
    Assumes filelist was built with build_dataset.py --v2v --clip-length {1 or >1}.
    """
    assert mode in ["i2i", "v2v"]
    if mode == "i2i":
        num_frames = 1
    else:
        num_frames = config.dataset.get("num_frames", 16)

    # 与训练脚本保持一致：使用 ir_band_indices / use_lightning 选择卫星通道
    ir_band_indices = config.dataset.get("ir_band_indices", None)
    use_lightning = config.dataset.get("use_lightning", True)

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
    
    # Log dataset info
    num_samples = len(dataset)
    num_batches = len(loader)
    num_frames_cfg = num_frames if mode == "i2i" else config.dataset.get("num_frames", 16)
    print(f"Dataset [{split.upper()}]:")
    print(f"  Split: {split}")
    print(f"  Mode: {mode} (sat2radar_v2v)")
    print(f"  Samples: {num_samples}")
    print(f"  Batches: {num_batches}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num frames: {num_frames_cfg}")
    print(f"  Frame stride: {config.dataset.get('frame_stride', 1)}")
    print(f"  Filelist: {config.dataset.filelist_path}")
    
    return loader


def _ae_config(base_config, in_channels, out_channels):
    """
    Build a FlowTiTok autoencoder config compatible with pretrained FlowTiTok_512.bin.

    与训练脚本 train_sat2radar_v2v.py 中的 _ae_config 行为保持一致：
      - 使用 sat_in_channels / radar_in_channels 指定 in/out 通道；
      - 如果配置中提供 ae_image_size，则将 AE 内部的 dataset.crop_size 覆盖为该分辨率，
        并在 config 上记录 ae_image_size，便于 encode 时读取。
    """
    vq = dict(base_config.vq_model)
    vq["in_channels"] = in_channels
    vq["out_channels"] = out_channels
    cfg = ConfigDict(dict(base_config))
    cfg.vq_model = ConfigDict(vq)

    # 为 AE 单独设置预训练时的图像尺寸（例如 512x512），与 FlowTiTok_512 checkpoint 对齐
    ae_img_size = getattr(base_config, "ae_image_size", None)
    if ae_img_size is not None:
        ds = dict(getattr(base_config, "dataset", {}))
        ds["crop_size"] = ae_img_size
        cfg.dataset = ConfigDict(ds)
        # 也记录到 config 上，encode 时可读取
        cfg.ae_image_size = ae_img_size

    return cfg


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Test sat2radar I2I/V2V model with metrics.")
    parser.add_argument("--config", required=True, help="Python config file, e.g. configs/Sat2Radar-Video-XL.py")
    parser.add_argument("--ckpt", required=True, help="TrainState checkpoint dir, e.g. .../100000.ckpt")
    parser.add_argument("--out_dir", required=True, help="Directory to save predicted radar images")
    parser.add_argument("--split", default="test", help="Dataset split in filelist (train/val/test)")
    parser.add_argument("--mode", default="i2i", choices=["i2i", "v2v"], help="Evaluation mode: i2i (T=1) or v2v (T>=1)")
    parser.add_argument("--max_batches_metrics", type=int, default=10, help="Max batches for metrics (-1 for all)")
    parser.add_argument("--max_batches_images", type=int, default=2, help="Max batches to save images (-1 for all)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES override, e.g. '0'")
    parser.add_argument("--filelist_path", default=None, help="Optional override for config.dataset.filelist_path")
    parser.add_argument("--fss_thresholds", default="0,5,10,15,20,25,30,35,40,45,50,55,60")
    parser.add_argument("--fss_scales", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--cat_thresholds", default="5,15,25,35,45,55",
                        help="dBZ thresholds for CSI/POD/FAR/HSS (matches Diffi2i)")
    parser.add_argument("--metrics_json", default=None,
                        help="Path to write metrics JSON (default: <out_dir>/metrics.json)")
    parser.add_argument("--dump_arrays", action="store_true",
                        help="Save per-frame dBZ arrays gt_dbz.npy / pred_dbz.npy under <out_dir>/arrays")
    parser.add_argument("--arrays_dir", default=None,
                        help="Override directory for --dump_arrays output")
    # ── Generation-quality metrics (i2i: FID/sFID/KID, v2v: FVD/KVD/TC) ─────
    parser.add_argument("--skip_gen_metrics", action="store_true",
                        help="Disable all generation-quality metrics (FID/sFID/KID for i2i; FVD/KVD/TC for v2v)")
    parser.add_argument("--skip_fid", action="store_true", help="i2i: skip FID")
    parser.add_argument("--skip_sfid", action="store_true", help="i2i: skip sFID")
    parser.add_argument("--skip_kid", action="store_true", help="i2i: skip KID")
    parser.add_argument("--skip_fvd", action="store_true", help="v2v: skip FVD")
    parser.add_argument("--skip_kvd", action="store_true", help="v2v: skip KVD")
    parser.add_argument("--skip_tc", action="store_true", help="v2v: skip Temporal Consistency (RAFT)")
    parser.add_argument("--kid_subsets", type=int, default=50,
                        help="KID/KVD number of subsets for unbiased estimator")
    parser.add_argument("--kid_subset_size", type=int, default=100,
                        help="KID/KVD subset size (must be <= total samples per call)")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_py_config(args.config)
    if args.filelist_path:
        config.dataset.filelist_path = args.filelist_path
        print(f"[INFO] Override filelist_path: {args.filelist_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader
    loader = build_eval_dataloader(config, split=args.split, batch_size=args.batch_size, mode=args.mode)

    # FlowTok backbone & optimizer (for loading nnet / nnet_ema)
    train_state = flow_utils.initialize_train_state(config, device)
    train_state.load(args.ckpt)  # loads step, optimizer, lr_scheduler, nnet, nnet_ema
    nnet = train_state.nnet.to(device)
    nnet_ema = train_state.nnet_ema.to(device)
    nnet_ema.eval()

    # Optional adapters: loaded from ckpt directory if present.
    adapter_in_satellite = None
    adapter_out = None
    adapter_in_sat_cfg = getattr(config, "adapter_in_satellite", None)
    if adapter_in_sat_cfg is None:
        # Backward compatibility with old config key.
        adapter_in_sat_cfg = getattr(config, "adapter_in", None)
    if adapter_in_sat_cfg and adapter_in_sat_cfg.get("enabled", False):
        adapter_in_satellite = AdapterIn(
            in_channels=int(adapter_in_sat_cfg.get("in_channels", getattr(config, "sat_in_channels", 3))),
            out_channels=int(getattr(config, "sat_in_channels", 3)),
            mid_channels=int(adapter_in_sat_cfg.get("mid_channels", 32)),
            num_blocks=int(adapter_in_sat_cfg.get("num_blocks", 3)),
        ).to(device)
        adapter_in_satellite_path = os.path.join(args.ckpt, "adapter_in_satellite.pth")
        legacy_adapter_in_path = os.path.join(args.ckpt, "adapter_in.pth")
        if os.path.isfile(adapter_in_satellite_path):
            adapter_in_satellite.load_state_dict(
                torch.load(adapter_in_satellite_path, map_location="cpu")
            )
            adapter_in_satellite.eval()
            print(f"[INFO] Loaded adapter_in_satellite from {adapter_in_satellite_path}")
        elif os.path.isfile(legacy_adapter_in_path):
            adapter_in_satellite.load_state_dict(
                torch.load(legacy_adapter_in_path, map_location="cpu")
            )
            adapter_in_satellite.eval()
            print(f"[INFO] Loaded legacy adapter_in from {legacy_adapter_in_path}")
        else:
            print(
                "[WARN] adapter_in_satellite is enabled in config but checkpoint file not found: "
                f"{adapter_in_satellite_path} (or legacy {legacy_adapter_in_path})"
            )
            adapter_in_satellite = None
    if getattr(config, "adapter_out", None) and config.adapter_out.get("enabled", False):
        adapter_out = AdapterOut(
            in_channels=int(getattr(config, "radar_out_channels", 3)),
            out_channels=1,
            mid_channels=int(config.adapter_out.get("mid_channels", 16)),
            num_blocks=int(config.adapter_out.get("num_blocks", 2)),
        ).to(device)
        adapter_out_path = os.path.join(args.ckpt, "adapter_out.pth")
        if os.path.isfile(adapter_out_path):
            adapter_out.load_state_dict(torch.load(adapter_out_path, map_location="cpu"))
            adapter_out.eval()
            print(f"[INFO] Loaded adapter_out from {adapter_out_path}")
        else:
            print(f"[WARN] adapter_out is enabled in config but checkpoint file not found: {adapter_out_path}")
            adapter_out = None

    # Pretrained FlowTiTok autoencoders
    # 这里和 train_sat2radar_v2v.py 保持兼容：
    #   - 新配置会通过 config.sat_in_channels / radar_in_channels 指定通道
    #   - 旧配置则回退到 11ch sat / 1ch radar
    sat_in_ch = getattr(config, "sat_in_channels", None)
    sat_out_ch = getattr(config, "sat_out_channels", None)
    radar_in_ch = getattr(config, "radar_in_channels", None)
    radar_out_ch = getattr(config, "radar_out_channels", None)
    if sat_in_ch is None or sat_out_ch is None:
        sat_in_ch, sat_out_ch = 11, 11
    if radar_in_ch is None or radar_out_ch is None:
        radar_in_ch, radar_out_ch = 1, 1

    sat_ae_config = _ae_config(config, sat_in_ch, sat_out_ch)
    radar_ae_config = _ae_config(config, radar_in_ch, radar_out_ch)

    sat_autoencoder = FlowTiTok(sat_ae_config).to(device)
    sat_autoencoder.load_state_dict(
        torch.load(config.sat_tokenizer_checkpoint, map_location="cpu"),
        strict=False,
    )
    sat_autoencoder.eval()
    sat_autoencoder.requires_grad_(False)

    radar_autoencoder = FlowTiTok(radar_ae_config).to(device)
    radar_autoencoder.load_state_dict(
        torch.load(config.radar_tokenizer_checkpoint, map_location="cpu"),
        strict=False,
    )
    radar_autoencoder.eval()
    radar_autoencoder.requires_grad_(False)

    # Text guidance encoder for FlowTiTok decoder（基于文件名的弱描述）
    clip_model_name = "ViT-L-14-336"
    local_clip_ckpt = os.environ.get("OPENCLIP_LOCAL_CKPT", None)
    try:
        if local_clip_ckpt and os.path.isfile(local_clip_ckpt):
            print(f"[INFO] Loading open_clip '{clip_model_name}' from local checkpoint: {local_clip_ckpt}")
            clip_encoder, _, _ = open_clip.create_model_and_transforms(
                clip_model_name, pretrained=None
            )
            state_dict = torch.load(local_clip_ckpt, map_location="cpu")
            missing, unexpected = clip_encoder.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[INFO] open_clip loaded with missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
        else:
            clip_encoder, _, _ = open_clip.create_model_and_transforms(
                clip_model_name, pretrained="openai"
            )
        del clip_encoder.visual
        clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
        clip_encoder.transformer.batch_first = False
        clip_encoder.eval()
        clip_encoder.requires_grad_(False)
        clip_encoder.to(device)
    except Exception as e:
        clip_encoder = None
        clip_tokenizer = None
        print(
            f"[WARN] open_clip not available, FlowTiTok decoder will run without text guidance. Error: {e}"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    num_latent_tokens = config.vq_model.num_latent_tokens

    def build_condition_tokens_from_sat_video(sat_video):
        use_sat_lgt_tokens = getattr(config, "cond_use_sat_lightning_tokens", False)
        if not use_sat_lgt_tokens:
            return encode_video_with_autoencoder(
                sat_autoencoder,
                sat_video,
                config.vq_model.scale_factor,
                adapter_in=adapter_in_satellite,
            )

        sat_ir_video = sat_video[:, :, :3, :, :]
        # Dataset output layout: [selected IR bands..., lightning].
        # Lightning is the last channel after preprocessing (not second-to-last).
        lgt_slice = sat_video[:, :, -1:, :, :]
        lgt_video = lgt_slice.repeat(1, 1, 3, 1, 1)

        sat_ir_tokens = encode_video_with_autoencoder(
            sat_autoencoder, sat_ir_video, config.vq_model.scale_factor, adapter_in=None
        )
        lgt_tokens = encode_video_with_autoencoder(
            sat_autoencoder, lgt_video, config.vq_model.scale_factor, adapter_in=None
        )

        fusion = getattr(config, "cond_token_fusion", "mean")
        if fusion == "sum":
            return sat_ir_tokens + lgt_tokens
        return 0.5 * (sat_ir_tokens + lgt_tokens)

    # FSS / metrics accumulators
    def _parse_csv_numbers(value, cast=float):
        if value is None or value == "":
            return []
        return [cast(v.strip()) for v in value.split(",") if v.strip() != ""]

    thrs = _parse_csv_numbers(args.fss_thresholds, cast=float)
    scales = _parse_csv_numbers(args.fss_scales, cast=int)
    cat_thresholds = _parse_csv_numbers(args.cat_thresholds, cast=float)
    z_min, z_max = 0.0, 60.0  # radar scaling range in dataset

    # Per-frame counters / sums (all on dBZ scale to match Diffi2i)
    mse_total = 0.0   # mean((p-g)^2) over (sample, frame), in dBZ
    mse_count = 0
    ssim_total = 0.0  # SSIM with data_range=60 (dBZ scale)
    ssim_count = 0
    mae_sum = 0.0     # sum of mean(|p-g|) per frame, in dBZ
    sq_sum = 0.0      # sum of mean((p-g)^2) per frame, in dBZ (RMSE)
    bias_sum = 0.0    # sum of mean(p-g) per frame, in dBZ
    n_frames = 0
    seen_samples = 0
    # Per-frame dBZ arrays for global metrics (R², CSI/POD/FAR/HSS, Sobel, hist)
    all_gt_dbz = []
    all_pred_dbz = []

    # pysteps FSS accumulators (one per threshold-scale combination)
    pysteps_fss_objects = {}
    if PYSTEPS_AVAILABLE:
        for thr in thrs:
            for scale in scales:
                key = (thr, scale)
                pysteps_fss_objects[key] = fss_init(thr=thr, scale=float(scale))

    # ── Generation-quality metrics (mode-specific) ─────────────────────────
    gen_metrics: dict = {}
    tc_metric = None
    use_gen = GEN_METRICS_AVAILABLE and not args.skip_gen_metrics
    if use_gen:
        if args.mode == "i2i":
            gen_metrics = make_i2i_metrics(
                device,
                use_fid=not args.skip_fid,
                use_sfid=not args.skip_sfid,
                use_kid=not args.skip_kid,
                kid_subsets=args.kid_subsets,
                kid_subset_size=args.kid_subset_size,
            )
        else:  # v2v
            gen_metrics = make_v2v_metrics(
                device,
                use_fvd=not args.skip_fvd,
                use_kvd=not args.skip_kvd,
                kid_subsets=args.kid_subsets,
                kid_subset_size=args.kid_subset_size,
            )
            if not args.skip_tc:
                tc_metric = TemporalConsistency(device, dbz_scale=z_max - z_min)
        active = [k for k in gen_metrics if not k.startswith("_")]
        if tc_metric is not None:
            active.append("tc")
        print(f"[INFO] Generation-quality metrics enabled ({args.mode}): {active}")

    def infer_batch(batch, batch_idx: int):
        nonlocal mse_total, mse_count, ssim_total, ssim_count, pysteps_fss_objects
        nonlocal mae_sum, sq_sum, bias_sum, n_frames, seen_samples

        sat_video = batch["sat_video"].to(
            device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )  # [B, T_max, 11, H, W]
        radar_video_gt = batch["radar_video"].to(
            device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )  # [B, T_max, 1, H, W]
        valid_mask = batch.get("valid_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device)

        B, T_max, C_sat, H, W = sat_video.shape

        # Encode sat video -> tokens [B, T_eff*L, C_tok]
        sat_tokens = build_condition_tokens_from_sat_video(sat_video)  # [B, T_max*L, C]

        # FlowTok text encoder branch: 默认使用 textVAE 对 sat tokens 编码得到 x0。
        # 若在配置中显式关闭 textVAE（config.use_text_vae_encoder == False），
        # 则直接使用 sat_tokens 作为 flow 起点，实现“sat tokens -> radar tokens”的显式映射。
        use_text_vae_encoder = getattr(config, "use_text_vae_encoder", True)
        if use_text_vae_encoder:
            x0, _, _ = nnet_ema(sat_tokens, text_encoder=True)
        else:
            x0 = sat_tokens
        if config.nnet.model_args.noising_type != "none":
            x0 = x0 + torch.randn_like(x0) * config.sample.noise_scale

        guidance_scale = config.sample.scale
        ode_solver = ODEEulerFlowMatchingSolver(
            nnet_ema,
            step_size_type="step_in_dsigma",
            guidance_scale=guidance_scale,
        )
        z, _ = ode_solver.sample(
            x_T=x0,
            batch_size=B,
            sample_steps=config.sample.sample_steps,
            unconditional_guidance_scale=guidance_scale,
            has_null_indicator=guidance_scale > 1.0,
        )  # [B, L, C_tok]

        # Reshape tokens back to [B, T_eff, C_tok, 1, L_frame] and decode.
        L = z.shape[1]
        assert (
            L % num_latent_tokens == 0
        ), f"Sequence length must be multiple of num_latent_tokens: got {L} vs {num_latent_tokens}"
        T_eff = L // num_latent_tokens

        z = z.view(B, T_eff, num_latent_tokens, z.shape[2])  # [B, T, Lf, C]
        z = z.view(B * T_eff, num_latent_tokens, z.shape[3])  # [B*T, Lf, C]
        z = z.permute(0, 2, 1).unsqueeze(2)  # [B*T, C, 1, Lf]

        # 构造基于文件名的弱文本描述，作为 FlowTiTok decoder 的 text_guidance
        text_guidance = None
        if clip_encoder is not None and clip_tokenizer is not None:
            radar_paths = batch.get("radar_paths")
            texts = []
            for i in range(B):
                for t in range(T_eff):
                    fname = "unknown"
                    if radar_paths and i < len(radar_paths):
                        paths_i = radar_paths[i]
                        if isinstance(paths_i, (list, tuple)) and t < len(paths_i):
                            fname = os.path.basename(str(paths_i[t]))
                    desc = f"A radar reflectivity image from {fname}."
                    texts.append(desc)

            if len(texts) == B * T_eff:
                try:
                    text_tokens = clip_tokenizer(texts).to(device)
                    cast_dtype = clip_encoder.transformer.get_cast_dtype()
                    text_tokens = clip_encoder.token_embedding(text_tokens).to(
                        cast_dtype
                    )  # [B*T, n_ctx, d_model]
                    text_tokens = (
                        text_tokens + clip_encoder.positional_embedding.to(cast_dtype)
                    )
                    text_tokens = text_tokens.permute(1, 0, 2)  # NLD -> LND
                    text_tokens = clip_encoder.transformer(
                        text_tokens, attn_mask=clip_encoder.attn_mask
                    )
                    text_tokens = text_tokens.permute(1, 0, 2)  # LND -> NLD
                    text_tokens = clip_encoder.ln_final(
                        text_tokens
                    )  # [B*T, n_ctx, d_model]
                    text_guidance = text_tokens
                except Exception as e:
                    print(f"[WARN] Failed to build CLIP text guidance: {e}")
                    text_guidance = None

        radar_pred = radar_autoencoder.decode_tokens(
            z / config.vq_model.scale_factor, text_guidance=text_guidance
        )  # [B*T_eff, C_out, H_pred, W_pred]
        if adapter_out is not None:
            # 直接映射到目标雷达分辨率/通道
            _, _, _, H_gt0, W_gt0 = radar_video_gt[:, :T_eff].shape
            radar_pred = adapter_out(radar_pred, out_size=(H_gt0, W_gt0))
        else:
            # 由于雷达物理上是单通道，只取第一个通道
            radar_pred = radar_pred[:, 0:1, ...]  # [B*T_eff, 1, H_pred, W_pred]
        radar_pred = torch.clamp(radar_pred, 0.0, 1.0)
        radar_pred = radar_pred.view(B, T_eff, 1, radar_pred.shape[-2], radar_pred.shape[-1])
        radar_gt = torch.clamp(radar_video_gt[:, :T_eff], 0.0, 1.0)  # [B, T_eff, 1, H_gt, W_gt]

        # 如果预测雷达的分辨率与原始 GT 不一致（例如 AE 在 512x512 上预训练，
        # 而数据集裁剪为 128x128），则在计算指标 / 可视化前把预测 resize 回 GT 的分辨率，
        # 以便 MSE / SSIM / FSS 和 PNG/GIF 的空间尺寸与训练/验证保持一致。
        _, _, _, H_gt, W_gt = radar_gt.shape
        if radar_pred.shape[-2] != H_gt or radar_pred.shape[-1] != W_gt:
            Bv, Tv, _, H_pred, W_pred = radar_pred.shape
            radar_pred_flat = radar_pred.view(Bv * Tv, 1, H_pred, W_pred)
            radar_pred_flat = F.interpolate(
                radar_pred_flat,
                size=(H_gt, W_gt),
                mode="bilinear",
                align_corners=False,
            )
            radar_pred = radar_pred_flat.view(Bv, Tv, 1, H_gt, W_gt)

        # Metrics per (b, t) where valid_mask is True (or all frames if mask is None).
        # All metrics computed on dBZ scale to match Diffi2i validate_test_ckpt.py.
        radar_pred_cpu = radar_pred.detach().cpu()
        radar_gt_cpu = radar_gt.detach().cpu()

        do_metrics_this_batch = (args.max_batches_metrics < 0
                                 or batch_idx < args.max_batches_metrics)

        if do_metrics_this_batch:
            for b in range(B):
                any_valid = False
                for t in range(T_eff):
                    if valid_mask is not None and not valid_mask[b, t]:
                        continue
                    any_valid = True
                    gt = radar_gt_cpu[b, t, 0].numpy()
                    pred = radar_pred_cpu[b, t, 0].numpy()

                    g_dbz = (gt * (z_max - z_min) + z_min).astype(np.float32)
                    p_dbz = (pred * (z_max - z_min) + z_min).astype(np.float32)

                    diff = p_dbz - g_dbz
                    mse_total += float(np.mean(diff ** 2))
                    mse_count += 1
                    sq_sum += float(np.mean(diff ** 2))
                    mae_sum += float(np.mean(np.abs(diff)))
                    bias_sum += float(np.mean(diff))
                    ssim_total += float(ssim(g_dbz, p_dbz, data_range=60.0))
                    ssim_count += 1
                    n_frames += 1

                    all_gt_dbz.append(g_dbz)
                    all_pred_dbz.append(p_dbz)

                    if PYSTEPS_AVAILABLE:
                        for thr in thrs:
                            for scale in scales:
                                key = (thr, scale)
                                fss_accum(pysteps_fss_objects[key], p_dbz, g_dbz)
                if any_valid:
                    seen_samples += 1

            # ── Generation-quality metrics (mode-specific) ──────────────────
            if gen_metrics or tc_metric is not None:
                # radar_pred / radar_gt are still on device; both [B, T_eff, 1, H, W] in [0,1]
                if args.mode == "i2i":
                    # Flatten (b, t) -> N images, drop invalid frames via valid_mask
                    if valid_mask is not None:
                        keep = valid_mask[:, :T_eff].reshape(-1).to(radar_gt.device).bool()
                    else:
                        keep = torch.ones(B * T_eff, dtype=torch.bool, device=radar_gt.device)
                    if keep.any():
                        gt_flat = radar_gt[:, :T_eff].reshape(B * T_eff, 1, radar_gt.shape[-2], radar_gt.shape[-1])[keep]
                        pr_flat = radar_pred[:, :T_eff].reshape(B * T_eff, 1, radar_pred.shape[-2], radar_pred.shape[-1])[keep]
                        gt3 = radar_to_3ch(gt_flat)
                        pr3 = radar_to_3ch(pr_flat)
                        for k in ("fid", "sfid", "kid"):
                            if k in gen_metrics:
                                gen_metrics[k].update(gt3, real=True)
                                gen_metrics[k].update(pr3, real=False)
                else:  # v2v
                    gt3_clip = radar_to_3ch(radar_gt[:, :T_eff])    # [B, T_eff, 3, H, W]
                    pr3_clip = radar_to_3ch(radar_pred[:, :T_eff])
                    for k in ("fvd", "kvd"):
                        if k in gen_metrics:
                            gen_metrics[k].update(gt3_clip, real=True)
                            gen_metrics[k].update(pr3_clip, real=False)
                    if tc_metric is not None and T_eff >= 2:
                        vm = valid_mask[:, :T_eff] if valid_mask is not None else None
                        try:
                            tc_metric.update(
                                radar_gt[:, :T_eff],
                                radar_pred[:, :T_eff],
                                valid_mask=vm,
                            )
                        except Exception as e:
                            print(f"[WARN] TC update failed for batch {batch_idx}: {e}")

        # Save images / videos：
        #   - PNG：每个样本保存第一个有效帧的 [sat_IR | sat_lightning | radar_gt | radar_pred]
        #   - GIF（仅 v2v 且安装了 imageio）：多样本堆叠，沿时间维展示演化
        if args.max_batches_images < 0 or batch_idx < args.max_batches_images:
            cmap_rad = _cmap_or_fallback("HomeyerRainbow", fallback="viridis")
            cmap_ir = _cmap_or_fallback("cmc.batlow_r", fallback="viridis")
            cmap_lgt = _cmap_or_fallback("Reds", fallback="Reds")
            ir_band_indices = getattr(config.dataset, "ir_band_indices", None)
            lgt_channel_idx = (
                len(ir_band_indices)
                if getattr(config.dataset, "use_lightning", False)
                and ir_band_indices is not None
                else None
            )

            sat_ir_video_01 = sat_video[:, :T_eff, 0].detach().cpu().numpy()
            if lgt_channel_idx is not None and sat_video.shape[2] > lgt_channel_idx:
                sat_lgt_video_01 = sat_video[:, :T_eff, lgt_channel_idx].detach().cpu().numpy()
            else:
                sat_lgt_video_01 = np.zeros_like(sat_ir_video_01)
            gt_video_01 = radar_gt_cpu[:, :T_eff, 0].numpy()
            pred_video_01 = radar_pred_cpu[:, :T_eff, 0].numpy()

            # PNG：每样本一张，取第一个有效帧
            for i in range(B):
                t0 = 0
                if valid_mask is not None:
                    valid_t = torch.nonzero(valid_mask[i], as_tuple=False)
                    if len(valid_t) > 0:
                        t0 = int(valid_t[0].item())

                time_stem = ""
                radar_paths = batch.get("radar_paths")
                if radar_paths and i < len(radar_paths) and t0 < len(radar_paths[i]):
                    p = radar_paths[i][t0]
                    time_stem = os.path.splitext(os.path.basename(str(p)))[0].replace(os.sep, "_").replace(":", "_")
                out_name = f"{time_stem}.png" if time_stem else f"batch{batch_idx}_i{i}.png"
                out_path = os.path.join(args.out_dir, out_name)
                _save_four_panel_sat_light_radar(
                    sat_ir_video_01[i, t0],
                    sat_lgt_video_01[i, t0],
                    gt_video_01[i, t0],
                    pred_video_01[i, t0],
                    cmap_ir,
                    cmap_lgt,
                    cmap_rad,
                    out_path,
                )

            # GIF：仅在 v2v 模式下，使用 matplotlib FuncAnimation（与 Diffi2i 一致）
            if args.mode == "v2v" and T_eff > 1:
                max_samples = min(B, 8)
                gif_path = os.path.join(args.out_dir, f"batch{batch_idx}_v2v.gif")
                vmask = valid_mask[:max_samples].cpu() if valid_mask is not None else None
                try:
                    _save_v2v_gif_mpl(
                        sat_ir_video_01[:max_samples],
                        sat_lgt_video_01[:max_samples],
                        gt_video_01[:max_samples],
                        pred_video_01[:max_samples],
                        vmask,
                        gif_path,
                        cmap_ir,
                        cmap_lgt,
                        cmap_rad,
                        fps=4,
                    )
                except Exception as e:
                    print(f"[WARN] Failed to save GIF {gif_path}: {e}")

    n_batches = len(loader)
    total = (
        n_batches
        if (args.max_batches_metrics < 0 or args.max_batches_images < 0)
        else min(n_batches, max(args.max_batches_metrics, args.max_batches_images))
    )
    pbar = tqdm(enumerate(loader), total=total, desc="test_sat2radar", unit="batch")
    for b_idx, batch in pbar:
        do_metrics = args.max_batches_metrics < 0 or b_idx < args.max_batches_metrics
        do_images = args.max_batches_images < 0 or b_idx < args.max_batches_images
        if not do_metrics and not do_images:
            break
        infer_batch(batch, b_idx)
        if mse_count > 0:
            pbar.set_postfix(mse=f"{mse_total / mse_count:.4f}", refresh=False)

    # ── Aggregate metrics (match Diffi2i validate_test_ckpt.py) ──────────────
    metrics = {
        "ckpt": args.ckpt,
        "config": args.config,
        "split": args.split,
        "mode": args.mode,
        "n_frames": int(n_frames),
        "seen_samples": int(seen_samples),
        "fss_thresholds": list(thrs),
        "fss_scales": list(scales),
        "cat_thresholds": list(cat_thresholds),
    }

    if n_frames == 0:
        print("[WARN] No valid frames seen — skipping metric aggregation.")
    else:
        all_gt = np.stack(all_gt_dbz)      # [N, H, W] in dBZ
        all_pred = np.stack(all_pred_dbz)
        gt_flat = all_gt.ravel()
        pred_flat = all_pred.ravel()

        # Frame-averaged scalars (dBZ)
        mse_dbz = mse_total / mse_count
        ssim_val = ssim_total / ssim_count
        mae_dbz = mae_sum / n_frames
        rmse_dbz = float(np.sqrt(sq_sum / n_frames))
        bias_dbz = bias_sum / n_frames
        psnr_db = float(20.0 * np.log10(60.0 / rmse_dbz)) if rmse_dbz > 0 else float("inf")
        try:
            r2 = float(sk_r2(gt_flat, pred_flat))
        except Exception:
            r2 = float("nan")

        metrics.update({
            "mse_dbz": mse_dbz,
            "mae_dbz": mae_dbz,
            "rmse_dbz": rmse_dbz,
            "psnr_db": psnr_db,
            "bias_dbz": bias_dbz,
            "ssim": ssim_val,
            "r2": r2,
        })

        # FSS (accumulated) — avg + weighted
        fss_per = {}
        if PYSTEPS_AVAILABLE and pysteps_fss_objects:
            for (thr, scale), fss_obj in pysteps_fss_objects.items():
                fv = fss_compute(fss_obj)
                fss_per[(float(thr), int(scale))] = float(fv) if np.isfinite(fv) else float("nan")

            thrs_arr = np.asarray(thrs, dtype=float)
            scales_arr = np.asarray(scales, dtype=int)
            fss_matrix = np.full((len(thrs_arr), len(scales_arr)), np.nan, dtype=float)
            for (thr, scale), val in fss_per.items():
                fss_matrix[np.where(thrs_arr == thr)[0][0],
                           np.where(scales_arr == scale)[0][0]] = val
            avg_fss = float(np.nanmean(fss_matrix))

            # Diffi2i weighted-FSS scheme
            thr_weights_dict = {0: 0.5, 5: 0.7, 10: 0.9, 15: 1.0, 20: 1.2, 25: 1.3,
                                30: 1.5, 35: 2.0, 40: 2.0, 45: 1.8, 50: 1.6,
                                55: 1.4, 60: 1.2}
            thr_w = np.array([thr_weights_dict.get(int(t), 1.0) for t in thrs_arr],
                             dtype=float)
            thr_profile = np.nanmean(fss_matrix, axis=1)
            mask = ~np.isnan(thr_profile)
            if mask.any():
                w_eff = np.where(mask, thr_w, 0.0)
                w_eff = w_eff / w_eff.sum() if w_eff.sum() > 0 else w_eff
                weighted_fss = float(np.nansum(thr_profile * w_eff))
            else:
                weighted_fss = float("nan")

            metrics["avg_fss"] = avg_fss
            metrics["weighted_fss"] = weighted_fss
            metrics["fss_per_thr_scale"] = {
                f"thr{int(thr) if float(thr).is_integer() else thr}_scale{int(scale)}": v
                for (thr, scale), v in fss_per.items()
            }

        # Categorical metrics @ multiple thresholds (match Diffi2i)
        if PYSTEPS_AVAILABLE and pvdcat is not None:
            cat_per_thr = {}
            for thr in cat_thresholds:
                try:
                    sc = pvdcat.det_cat_fct(pred_flat, gt_flat, thr=float(thr),
                                            scores=["CSI", "POD", "FAR", "HSS"], axis=None)
                    cat_per_thr[str(int(thr) if float(thr).is_integer() else thr)] = {
                        "csi": float(sc["CSI"]),
                        "pod": float(sc["POD"]),
                        "far": float(sc["FAR"]),
                        "hss": float(sc["HSS"]),
                    }
                except Exception as e:
                    print(f"[WARN] det_cat_fct failed at thr={thr}: {e}")
            metrics["cat_per_thr"] = cat_per_thr
            if "35" in cat_per_thr:
                metrics["csi35"] = cat_per_thr["35"]["csi"]
                metrics["pod35"] = cat_per_thr["35"]["pod"]
                metrics["far35"] = cat_per_thr["35"]["far"]
                metrics["hss35"] = cat_per_thr["35"]["hss"]

        # Sobel mean gradient per-frame for k=1,3,5,7
        if HAS_CV2:
            grad_sobel = {}
            for k in [1, 3, 5, 7]:
                pred_g = np.array([_sobel_mean(im, k) for im in all_pred], dtype=np.float64)
                gt_g = np.array([_sobel_mean(im, k) for im in all_gt], dtype=np.float64)
                grad_sobel[f"k{k}"] = {
                    "pred_per_sample": pred_g.tolist(),
                    "gt_per_sample": gt_g.tolist(),
                    "pred_mean": float(pred_g.mean()),
                    "pred_std": float(pred_g.std()),
                    "gt_mean": float(gt_g.mean()),
                    "gt_std": float(gt_g.std()),
                }
            metrics["grad_sobel"] = grad_sobel

        # Reflectivity histogram (60 bins, 0–60 dBZ)
        hist_bins = np.arange(0, 61, 1, dtype=float)
        pred_hist, _ = np.histogram(pred_flat, bins=hist_bins, density=True)
        gt_hist, _ = np.histogram(gt_flat, bins=hist_bins, density=True)
        metrics["refl_hist_60bins"] = {
            "bin_edges": hist_bins.tolist(),
            "pred_density": pred_hist.tolist(),
            "gt_density": gt_hist.tolist(),
        }

        # ── Print summary ─────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print(f"Metrics on {n_frames} frames ({seen_samples} samples), all in dBZ:")
        print("=" * 60)
        print(f"  MSE={mse_dbz:.4f} | MAE={mae_dbz:.4f} | RMSE={rmse_dbz:.4f} | "
              f"PSNR={psnr_db:.2f} dB | bias={bias_dbz:+.4f}")
        print(f"  SSIM={ssim_val:.4f} | R²={r2:.4f}")
        if "avg_fss" in metrics:
            print(f"  avg_FSS={metrics['avg_fss']:.4f} | weighted_FSS={metrics['weighted_fss']:.4f}")
        if "csi35" in metrics:
            print(f"  CSI35={metrics['csi35']:.4f} | POD35={metrics['pod35']:.4f} | "
                  f"FAR35={metrics['far35']:.4f} | HSS35={metrics['hss35']:.4f}")
        if "grad_sobel" in metrics:
            g3 = metrics["grad_sobel"]["k3"]
            print(f"  Sobel|k3 grad: pred {g3['pred_mean']:.3f}+/-{g3['pred_std']:.3f}, "
                  f"gt {g3['gt_mean']:.3f}+/-{g3['gt_std']:.3f}")

        # ── Generation-quality metrics (FID/sFID/KID for i2i; FVD/KVD/TC for v2v) ──
        gen = {}
        for k in ("fid", "sfid", "kid", "fvd", "kvd"):
            if k in gen_metrics:
                try:
                    out = gen_metrics[k].compute()
                    if isinstance(out, (tuple, list)):
                        gen[k] = {"mean": float(out[0].item()), "std": float(out[1].item())}
                    else:
                        gen[k] = float(out.item())
                except Exception as e:
                    print(f"[WARN] failed to compute {k}: {e}")
                    gen[k] = None
        if tc_metric is not None:
            tc_val = tc_metric.compute()
            gen["tc_dbz_sq"] = tc_val
            gen["tc_count_pairs"] = int(tc_metric.tc_count)
        if gen:
            metrics["gen_metrics"] = gen
            label_line = []
            if "fid" in gen and gen["fid"] is not None:
                label_line.append(f"FID={gen['fid']:.3f}")
            if "sfid" in gen and gen["sfid"] is not None:
                label_line.append(f"sFID={gen['sfid']:.3f}")
            if "kid" in gen and gen["kid"] is not None:
                label_line.append(f"KID={gen['kid']['mean']:.4f}±{gen['kid']['std']:.4f}")
            if "fvd" in gen and gen["fvd"] is not None:
                label_line.append(f"FVD={gen['fvd']:.3f}")
            if "kvd" in gen and gen["kvd"] is not None:
                label_line.append(f"KVD={gen['kvd']['mean']:.4f}±{gen['kvd']['std']:.4f}")
            if "tc_dbz_sq" in gen and gen["tc_dbz_sq"] is not None:
                label_line.append(f"TC={gen['tc_dbz_sq']:.2f} dBZ² (n={gen['tc_count_pairs']})")
            if label_line:
                print("  " + " | ".join(label_line))

        # Per (thr, scale) FSS breakdown
        if PYSTEPS_AVAILABLE and pysteps_fss_objects:
            print("\nPer threshold-scale FSS breakdown (accumulated):")
            print(f"{'Threshold':<12} {'Scale':<8} {'FSS':<15}")
            print("-" * 40)
            for (thr, scale) in sorted(pysteps_fss_objects.keys()):
                print(f"{thr:<12.1f} {scale:<8} "
                      f"{fss_per[(float(thr), int(scale))]:<15.6f}")
        print("=" * 60)

    # ── Write metrics.json ───────────────────────────────────────────────────
    metrics_json_path = args.metrics_json or os.path.join(args.out_dir, "metrics.json")
    metrics_dir = os.path.dirname(metrics_json_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[METRICS] wrote {metrics_json_path}")

    # ── Optional: dump per-frame dBZ arrays for downstream analysis ──────────
    if args.dump_arrays and n_frames > 0:
        arrays_dir = args.arrays_dir or os.path.join(args.out_dir, "arrays")
        os.makedirs(arrays_dir, exist_ok=True)
        gt_path = os.path.join(arrays_dir, "gt_dbz.npy")
        pred_path = os.path.join(arrays_dir, "pred_dbz.npy")
        np.save(gt_path, all_gt.astype(np.float32))
        np.save(pred_path, all_pred.astype(np.float32))
        print(f"[ARRAYS] dumped {gt_path}, {pred_path} "
              f"(shape {all_gt.shape}, ~{all_gt.nbytes / 1e9:.2f} GB each)")

    print(f"\nSaved sat2radar {args.mode} predictions to {args.out_dir}")


if __name__ == "__main__":
    main()

