"""Generation-quality metrics shared by sat2radar i2i / v2v test scripts.

i2i: FID + sFID + KID (torchmetrics, InceptionV3 features).
v2v: FVD + KVD (torchmetrics FID/KID with I3D-Kinetics-400 features) and
     Temporal Consistency via RAFT optical-flow warping.

Single-channel radar in [0,1] is replicated to 3-channel RGB before being
fed to any pretrained backbone (Inception / I3D / RAFT).
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    TORCHMETRICS_OK = True
except ImportError:
    TORCHMETRICS_OK = False
    FrechetInceptionDistance = None
    KernelInceptionDistance = None

try:
    from pytorchvideo.models.hub import i3d_r50
    PYTORCHVIDEO_OK = True
except ImportError:
    PYTORCHVIDEO_OK = False
    i3d_r50 = None

try:
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    RAFT_OK = True
except ImportError:
    RAFT_OK = False
    raft_small = None
    Raft_Small_Weights = None


I3D_MIN_T = 8  # I3D-R50 temporal max-pool needs T >= 8


def radar_to_3ch(x: torch.Tensor) -> torch.Tensor:
    """Replicate a 1-channel radar tensor along the channel dim to 3 channels.

    Accepts [B, 1, H, W] or [B, T, 1, H, W] and returns the same shape with C=3.
    Values are clamped to [0, 1] (the convention assumed by torchmetrics with
    normalize=True and the RAFT/I3D preprocessing below).
    """
    x = x.clamp(0.0, 1.0)
    if x.dim() == 4:
        return x.repeat(1, 3, 1, 1)
    if x.dim() == 5:
        return x.repeat(1, 1, 3, 1, 1)
    raise ValueError(f"radar_to_3ch expects 4D or 5D tensor, got {tuple(x.shape)}")


class I3DFeatures(nn.Module):
    """I3D-R50 (Kinetics-400) wrapped as a torchmetrics-compatible feature extractor.

    Accepts:
      - 4D ``[B, 3, H, W]`` — single image (only used by torchmetrics' shape probe);
        padded along time to ``T=I3D_MIN_T`` by frame replication.
      - 5D ``[B, T, 3, H, W]`` — real video clip; resized spatially to 224x224.

    Returns ``[B, 2048]`` Kinetics-400 features.
    """

    def __init__(self):
        super().__init__()
        if not PYTORCHVIDEO_OK:
            raise RuntimeError("pytorchvideo not available; install via `pip install pytorchvideo`")
        net = i3d_r50(pretrained=True)
        net.blocks[-1].proj = nn.Identity()
        net.blocks[-1].dropout = nn.Identity()
        net.train(False)
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)  # [B, 1, 3, H, W]
        if x.dim() != 5:
            raise ValueError(f"I3DFeatures expects 4D/5D tensor, got {tuple(x.shape)}")
        B, T, C, H, W = x.shape
        if T < I3D_MIN_T:
            pad = I3D_MIN_T - T
            x = torch.cat([x, x[:, -1:].expand(B, pad, C, H, W)], dim=1)
            T = I3D_MIN_T
        if H != 224 or W != 224:
            x = x.reshape(B * T, C, H, W)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = x.reshape(B, T, C, 224, 224)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, 224, 224]
        return self.net(x)


def make_i2i_metrics(
    device: torch.device,
    use_fid: bool = True,
    use_sfid: bool = True,
    use_kid: bool = True,
    kid_subsets: int = 50,
    kid_subset_size: int = 100,
) -> Dict[str, object]:
    """Build i2i generation metrics (FID, sFID, KID) on ``device``."""
    if not TORCHMETRICS_OK:
        warnings.warn("torchmetrics missing — i2i FID/sFID/KID disabled")
        return {}
    out: Dict[str, object] = {}
    if use_fid:
        out["fid"] = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    if use_sfid:
        # sFID: FID computed on InceptionV3 intermediate (smaller, spatially-preserving)
        # feature pool, following the DiT / ADM convention.
        out["sfid"] = FrechetInceptionDistance(feature=192, normalize=True).to(device)
    if use_kid:
        out["kid"] = KernelInceptionDistance(
            feature=2048, normalize=True,
            subsets=kid_subsets, subset_size=kid_subset_size,
        ).to(device)
    return out


def make_v2v_metrics(
    device: torch.device,
    use_fvd: bool = True,
    use_kvd: bool = True,
    kid_subsets: int = 50,
    kid_subset_size: int = 50,
) -> Dict[str, object]:
    """Build v2v generation metrics (FVD, KVD) using I3D features on ``device``."""
    if not TORCHMETRICS_OK:
        warnings.warn("torchmetrics missing — v2v FVD/KVD disabled")
        return {}
    if not PYTORCHVIDEO_OK:
        warnings.warn("pytorchvideo missing — v2v FVD/KVD disabled")
        return {}
    # Single shared I3D backbone for both metrics (saves GPU memory & weight loading)
    i3d = I3DFeatures().to(device)
    out: Dict[str, object] = {"_i3d": i3d}
    if use_fvd:
        out["fvd"] = FrechetInceptionDistance(feature=i3d, normalize=True).to(device)
    if use_kvd:
        out["kvd"] = KernelInceptionDistance(
            feature=i3d, normalize=True,
            subsets=kid_subsets, subset_size=kid_subset_size,
        ).to(device)
    return out


class TemporalConsistency:
    """RAFT-based temporal consistency: warp pred_t with GT optical flow and
    compare to pred_{t+1}. Lower is better; 0 means perfectly consistent.

    Accumulates squared error in the dBZ domain (× 60 if inputs are in [0,1])
    so it is comparable to ``mse_dbz`` reported elsewhere.
    """

    def __init__(self, device: torch.device, dbz_scale: float = 60.0):
        if not RAFT_OK:
            raise RuntimeError("torchvision RAFT not available")
        self.device = device
        self.dbz_scale = dbz_scale
        weights = Raft_Small_Weights.DEFAULT
        self.transforms = weights.transforms()
        m = raft_small(weights=weights, progress=False)
        m.train(False)
        for p in m.parameters():
            p.requires_grad_(False)
        self.model = m.to(device)
        self.tc_sq_sum: float = 0.0
        self.tc_count: int = 0
        self.per_pair_history: list = []  # optional per-batch means

    @staticmethod
    def _warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp ``img`` [B, C, H, W] forward by ``flow`` [B, 2, H, W] (x then y, pixels)."""
        B, _, H, W = img.shape
        yy, xx = torch.meshgrid(
            torch.arange(H, device=img.device, dtype=img.dtype),
            torch.arange(W, device=img.device, dtype=img.dtype),
            indexing="ij",
        )
        base = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        grid_pix = base + flow
        grid = torch.zeros(B, H, W, 2, device=img.device, dtype=img.dtype)
        grid[..., 0] = grid_pix[:, 0] / max(W - 1, 1) * 2 - 1
        grid[..., 1] = grid_pix[:, 1] / max(H - 1, 1) * 2 - 1
        return F.grid_sample(img, grid, mode="bilinear",
                             padding_mode="border", align_corners=True)

    @torch.no_grad()
    def update(
        self,
        gt_clip_01: torch.Tensor,
        pred_clip_01: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Optional[float]:
        """Accumulate TC over all valid adjacent frame pairs in a clip batch.

        Args:
          gt_clip_01:   [B, T, 1, H, W] in [0, 1]
          pred_clip_01: [B, T, 1, H, W] in [0, 1]
          valid_mask:   optional [B, T] bool — a pair (t, t+1) is valid only if both ends are.

        Returns the batch-mean TC (in dBZ²) if any pair was processed, else None.
        """
        if gt_clip_01.dim() != 5 or pred_clip_01.dim() != 5:
            raise ValueError("expected 5D [B,T,1,H,W] tensors")
        B, T, C, H, W = gt_clip_01.shape
        if T < 2:
            return None
        gt_clip = gt_clip_01.to(self.device).clamp(0.0, 1.0)
        pred_clip = pred_clip_01.to(self.device).clamp(0.0, 1.0)

        # Pair mask: pair t valid iff frame t and t+1 are both valid
        if valid_mask is not None:
            vm = valid_mask.to(self.device).bool()
            pair_mask = vm[:, :-1] & vm[:, 1:]  # [B, T-1]
        else:
            pair_mask = torch.ones(B, T - 1, dtype=torch.bool, device=self.device)
        if not pair_mask.any():
            return None

        # Build all (t, t+1) pairs in a single RAFT batch.
        # RAFT needs H,W >= 128 (feature map ≥ 16 after /8 stride). Upsample if needed,
        # run flow at the larger resolution, then downsample flow back & rescale magnitude.
        RAFT_MIN_HW = 128
        scale = max(RAFT_MIN_HW / H, RAFT_MIN_HW / W, 1.0)
        if scale > 1.0:
            H_r = int(round(H * scale)); W_r = int(round(W * scale))
        else:
            H_r, W_r = H, W
        gt_3 = gt_clip.repeat(1, 1, 3, 1, 1)  # [B,T,3,H,W]
        a = gt_3[:, :-1].reshape(B * (T - 1), 3, H, W)
        b = gt_3[:, 1:].reshape(B * (T - 1), 3, H, W)
        if scale > 1.0:
            a = F.interpolate(a, size=(H_r, W_r), mode="bilinear", align_corners=False)
            b = F.interpolate(b, size=(H_r, W_r), mode="bilinear", align_corners=False)
        a_in, b_in = self.transforms(a, b)
        flow = self.model(a_in, b_in)[-1]  # [B*(T-1), 2, H_r, W_r] in pixels
        if scale > 1.0:
            flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
            # Rescale flow magnitude back to original-pixel units
            flow[:, 0] *= W / W_r
            flow[:, 1] *= H / H_r

        p_t = pred_clip[:, :-1].reshape(B * (T - 1), 1, H, W)
        p_t1 = pred_clip[:, 1:].reshape(B * (T - 1), 1, H, W)
        p_t_warped = self._warp(p_t, flow)

        # Per-pair MSE in dBZ domain (scale 0..60), then mask
        sq = ((p_t_warped - p_t1) * self.dbz_scale).pow(2).mean(dim=(1, 2, 3))  # [B*(T-1)]
        sq = sq.view(B, T - 1)
        sq = sq[pair_mask]
        n = int(sq.numel())
        if n == 0:
            return None
        s = float(sq.sum().item())
        self.tc_sq_sum += s
        self.tc_count += n
        mean_this = s / n
        self.per_pair_history.append(mean_this)
        return mean_this

    def compute(self) -> Optional[float]:
        if self.tc_count == 0:
            return None
        return self.tc_sq_sum / self.tc_count
