"""Canonical FVD/KVD video-feature extractor (Inflated-Inception-v1 I3D, Kinetics-400).

This is the *de-facto* PyTorch FVD network — the TorchScript ``i3d_torchscript.pt``
re-implementation of the DeepMind Kinetics-400 I3D used by StyleGAN-V
(``src/metrics/frechet_video_distance.py``), videoGPT, Open-Sora, and the widely
used ``JunyaoHu/common_metrics_on_video_quality`` reference. Absolute FVD/KVD
computed with this backbone are directly comparable to published numbers.

It REPLACES the previous ``I3DFeatures`` (pytorchvideo ``i3d_r50`` / I3D-ResNet-50,
2048-d), whose features are *not* the canonical FVD network and therefore not
comparable to any published FVD.

Canonical convention matched exactly (from ``common_metrics_on_video_quality``):
  * detector input layout : ``[B, C, T, 224, 224]`` (BCTHW)
  * value range           : ``[-1, 1]``  via ``(x - 0.5) * 2`` on a ``[0, 1]`` clip
  * spatial preprocessing : resize shorter side to 224 (bilinear, align_corners=False)
                            then center-crop 224x224
  * detector call         : ``i3d(x, rescale=False, resize=False, return_features=True)``
  * output feature dim    : ``400``  (Kinetics-400 logits-space features, pre-softmax)
  * minimum frames        : T >= 9  (the TorchScript net's temporal pooling; verified).
                            Our sat->radar clips are 16 frames, so this is satisfied;
                            shorter inputs are padded by temporal frame replication.

Weights (``i3d_torchscript.pt``, ~48.8 MB) are loaded fully offline from a local
cache — no runtime download. Original source:
``https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1``
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Weight-file resolution (offline only; compute nodes have no internet).
# ---------------------------------------------------------------------------
_I3D_FILENAME = "i3d_torchscript.pt"

# Feature dimension of the canonical Kinetics-400 I3D (pre-softmax logits space).
I3D_FEATURE_DIM = 400

# TorchScript temporal pooling requires at least this many frames (verified: T<9 raises).
I3D_MIN_T = 9


def _candidate_ckpt_paths() -> list[str]:
    """Ordered list of local paths to try for ``i3d_torchscript.pt`` (offline)."""
    cands: list[str] = []
    # 1) explicit override
    env_explicit = os.environ.get("I3D_TORCHSCRIPT_PATH")
    if env_explicit:
        cands.append(env_explicit)
    # 2) $TORCH_HOME/hub/checkpoints/
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        cands.append(os.path.join(torch_home, "hub", "checkpoints", _I3D_FILENAME))
    # 3) the two standard gen-metric cache dirs on this workspace
    cands.append(f"/scratch/kl02/yh0308/hf_cache/hub/checkpoints/{_I3D_FILENAME}")
    cands.append(f"/g/data/kl02/yh0308/.cache/torch/hub/checkpoints/{_I3D_FILENAME}")
    # de-dup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _resolve_ckpt(ckpt_path: Optional[str] = None) -> str:
    paths = [ckpt_path] if ckpt_path else []
    paths += _candidate_ckpt_paths()
    for p in paths:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"{_I3D_FILENAME} not found in any cache dir. Tried: {paths}. "
        "Download it on a login node from "
        "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1 "
        "and place it under $TORCH_HOME/hub/checkpoints/ (compute nodes are offline)."
    )


class I3DCanonicalFeatures(nn.Module):
    """Canonical FVD/KVD feature extractor (Kinetics-400 Inflated-Inception I3D).

    Drop-in replacement for the old ``I3DFeatures``: same input contract, but
    returns the *canonical* 400-d I3D features instead of I3D-R50 2048-d.

    Input contract (matches the previous ``I3DFeatures.forward`` so wiring is a
    drop-in for ``torchmetrics`` FID/KID with ``feature=<this module>``):

      * 5D ``[B, T, C, H, W]`` — a real video clip (the primary path). ``C`` is 3
        (single-channel radar is replicated to RGB upstream). If instead the
        channel-first layout ``[B, C, T, H, W]`` is passed, it is auto-detected
        (whichever of dim-1 / dim-2 equals 3 is the channel axis) and permuted.
      * 4D ``[B, C, H, W]`` — a single image, only used by torchmetrics' optional
        shape probe; a time axis is added and the clip is padded to ``I3D_MIN_T``.

    Returns ``[B, 400]`` Kinetics-400 features (pre-softmax logits space).

    ``num_features = 400`` is exposed as an int attribute so torchmetrics reads
    the feature dimension directly and SKIPS its random-tensor probe — the probe
    would feed a single frame (T=1) and crash this net (needs T >= 9).
    """

    # torchmetrics FID/KID read this to size their running-stat buffers and skip
    # the probe forward (see torchmetrics/image/fid.py: hasattr(feature,"num_features")).
    num_features: int = I3D_FEATURE_DIM

    def __init__(self, ckpt_path: Optional[str] = None):
        super().__init__()
        resolved = _resolve_ckpt(ckpt_path)
        # Fully offline TorchScript load.
        i3d = torch.jit.load(resolved, map_location="cpu")
        i3d.eval()
        for p in i3d.parameters():
            p.requires_grad_(False)
        self.i3d = i3d
        self.ckpt_path = resolved

    # -- helpers -------------------------------------------------------------
    def _device(self) -> torch.device:
        try:
            return next(self.i3d.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _resize_center_crop(frames: torch.Tensor, res: int = 224) -> torch.Tensor:
        """Resize shorter side to ``res`` (bilinear), then center-crop ``res``x``res``.

        ``frames``: ``[N, C, H, W]``. Matches common_metrics ``preprocess_single``.
        """
        _, _, h, w = frames.shape
        if h != res or w != res:
            scale = res / min(h, w)
            if h < w:
                target = (res, int(math.ceil(w * scale)))
            else:
                target = (int(math.ceil(h * scale)), res)
            frames = F.interpolate(frames, size=target, mode="bilinear", align_corners=False)
            _, _, h2, w2 = frames.shape
            hs = (h2 - res) // 2
            ws = (w2 - res) // 2
            frames = frames[:, :, hs:hs + res, ws:ws + res]
        return frames

    # -- forward -------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torchmetrics does not move inputs to the feature module's device.
        x = x.to(self._device())

        if x.dim() == 4:
            # [B, C, H, W] probe -> single-frame clip [B, T=1, C, H, W]
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(
                f"I3DCanonicalFeatures expects 4D/5D tensor, got {tuple(x.shape)}"
            )

        # Normalize layout to [B, T, C, H, W].
        d1, d2 = x.shape[1], x.shape[2]
        if d1 == 3 and d2 != 3:
            # channel-first [B, C, T, H, W] -> [B, T, C, H, W]
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        # else: already [B, T, C, H, W] (also the ambiguous C==T==3 case: keep as-is)

        B, T, C, H, W = x.shape

        # Radar/RGB clip is in [0, 1] upstream; be defensive.
        x = x.clamp(0.0, 1.0)

        # Pad temporally to the net's minimum by replicating the last frame.
        if T < I3D_MIN_T:
            pad = I3D_MIN_T - T
            x = torch.cat([x, x[:, -1:].expand(B, pad, C, H, W)], dim=1)
            T = I3D_MIN_T

        # Spatial: resize shorter side to 224 + center crop (per-frame).
        x = x.reshape(B * T, C, H, W)
        x = self._resize_center_crop(x, 224)
        x = x.reshape(B, T, C, 224, 224)

        # Value range: [0, 1] -> [-1, 1].
        x = (x - 0.5) * 2.0

        # Detector expects BCTHW.
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, 224, 224]

        feats = self.i3d(x, rescale=False, resize=False, return_features=True)
        return feats  # [B, 400]
