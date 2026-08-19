"""Spatial-FID (sFID) feature extractor for the sat2radar i2i generation metrics.

sFID ("spatial FID", Nash et al. 2021, *Generating Images with Sparse
Representations*; adopted by ADM / OpenAI guided-diffusion and DiT) is the
Frechet distance computed on an *intermediate spatial* Inception activation
that is flattened **without** global average pooling, so the spatial layout of
the feature map is preserved.

Canonical reference — OpenAI guided-diffusion ``evaluations/evaluator.py``:

    FID_POOL_NAME    = "pool_3:0"          # -> 2048-d  (standard FID)
    FID_SPATIAL_NAME = "mixed_6/conv:0"    # spatial features for sFID
    ...
    spatial = spatial[..., :7]             # keep first 7 channels
    ...
    preds.append(spatial_pred.reshape([spatial_pred.shape[0], -1]))

That graph (``classify_image_graph_def.pb`` / inception-2015-12-05) resizes the
input to 299x299 internally, so ``mixed_6/conv`` is a 17x17 spatial map.  With
the first-7-channel slice the flattened feature is therefore 7 * 17 * 17 = 2023.

TF -> PyTorch layer mapping
---------------------------
The frozen TF graph names its Inception blocks ``mixed, mixed_1, ... mixed_10``.
The four 17x17 "figure-6" (InceptionC) blocks are ``mixed_4, mixed_5, mixed_6,
mixed_7`` which correspond to PyTorch ``Mixed_6b, Mixed_6c, Mixed_6d, Mixed_6e``.
Hence ``mixed_6`` == PyTorch ``Mixed_6d``.

``mixed_6/conv:0`` is the ``branch1x1`` (1x1 conv) branch of that block, and the
PyTorch InceptionC concatenates its branches as
``cat([branch1x1, branch7x7, branch7x7dbl, branch_pool], dim=1)``.  The first 7
channels of the *block output* are therefore exactly the first 7 channels of
``branch1x1`` == ``mixed_6/conv[..., :7]``.  So slicing ``Mixed_6d`` output
``[:, :7]`` reproduces the ADM spatial feature.  Grid = 17x17 -> dim = 2023.

Weights
-------
This module reuses the already-cached TF-ported FID Inception weights
(``weights-inception-2015-12-05-6726825d.pth``, the pytorch-fid / torch-fidelity
Inception) by instantiating torchmetrics' ``NoTrainInceptionV3`` exactly as the
plain FID metric does — no new weights are introduced.  Loading is fully offline
as long as that .pth is present in ``$TORCH_HOME/hub/checkpoints/``.

Preprocessing
-------------
torchmetrics does NOT apply its ``normalize=True`` (float[0,1] -> uint8[0,255])
preprocessing when ``feature`` is a *custom* ``nn.Module`` (see
``torchmetrics/image/fid.py``: ``used_custom_model`` is set and line
``imgs = (imgs*255).byte() if self.normalize and (not self.used_custom_model)``
leaves the input untouched).  So this module performs the FID-Inception
preprocessing itself: it takes float [0,1] ``[B,3,H,W]`` input, converts it to
the uint8 [0,255] range that ``NoTrainInceptionV3`` expects, and lets the
wrapped model do its own internal resize(299) + normalization.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from torchmetrics.image.fid import NoTrainInceptionV3

# PyTorch block that matches the TF `mixed_6` node used by ADM/guided-diffusion.
_SPATIAL_BLOCK = "Mixed_6d"
# Number of leading channels kept (ADM: `spatial = spatial[..., :7]`).
_KEEP_CHANNELS = 7
# 17x17 grid at 299x299 input -> 7 * 17 * 17.
SFID_FEATURE_DIM = 2023


class SpatialInceptionFeatures(nn.Module):
    """torchmetrics-compatible spatial-Inception feature extractor for sFID.

    ``forward(imgs) -> [B, 2023]`` where ``imgs`` is a float ``[B, 3, H, W]``
    tensor in ``[0, 1]`` (single-channel radar already replicated to 3 channels
    upstream).  The returned features are the first-7-channel slice of the
    17x17 ``Mixed_6d`` activation, flattened without global pooling, matching
    the OpenAI guided-diffusion / ADM sFID "spatial" feature.

    Exposes ``num_features`` so ``FrechetInceptionDistance(feature=...)`` sizes
    its statistics directly and skips the dtype-sensitive dummy probe.
    """

    #: read by FrechetInceptionDistance to size its covariance state
    num_features: int = SFID_FEATURE_DIM

    def __init__(self) -> None:
        super().__init__()
        # Reuses the cached weights-inception-2015-12-05-6726825d.pth (offline).
        # features_list=["2048"] runs the full forward (through Mixed_6d) so the
        # hook below fires; the returned 2048 vector is ignored.
        self.inception = NoTrainInceptionV3(
            name="inception-v3-compat", features_list=["2048"]
        )
        self.inception.eval()
        for p in self.inception.parameters():
            p.requires_grad_(False)

        self._spatial: torch.Tensor | None = None
        block = getattr(self.inception, _SPATIAL_BLOCK)
        block.register_forward_hook(self._capture_hook)

    def _capture_hook(self, _module, _inp, output: torch.Tensor) -> None:
        # output: [B, 768, 17, 17]  (Mixed_6d block, concatenated branches)
        self._spatial = output

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        # torchmetrics' FID does not move inputs to the feature module's device
        # for custom modules, so align here.
        dev = next(self.inception.parameters()).device
        imgs = imgs.to(dev)

        # Do the FID-Inception preprocessing ourselves: float[0,1] -> uint8[0,255].
        # (No-op if a uint8 tensor is passed, e.g. by an internal shape probe.)
        if imgs.dtype != torch.uint8:
            imgs = (imgs.clamp(0.0, 1.0) * 255).to(torch.uint8)

        self._spatial = None
        # Trigger the forward (NoTrainInceptionV3 resizes to 299 + normalizes
        # internally); the hook captures the Mixed_6d spatial activation.
        _ = self.inception(imgs)
        if self._spatial is None:
            raise RuntimeError(
                f"sFID hook on {_SPATIAL_BLOCK} did not fire; "
                "NoTrainInceptionV3 layout may have changed."
            )
        spatial = self._spatial[:, :_KEEP_CHANNELS]          # [B, 7, 17, 17]
        self._spatial = None
        return spatial.reshape(spatial.shape[0], -1).float()  # [B, 2023]
