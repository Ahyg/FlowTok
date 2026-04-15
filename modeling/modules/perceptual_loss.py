"""This file contains perceptual loss module using LPIPS and ConvNeXt-S.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""

import os

import torch
import torch.nn.functional as F

from torchvision import models
from .lpips import LPIPS

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_convnext_small_imagenet():
    """ConvNeXt-Small ImageNet1K weights. Offline: set CONVNEXT_SMALL_IMAGENET_PTH or stage under TORCH_HOME/hub/checkpoints/."""
    local = os.environ.get("CONVNEXT_SMALL_IMAGENET_PTH", "").strip()
    if not local:
        th = os.environ.get("TORCH_HOME", "").strip()
        if th:
            ckpt_dir = os.path.join(th, "hub", "checkpoints")
            for name in (
                "convnext_small-0c510722.pth",
                "convnext_small-47c16186.pth",
            ):
                cand = os.path.join(ckpt_dir, name)
                if os.path.isfile(cand):
                    local = cand
                    break
    m = models.convnext_small(weights=None)
    if local:
        if not os.path.isfile(local):
            raise FileNotFoundError(
                f"CONVNEXT_SMALL_IMAGENET_PTH is set but file not found: {local}"
            )
        print(f"[INFO] Loading ConvNeXt-Small from local weights: {local}")
        sd = torch.load(local, map_location="cpu")
        if isinstance(sd, dict):
            if "state_dict" in sd:
                sd = sd["state_dict"]
            elif "model" in sd:
                sd = sd["model"]
        m.load_state_dict(sd, strict=True)
        return m
    return models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)


class PerceptualLoss(torch.nn.Module):
    def __init__(self, model_name: str = "convnext_s", per_channel: bool = False):
        """Initializes the PerceptualLoss class.

        Args:
            model_name: A string, the name of the perceptual loss model to use.
            per_channel: If True, run LPIPS/ConvNeXt on each channel independently
                (replicated to 3-channel grayscale-RGB) and average the losses.
                If False (default), fall back to the legacy behavior that adapts
                non-3-channel inputs via replicate (C=1) or channel-drop
                (C>=7: [0,2,6]; 4<=C<7: [:3]).

        Raise:
            ValueError: If the model_name does not contain "lpips" or "convnext_s".
        """
        super().__init__()
        if ("lpips" not in model_name) and (
            "convnext_s" not in model_name):
            raise ValueError(f"Unsupported Perceptual Loss model name {model_name}")
        self.per_channel = bool(per_channel)
        self.lpips = None
        self.convnext = None
        self.loss_weight_lpips = None
        self.loss_weight_convnext = None

        # Parsing the model name. We support name formatted in
        # "lpips-convnext_s-{float_number}-{float_number}", where the 
        # {float_number} refers to the loss weight for each component.
        # E.g., lpips-convnext_s-1.0-2.0 refers to compute the perceptual loss
        # using both the convnext_s and lpips, and average the final loss with
        # (1.0 * loss(lpips) + 2.0 * loss(convnext_s)) / (1.0 + 2.0).
        if "lpips" in model_name:
            self.lpips = LPIPS().eval()

        if "convnext_s" in model_name:
            self.convnext = _load_convnext_small_imagenet().eval()

        if "lpips" in model_name and "convnext_s" in model_name:
            loss_config = model_name.split('-')[-2:]
            self.loss_weight_lpips, self.loss_weight_convnext = float(loss_config[0]), float(loss_config[1])
            print(f"self.loss_weight_lpips, self.loss_weight_convnext: {self.loss_weight_lpips}, {self.loss_weight_convnext}")

        self.register_buffer("imagenet_mean", torch.Tensor(_IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("imagenet_std", torch.Tensor(_IMAGENET_STD)[None, :, None, None])

        for param in self.parameters():
            param.requires_grad = False
    
    def _compute_loss_3ch(self, input_3ch: torch.Tensor, target_3ch: torch.Tensor):
        """Compute LPIPS + ConvNeXt loss on a 3-channel tensor [B, 3, H, W]."""
        loss = 0.
        num_losses = 0.
        if self.lpips is not None:
            lpips_loss = self.lpips(input_3ch, target_3ch)
            if self.loss_weight_lpips is None:
                loss = loss + lpips_loss
                num_losses += 1
            else:
                num_losses += self.loss_weight_lpips
                loss = loss + self.loss_weight_lpips * lpips_loss

        if self.convnext is not None:
            input_up = torch.nn.functional.interpolate(
                input_3ch, size=224, mode="bilinear", align_corners=False, antialias=True)
            target_up = torch.nn.functional.interpolate(
                target_3ch, size=224, mode="bilinear", align_corners=False, antialias=True)
            pred_input = self.convnext((input_up - self.imagenet_mean) / self.imagenet_std)
            pred_target = self.convnext((target_up - self.imagenet_mean) / self.imagenet_std)
            convnext_loss = torch.nn.functional.mse_loss(
                pred_input, pred_target, reduction="mean")
            if self.loss_weight_convnext is None:
                num_losses += 1
                loss = loss + convnext_loss
            else:
                num_losses += self.loss_weight_convnext
                loss = loss + self.loss_weight_convnext * convnext_loss

        return loss / num_losses

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Computes the perceptual loss.

        Two modes controlled by ``self.per_channel``:
          - ``per_channel=True``: every input channel is replicated to a
            3-channel grayscale-RGB tensor, fed through the perceptual backbone,
            and the per-channel losses are averaged. All channels (including
            sparse ones like lightning) receive structural gradients.
          - ``per_channel=False`` (default, legacy): adapt non-3-channel inputs
            by replicating (C=1) or channel-dropping (C>=7 -> [0,2,6];
            4<=C<7 -> [:3]), then run one forward. Preserved to keep
            reproducibility of experiments trained before the per-channel switch.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        # Always in eval mode.
        self.eval()
        B, C, H, W = input.shape

        if self.per_channel:
            if C == 1:
                return self._compute_loss_3ch(
                    input.repeat(1, 3, 1, 1), target.repeat(1, 3, 1, 1))
            total = None
            for c in range(C):
                x = input[:, c:c + 1].repeat(1, 3, 1, 1)
                y = target[:, c:c + 1].repeat(1, 3, 1, 1)
                loss_c = self._compute_loss_3ch(x, y)
                total = loss_c if total is None else total + loss_c
            return total / float(C)

        # Legacy path: pre-2026-04 behavior.
        if C != 3:
            if C == 1:
                input = input.repeat(1, 3, 1, 1)
                target = target.repeat(1, 3, 1, 1)
            elif C >= 7:
                input = input[:, [0, 2, 6], ...]
                target = target[:, [0, 2, 6], ...]
            else:
                input = input[:, :3]
                target = target[:, :3]
        return self._compute_loss_3ch(input, target)