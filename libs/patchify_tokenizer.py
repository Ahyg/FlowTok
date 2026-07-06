"""Parameter-free pixel patchify/unpatchify drop-in for FlowTiTok (Ablation 1: "pixfact").

Ablation 1 runs the factorized DiT + flow-matching in PIXEL space instead of the
learned FlowTiTok token space. This module mimics the two methods the train/eval
pipeline calls on a FlowTiTok autoencoder -- ``encode`` and ``decode_tokens`` --
but performs an EXACTLY-INVERTIBLE space-to-depth patchify (fold) / depth-to-space
unpatchify, so there is no learned or lossy tokenizer. Pixels are mapped
[0,1] -> [-1,1] on encode (to match the flow's N(0,1) prior) and back on decode.

Contract matched (see scripts/train_sat2radar_v2v.py:106 encode and :1375 decode):
  encode(x)                             -> (z, dict);  z: [N, C*P*P, 1, L], L=(H/P)*(W/P)
  decode_tokens(z, text_guidance, output_size=None) -> [N, C, H, W]

A ``.config`` shim exposes exactly the attributes ``encode_video_with_autoencoder``
reads (``dataset['crop_size']``, ``vq_model['in_channels']``, ``ae_image_size``), so
the patchifier is a true drop-in: target_size resolves to crop_size (no resize) and
in_channels==C (no 1->3 channel repeat). The module is parameter-free, so .eval(),
.to(device) and .requires_grad_(False) are harmless no-ops.
"""
import types

import torch
import torch.nn as nn
from einops import rearrange


class PixelPatchifier(nn.Module):
    def __init__(self, patch_size: int, channels: int, crop_size: int = 128):
        super().__init__()
        self.p = int(patch_size)
        self.c = int(channels)
        self.crop_size = int(crop_size)
        assert self.crop_size % self.p == 0, "crop_size must be divisible by patch_size"
        self.hw = self.crop_size // self.p  # tokens per spatial axis (square frames)
        # --- config shim so encode_video_with_autoencoder() treats us like a FlowTiTok ---
        self.config = types.SimpleNamespace(
            dataset={"crop_size": self.crop_size},  # target_size == crop_size -> no resize
            vq_model={"in_channels": self.c},        # expected_in_ch == C -> no 1->3 repeat
            ae_image_size=self.crop_size,
        )

    @torch.no_grad()
    def encode(self, x):
        # x: [N, C, H, W] in [0,1]; return (z, None) with z: [N, C*P*P, 1, L]
        x = x * 2.0 - 1.0
        z = rearrange(x, "n c (h p1) (w p2) -> n (c p1 p2) (h w)", p1=self.p, p2=self.p)
        return z.unsqueeze(2), None

    @torch.no_grad()
    def decode_tokens(self, z, text_guidance=None, output_size=None):
        # z: [N, C*P*P, 1, L]; return [N, C, H, W] in [0,1]
        z = z.squeeze(2)
        x = rearrange(
            z, "n (c p1 p2) (h w) -> n c (h p1) (w p2)",
            p1=self.p, p2=self.p, c=self.c, h=self.hw, w=self.hw,
        )
        x = (x + 1.0) * 0.5
        return torch.clamp(x, 0.0, 1.0)

    # alias for FlowTiTok parity (the v2v path uses decode_tokens; kept for safety)
    def decode(self, z, text_guidance=None, output_size=None):
        return self.decode_tokens(z, text_guidance, output_size)


if __name__ == "__main__":
    for P, C in [(8, 1), (8, 11), (16, 1)]:
        m = PixelPatchifier(P, C, crop_size=128)
        x = torch.rand(5, C, 128, 128)
        z, _ = m.encode(x)
        L = (128 // P) ** 2
        assert z.shape == (5, C * P * P, 1, L), (tuple(z.shape), (5, C * P * P, 1, L))
        xr = m.decode_tokens(z)
        err = (xr - x).abs().max().item()
        assert err < 1e-5, f"round-trip err {err} for P={P} C={C}"
        print(f"OK P={P} C={C}: z={tuple(z.shape)} round-trip max-err={err:.2e}")
