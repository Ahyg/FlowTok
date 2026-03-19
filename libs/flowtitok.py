"""This file contains the model definition of TA-TiTok.

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

import logging
import os
import torch
import torch.nn as nn
from einops import rearrange
from torch.cuda.amp import autocast

from libs.model.blocks import FlowTiTokEncoder, FlowTiTokDecoder

logger = logging.getLogger(__name__)


class DiagonalGaussianDistribution(object):
    @autocast(enabled=False)
    def __init__(self, parameters, deterministic=False):
        """Initializes a Gaussian distribution instance given the parameters.

        Args:
            parameters (torch.Tensor): The parameters for the Gaussian distribution. It is expected
                to be in shape [B, 2 * C, *], where B is batch size, and C is the embedding dimension.
                First C channels are used for mean and last C are used for logvar in the Gaussian distribution.
            deterministic (bool): Whether to use deterministic sampling. When it is true, the sampling results
                is purely based on mean (i.e., std = 0).
        """
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters.float(), 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    @autocast(enabled=False)
    def sample(self):
        x = self.mean.float() + self.std.float() * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    @autocast(enabled=False)
    def mode(self):
        return self.mean

    @autocast(enabled=False)
    def kl(self):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            return 0.5 * torch.sum(torch.pow(self.mean.float(), 2)
                                    + self.var.float() - 1.0 - self.logvar.float(),
                                    dim=[1, 2])


class FlowTiTok(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.encoder = FlowTiTokEncoder(config)
        self.decoder = FlowTiTokDecoder(config)

        self.num_latent_tokens = config.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)

        self.quantize = DiagonalGaussianDistribution

        self.return_quantized = config.vq_model.get("return_quantized", True)
        self.use_pretrained = config.vq_model.get("use_pretrained", True)
        self.nan_debug = config.vq_model.get("nan_debug", False)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def extract_latents(self, x):
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        if not self.return_quantized:
            z = z.squeeze(2).permute(0,2,1)
            if self.use_pretrained:
                z = z.view(z.shape[0], z.shape[1], self.encoder.token_size // 2, -1)
                z = z.mean(-1)
            return z
        else:
            posteriors = self.quantize(z)
            z_quantized = posteriors.sample()
            z_quantized = z_quantized.squeeze(2).permute(0,2,1)
            return z_quantized

    def encode(self, x):
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        if self.nan_debug and not torch.isfinite(z).all():
            raise RuntimeError("Non-finite detected after encoder output.")
        posteriors = self.quantize(z)
        if self.nan_debug:
            if hasattr(posteriors, "mean") and hasattr(posteriors, "logvar"):
                if (not torch.isfinite(posteriors.mean).all()) or (not torch.isfinite(posteriors.logvar).all()):
                    raise RuntimeError("Non-finite detected in posteriors (mean/logvar).")
        z_quantized = posteriors.sample()
        if self.nan_debug and not torch.isfinite(z_quantized).all():
            raise RuntimeError("Non-finite detected after posterior sampling.")
        result_dict = posteriors
        return z_quantized, result_dict

    def decode(self, z_quantized, text_guidance):
        decoded = self.decoder(z_quantized, text_guidance)
        if self.nan_debug and not torch.isfinite(decoded).all():
            raise RuntimeError("Non-finite detected after decoder output.")
        return decoded
    
    def decode_tokens(self, tokens, text_guidance):
        z_quantized = tokens
        decoded = self.decode(z_quantized, text_guidance)
        return decoded
    
    def forward(self, x, text_guidance):
        z_quantized, result_dict = self.encode(x)
        decoded = self.decode(z_quantized, text_guidance)
        return decoded, result_dict

    def save_pretrained_weight(self, output_dir, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        save_function = kwargs.get("save_function", torch.save)
        save_function(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    def load_pretrained_weight(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=True)

    def load_pretrained_with_channel_adapt(self, checkpoint_path):
        """Load pretrained weights with smart channel adaptation.

        Handles shape mismatches in channel-dependent layers when the model's
        in_channels / out_channels differ from the pretrained checkpoint (typically 3).
        All other parameters are loaded normally via strict=False.
        """
        pretrained_sd = torch.load(checkpoint_path, map_location="cpu")
        model_sd = self.state_dict()

        channel_keys = self._find_channel_mismatch_keys(pretrained_sd, model_sd)
        if channel_keys:
            logger.info(
                f"Channel-adaptive loading: {len(channel_keys)} keys with shape mismatch "
                f"will be handled specially: {list(channel_keys.keys())}"
            )

        compatible_sd = {
            k: v for k, v in pretrained_sd.items()
            if k in model_sd and k not in channel_keys and model_sd[k].shape == v.shape
        }
        msg = self.load_state_dict(compatible_sd, strict=False)
        logger.info(
            f"Loaded {len(compatible_sd)} compatible keys. "
            f"Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}"
        )

        for key, (pre_shape, model_shape) in channel_keys.items():
            pre_w = pretrained_sd[key]
            new_w = model_sd[key].clone()
            self._adapt_channel_weight(key, pre_w, new_w, pre_shape, model_shape)
            with torch.no_grad():
                param = self
                for attr in key.split("."):
                    param = getattr(param, attr) if not attr.isdigit() else param[int(attr)]
                param.copy_(new_w)

        logger.info("Channel-adaptive pretrained weight loading complete.")

    @staticmethod
    def _find_channel_mismatch_keys(pretrained_sd, model_sd):
        mismatched = {}
        for k in pretrained_sd:
            if k in model_sd and pretrained_sd[k].shape != model_sd[k].shape:
                mismatched[k] = (pretrained_sd[k].shape, model_sd[k].shape)
        return mismatched

    @staticmethod
    def _adapt_channel_weight(key, pre_w, new_w, pre_shape, model_shape):
        """Copy pretrained weights into the new tensor with channel adaptation.

        Strategy: when the channel dimension mismatches, compute the mean
        across the pretrained channel axis and broadcast (repeat) it to fill
        every new channel.  This gives a uniform, pretrained-informed starting
        point for all channels instead of leaving some randomly initialized.
        """

        if "patch_embed.weight" in key:
            # Encoder input projection: (width, C_in_pre, P, P) -> (width, C_in_new, P, P)
            c_new = model_shape[1]
            mean_w = pre_w.mean(dim=1, keepdim=True)          # (width, 1, P, P)
            new_w.copy_(mean_w.expand_as(new_w))
            logger.info(f"  {key}: mean-broadcast input channels {pre_shape[1]} -> {c_new}")

        elif "ffn.0.weight" in key:
            # Decoder pixel projection: (P^2*C_pre, width, 1, 1) -> (P^2*C_new, width, 1, 1)
            width = pre_shape[1]
            c_new_total = model_shape[0]
            for c_old_guess in [3, 1]:
                if pre_shape[0] % c_old_guess == 0:
                    p_sq = pre_shape[0] // c_old_guess
                    c_new = c_new_total // p_sq
                    if c_new * p_sq == c_new_total:
                        # (P^2, C_old, width, 1, 1) -> mean over C_old -> (P^2, 1, width, 1, 1)
                        pre_reshaped = pre_w.reshape(p_sq, c_old_guess, width, 1, 1)
                        mean_ch = pre_reshaped.mean(dim=1, keepdim=True)  # (P^2, 1, w, 1, 1)
                        expanded = mean_ch.expand(p_sq, c_new, width, 1, 1)
                        new_w.copy_(expanded.reshape(model_shape))
                        logger.info(
                            f"  {key}: mean-broadcast output channels {c_old_guess} -> {c_new} "
                            f"(P^2={p_sq})"
                        )
                        return
            logger.warning(f"  {key}: could not determine channel layout, using random init")
            nn.init.trunc_normal_(new_w, mean=0.0, std=0.02)

        elif "ffn.0.bias" in key:
            c_pre_total = pre_shape[0]
            c_new_total = model_shape[0]
            for c_old_guess in [3, 1]:
                if c_pre_total % c_old_guess == 0:
                    p_sq = c_pre_total // c_old_guess
                    c_new = c_new_total // p_sq
                    if c_new * p_sq == c_new_total:
                        pre_reshaped = pre_w.reshape(p_sq, c_old_guess)
                        mean_ch = pre_reshaped.mean(dim=1, keepdim=True)  # (P^2, 1)
                        expanded = mean_ch.expand(p_sq, c_new)
                        new_w.copy_(expanded.reshape(model_shape))
                        logger.info(f"  {key}: mean-broadcast bias channels {c_old_guess} -> {c_new}")
                        return
            logger.warning(f"  {key}: could not determine channel layout for bias, using random init")
            new_w.zero_()

        elif "conv_out.weight" in key:
            # Decoder final refine conv: (C_pre, C_pre, k, k) -> (C_new, C_new, k, k)
            c_pre = pre_shape[0]
            c_new = model_shape[0]
            # mean over both input and output channel dims
            mean_w = pre_w.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)  # (1, 1, k, k)
            new_w.copy_(mean_w.expand_as(new_w))
            logger.info(f"  {key}: mean-broadcast conv_out channels {c_pre} -> {c_new}")

        elif "conv_out.bias" in key or "patch_embed.bias" in key:
            c_pre = pre_shape[0]
            c_new = model_shape[0]
            mean_b = pre_w.mean()
            new_w.fill_(mean_b.item())
            logger.info(f"  {key}: mean-fill bias {c_pre} -> {c_new}")

        else:
            logger.warning(
                f"  {key}: unrecognized mismatch {pre_shape} -> {model_shape}, using random init"
            )
            nn.init.trunc_normal_(new_w, mean=0.0, std=0.02)
