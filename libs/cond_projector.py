"""Lightweight projector to merge concatenated condition tokens back to target length.

Used when sat_ir tokens and lightning tokens are concatenated (2*L per frame)
and need to be projected to L tokens per frame to match radar token length.

Supports optional reparameterization (VAE-style) so the output can be
regularised toward N(0,1) via a KLD loss, similar to the textVAE in FlowTok.
"""

import torch
import torch.nn as nn


class CondTokenProjector(nn.Module):
    """Project concatenated condition tokens [B, T*L, 2C] -> [B, T*L, C].

    When ``reparameterize=True`` the projection outputs *2C* channels
    (mu and log_var), samples via the reparameterization trick, and
    exposes ``last_mu`` / ``last_logvar`` for external KLD computation.
    """

    def __init__(self, token_dim: int, num_latent_tokens: int,
                 hidden_mult: int = 2, reparameterize: bool = False):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.reparameterize = reparameterize

        out_dim = token_dim * 2 if reparameterize else token_dim
        self.proj = nn.Sequential(
            nn.Linear(2 * token_dim, hidden_mult * token_dim),
            nn.GELU(),
            nn.Linear(hidden_mult * token_dim, out_dim),
        )
        self._init_weights()

        # Cached for external loss computation (set during forward).
        self.last_mu = None
        self.last_logvar = None
        self._cached_input_mean = None

    def _init_weights(self):
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, sat_tokens: torch.Tensor, lgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sat_tokens: [B, T*L, C]
            lgt_tokens: [B, T*L, C]
        Returns:
            merged: [B, T*L, C]
        """
        x = torch.cat([sat_tokens, lgt_tokens], dim=-1)  # [B, T*L, 2C]
        out = self.proj(x)

        if self.reparameterize:
            mu, logvar = torch.chunk(out, 2, dim=-1)  # each [B, T*L, C]
            self.last_mu = mu
            self.last_logvar = logvar
            return self._reparameterize(mu, logvar)  # [B, T*L, C]
        else:
            self.last_mu = None
            self.last_logvar = None
            return out  # [B, T*L, C]
