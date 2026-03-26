"""Lightweight projector to merge concatenated condition tokens back to target length.

Used when sat_ir tokens and lightning tokens are concatenated (2*L per frame)
and need to be projected to L tokens per frame to match radar token length.
"""

import torch
import torch.nn as nn


class CondTokenProjector(nn.Module):
    """Project concatenated condition tokens [B, T*2L, C] -> [B, T*L, C].

    Strategy: reshape to per-frame pairs [B*T, 2, L, C], apply a small
    cross-token MLP that mixes the pair dimension, then output [B*T, L, C].
    """

    def __init__(self, token_dim: int, num_latent_tokens: int, hidden_mult: int = 2):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        # Input: 2 * token_dim (sat + lgt concatenated along feature dim after reshape)
        # Output: token_dim
        self.proj = nn.Sequential(
            nn.Linear(2 * token_dim, hidden_mult * token_dim),
            nn.GELU(),
            nn.Linear(hidden_mult * token_dim, token_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sat_tokens: torch.Tensor, lgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sat_tokens: [B, T*L, C]
            lgt_tokens: [B, T*L, C]
        Returns:
            merged: [B, T*L, C]
        """
        # Concatenate along feature dimension then project
        # This keeps the token count the same as radar (T*L)
        x = torch.cat([sat_tokens, lgt_tokens], dim=-1)  # [B, T*L, 2C]
        return self.proj(x)  # [B, T*L, C]
