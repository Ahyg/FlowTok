"""Condition token projector for concat fusion mode.

Merges sat_ir_tokens and lgt_tokens via channel concatenation [B, T*L, 2C]
and projects to [B, T*L, C].  The implementation mirrors the textVAE design
in FlowTok (context_encoder + context_projector + contrastive loss).

Components (all matching textVAE exactly):
  encoder   – FlowEncoder  (same as context_encoder)
  projector – 3-layer MLP   (same as context_projector)
  t2t_temperature – learnable logit scale for contrastive loss
"""

import torch
import torch.nn as nn

from libs.model.trans_autoencoder import FlowEncoder


class CondTokenProjector(nn.Module):
    """Project concatenated condition tokens [B, T*L, 2C] -> [B, T*L, C].

    Mirrors textVAE: FlowEncoder encodes to (mu, logvar), reparameterize
    to get z.  A separate projector MLP produces an anchor for the CLIP-style
    contrastive loss.
    """

    def __init__(self, token_dim: int, num_latent_tokens: int,
                 num_blocks: int = 6, num_attention_heads: int = 4,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.token_dim = token_dim
        self.num_latent_tokens = num_latent_tokens

        # ── Encoder (= textVAE context_encoder) ──────────────────────
        # Input:  [B, T*L, 2*token_dim]
        # Output: [B, T*L, 2*token_dim]  (mu + logvar, each token_dim)
        self.encoder = FlowEncoder(
            d_model=2 * token_dim,
            N=num_blocks,
            head_num=num_attention_heads,
            d_ff=d_ff,
            latten_size=2 * token_dim,
            dropout=dropout,
            last_norm=False,
        )

        # ── Projector (= textVAE context_projector) ─────────────────
        # Input:  [B, T*L, 2*token_dim]
        # Output: [B, T*L, token_dim]
        self.projector = nn.Sequential(
            nn.Linear(2 * token_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, token_dim),
        )

        # ── Learnable temperature (= textVAE t2t_temperature) ───────
        self.t2t_temperature = nn.Parameter(
            torch.log(torch.tensor(1.0 / 0.07))
        )

        # Cached for external loss computation (set during forward).
        self.last_mu = None
        self.last_logvar = None
        self.last_proj = None  # projector output for contrastive loss

    @staticmethod
    def _reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, sat_tokens: torch.Tensor, lgt_tokens: torch.Tensor,
                deterministic: bool = False) -> torch.Tensor:
        """
        Args:
            sat_tokens: [B, T*L, C]
            lgt_tokens: [B, T*L, C]
            deterministic: if True, return mu directly (no sampling noise).
                           Use this during inference/sampling.
        Returns:
            z: [B, T*L, C]
        """
        x = torch.cat([sat_tokens, lgt_tokens], dim=-1)  # [B, T*L, 2C]

        # Encoder path (= textVAE _text_encoder)
        out = self.encoder(x)  # [B, T*L, 2C]
        mu, logvar = torch.chunk(out, 2, dim=-1)  # each [B, T*L, C]
        self.last_mu = mu
        self.last_logvar = logvar

        # Projector path (= textVAE _text_projector) — cached for loss
        self.last_proj = self.projector(x)  # [B, T*L, C]

        if deterministic:
            return mu
        return self._reparameterize(mu, logvar)  # [B, T*L, C]
