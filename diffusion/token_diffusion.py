"""Diffi2i-style DDPM/DDIM operating on FlowTiTok latent tokens [B, L, C].

Mirrors Diffi2i-shrimp-proj2/src/diffusion.py (linear/cosine schedule,
q_sample, pred_x0/pred_eps target, DDIM/DDPM reverse). Conditioning is a
channel-concat of sat tokens with the noisy radar tokens (faithful to Diffi2i's
`torch.cat((x_t, cond), dim=1)`); the DiT must therefore be built with
cond_concat_channels=True so in_channels == 2*C and out_channels == C.

Opt-in only: selected via config.generation_algorithm == "diffusion". The flow
path is untouched, so existing runs/ckpts are unaffected.
"""
import math

import torch
import torch.nn as nn


class TokenDiffusion(nn.Module):
    def __init__(self, train_timesteps: int = 1000, schedule: str = "linear",
                 target: str = "pred_x0", gamma: str = "ddim"):
        super().__init__()
        assert schedule in ("linear", "cosine")
        assert target in ("pred_x0", "pred_eps")
        assert gamma in ("ddim", "ddpm")
        self.T = int(train_timesteps)
        self.schedule, self.target, self.gamma = schedule, target, gamma

        if schedule == "linear":
            beta = torch.linspace(1e-4, 2e-2, self.T + 1)
            alpha = torch.cumprod(1.0 - beta, dim=0) ** 0.5
        else:  # Diffi2i shifted-cosine
            ls = torch.linspace(0, 1, self.T + 1)
            f = torch.cos((ls + 0.008) / 1.008 * math.pi / 2) ** 2
            bar = f / f[0]
            beta = torch.zeros_like(bar)
            beta[1:] = (1 - bar[1:] / bar[:-1]).clamp(0, 0.999)
            alpha = torch.cumprod(1.0 - beta, dim=0) ** 0.5
        sigma = (1.0 - alpha ** 2).clamp(min=0) ** 0.5
        self.register_buffer("alpha_t", alpha)   # [T+1]
        self.register_buffer("sigma_t", sigma)   # [T+1]

    def _a(self, idx):
        # Device-agnostic: schedule buffers may sit on CPU while idx is on CUDA.
        return self.alpha_t.to(idx.device)[idx].view(-1, 1, 1)

    def _s(self, idx):
        return self.sigma_t.to(idx.device)[idx].view(-1, 1, 1)

    def q_sample(self, z1, t_idx, eps):
        return self._a(t_idx) * z1 + self._s(t_idx) * eps

    def loss(self, nnet, z1, cond):
        """z1: clean radar tokens [B,L,C]; cond: sat tokens [B,L,C]."""
        b = z1.shape[0]
        dev = z1.device
        t_idx = torch.randint(1, self.T + 1, (b,), device=dev)
        eps = torch.randn_like(z1)
        z_t = self.q_sample(z1, t_idx, eps)
        inp = torch.cat([z_t, cond], dim=-1)                # [B,L,2C]
        t_cont = t_idx.float() / self.T                     # fractional t in (0,1]
        null_ind = torch.zeros(b, dtype=torch.bool, device=dev)
        pred = nnet(inp, t=t_cont, null_indicator=null_ind)[0]
        target = z1 if self.target == "pred_x0" else eps
        ld = (0.5 * (pred - target).pow(2).flatten(1).mean(dim=-1)).mean()
        zero = z1.new_zeros([])
        return ld, {"diff_loss": ld, "contrastive_loss": zero, "kld_loss": zero}

    @torch.no_grad()
    def ddim_sample(self, nnet, cond, sample_steps: int = 500):
        b, L, C = cond.shape
        dev = cond.device
        z = torch.randn(b, L, C, device=dev)
        null_ind = torch.zeros(b, dtype=torch.bool, device=dev)
        subseq = torch.linspace(self.T, 0, sample_steps + 1,
                                device=dev).round().long()
        for ts, te in zip(subseq[:-1], subseq[1:]):
            ts_b = ts.repeat(b)
            out = nnet(torch.cat([z, cond], dim=-1),
                       t=ts_b.float() / self.T, null_indicator=null_ind)[0]
            if self.target == "pred_x0":
                pred_x0 = out
                pred_eps = (z - self._a(ts_b) * pred_x0) / self._s(ts_b).clamp(min=1e-6)
            else:
                pred_eps = out
                pred_x0 = (z - self._s(ts_b) * pred_eps) / self._a(ts_b).clamp(min=1e-6)
            te_b = te.repeat(b)
            if self.gamma == "ddim":
                z = self._a(te_b) * pred_x0 + self._s(te_b) * pred_eps
            else:  # ddpm
                gt = (self._s(te_b) / self._s(ts_b).clamp(min=1e-6)) * (
                    1 - self._a(ts_b) ** 2 / self._a(te_b).clamp(min=1e-6) ** 2
                ).clamp(min=0) ** 0.5
                noise = torch.zeros_like(z) if int(te) == 0 else torch.randn_like(z)
                z = (self._a(te_b) * pred_x0
                     + (self._s(te_b) ** 2 - gt ** 2).clamp(min=0) ** 0.5 * pred_eps
                     + gt * noise)
        return z
