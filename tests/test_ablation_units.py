"""CPU unit tests for the 4-model i2i ablation code paths.

No pytest dependency: run `python tests/test_ablation_units.py` — it collects
every top-level `test_*` function, runs it, prints PASS/FAIL, exits nonzero on
any failure. Tiny tensors, CPU only.
"""
import os
import sys
import traceback
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cfg(**kw):
    tv = SimpleNamespace(num_blocks=2, num_attention_heads=2, hidden_dim=64,
                         dropout_prob=0.0, clip_loss_weight=0.0)
    base = dict(channels=16, clip_dim=16, num_clip_token=77, cfg_indicator=0.0,
                use_t2t_temperature=False, textVAE=tv)
    base.update(kw)
    return SimpleNamespace(**base)


# ── Task 1: DiT in/out channel split ────────────────────────────────────────
def test_channel_split_legacy_default():
    from libs.model.flowtok_t2i import FlowTok_B
    m = FlowTok_B(_cfg(), num_latent_tokens=77)
    assert m.x_embedder.in_features == 16, m.x_embedder.in_features
    assert m.out_channels == 16
    y = m(torch.randn(2, 77, 16), t=torch.rand(2),
          null_indicator=torch.zeros(2, dtype=torch.bool))[0]
    assert tuple(y.shape) == (2, 77, 16), y.shape


def test_channel_split_concat():
    from libs.model.flowtok_t2i import FlowTok_B
    m = FlowTok_B(_cfg(cond_concat_channels=True), num_latent_tokens=77)
    assert m.x_embedder.in_features == 32, m.x_embedder.in_features
    assert m.out_channels == 16
    y = m(torch.randn(2, 77, 32), t=torch.rand(2),
          null_indicator=torch.zeros(2, dtype=torch.bool))[0]
    assert tuple(y.shape) == (2, 77, 16), y.shape


# ── Task 2: FlowMatching flow_prediction_target ─────────────────────────────
def test_flow_pred_target_switch():
    from diffusion.flow_matching import FlowMatching
    fm_v = FlowMatching(noising_type="none")
    fm_x = FlowMatching(noising_type="none", flow_prediction_target="radar_tokens")
    assert fm_v.flow_prediction_target == "velocity"
    assert fm_x.flow_prediction_target == "radar_tokens"
    x0 = torch.randn(3, 77, 16)
    x1 = torch.randn(3, 77, 16)
    t = torch.rand(3)
    v = fm_v.Dt_psi(t, x=x0, x1=x1)
    smin, smax = fm_v.sigma_min, fm_v.sigma_max
    v_from_x1 = (smin / smax - 1.0) * x0 + x1
    assert torch.allclose(v, v_from_x1, atol=1e-5), (v - v_from_x1).abs().max()


# ── Task 3: ODE solver x1→v reparam ─────────────────────────────────────────
class _DummyVel:
    def __init__(self, x0, x1):
        self.x0, self.x1 = x0, x1

    @property
    def device(self):
        return self.x0.device

    def __call__(self, x, t=None, null_indicator=None):
        return [None, (1e-5 / 1.0 - 1.0) * self.x0 + self.x1]


class _DummyX1:
    def __init__(self, x1):
        self.x1 = x1

    @property
    def device(self):
        return self.x1.device

    def __call__(self, x, t=None, null_indicator=None):
        return [None, self.x1]


def test_solver_x1_reparam_matches_velocity():
    from diffusion.flow_matching import ODEEulerFlowMatchingSolver
    torch.manual_seed(0)
    x0 = torch.randn(2, 77, 16)
    x1 = torch.randn(2, 77, 16)
    sv = ODEEulerFlowMatchingSolver(_DummyVel(x0.clone(), x1.clone()),
                                    step_size_type="step_in_dsigma", guidance_scale=1.0)
    zv, _ = sv.sample(x_T=x0.clone(), batch_size=2, sample_steps=20,
                      unconditional_guidance_scale=1.0, has_null_indicator=False)
    sx = ODEEulerFlowMatchingSolver(_DummyX1(x1.clone()),
                                    step_size_type="step_in_dsigma", guidance_scale=1.0)
    zx, _ = sx.sample(x_T=x0.clone(), batch_size=2, sample_steps=20,
                      unconditional_guidance_scale=1.0, has_null_indicator=False,
                      prediction_target="radar_tokens")
    assert torch.allclose(zv, zx, atol=1e-4), (zv - zx).abs().max()


# ── Task 4: TokenDiffusion ──────────────────────────────────────────────────
class _Oracle:
    def __init__(self, z1):
        self.z1 = z1

    def __call__(self, inp, t=None, null_indicator=None):
        return [self.z1]


def test_token_diffusion_schedule_and_ddim_oracle():
    from diffusion.token_diffusion import TokenDiffusion
    td = TokenDiffusion(train_timesteps=1000, schedule="linear",
                        target="pred_x0", gamma="ddim")
    assert tuple(td.alpha_t.shape) == (1001,)
    assert tuple(td.sigma_t.shape) == (1001,)
    # Diffi2i linear schedule starts beta=1e-4 → sigma[0]≈0.01 (terminal noise
    # floor); alpha[0]≈0.99995. Both monotone toward heavy noise at t=T.
    assert td.alpha_t[0] > 0.99 and td.sigma_t[0] < 0.02
    assert td.alpha_t[-1] < td.alpha_t[0]
    assert td.sigma_t[-1] > td.sigma_t[0]
    z1 = torch.randn(2, 77, 16)
    rec = td.ddim_sample(_Oracle(z1), cond=torch.zeros(2, 77, 16), sample_steps=50)
    # DDIM + perfect pred_x0 denoises to z1 up to the schedule's sigma[0]≈0.01
    # floor (maxerr ~0.04 ≪ noise scale 1.0) — confirms the reverse update.
    assert (rec - z1).abs().max() < 0.1, (rec - z1).abs().max()


def test_token_diffusion_loss_runs():
    from diffusion.token_diffusion import TokenDiffusion
    td = TokenDiffusion()

    class _Net:
        def __call__(self, inp, t=None, null_indicator=None):
            assert inp.shape[-1] == 32
            return [inp[..., :16]]

    loss, d = td.loss(_Net(), torch.randn(2, 77, 16), torch.randn(2, 77, 16))
    assert torch.isfinite(loss) and "diff_loss" in d


def _main():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    fails = 0
    for n, f in fns:
        try:
            f()
            print(f"PASS {n}")
        except Exception:
            fails += 1
            print(f"FAIL {n}")
            traceback.print_exc()
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    _main()
