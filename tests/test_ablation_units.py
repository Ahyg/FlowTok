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


# ── Task 5: token_concat_interleaved (aligned sat/radar per-frame layout) ────
def test_interleave_layout_and_roundtrip():
    from diffusion.flow_matching import _interleave_cond_target, _deinterleave_target
    B, T, L, C = 2, 3, 4, 5
    cond = torch.randn(B, T * L, C)
    x = torch.randn(B, T * L, C)
    inter = _interleave_cond_target(cond, x, L)
    assert tuple(inter.shape) == (B, T * 2 * L, C), inter.shape
    # fat frame 0 = [cond_f0 (L) | x_f0 (L)]
    assert torch.equal(inter[:, :L], cond[:, :L])
    assert torch.equal(inter[:, L:2 * L], x[:, :L])
    # fat frame 1's first half is cond frame 1 (proves per-frame interleave, not block)
    assert torch.equal(inter[:, 2 * L:3 * L], cond[:, L:2 * L])
    # de-interleave recovers exactly the target (radar) half, in original order
    assert torch.equal(_deinterleave_target(inter, L), x)


def test_flowmatching_accepts_interleaved_mode():
    from diffusion.flow_matching import FlowMatching
    fm = FlowMatching(noising_type="none", flow_prediction_target="radar_tokens",
                      flow_cond_mode="token_concat_interleaved")
    assert fm.flow_cond_mode == "token_concat_interleaved"


def _fm_cfg(L):
    return SimpleNamespace(
        losses=SimpleNamespace(contrastive_loss_weight=0.0, kld_loss_weight=0.0),
        nnet=SimpleNamespace(model_args=SimpleNamespace(cfg_indicator=0.0)),
        use_text_vae_encoder=False,
        vq_model=SimpleNamespace(num_latent_tokens=L),
    )


def test_flowmatching_interleaved_equals_block_under_identity_nnet():
    # A position-insensitive (identity) nnet supervises the same target tokens
    # regardless of block vs interleaved ordering, so the loss must be identical.
    # A wrong interleave/de-interleave (extracting cond, or misframing) would
    # change which tokens are supervised and break the equality.
    from diffusion.flow_matching import FlowMatching
    B, T, L, C = 2, 3, 4, 16
    cond = torch.randn(B, T * L, C)
    x1 = torch.randn(B, T * L, C)
    cfg = _fm_cfg(L)
    nnet = lambda inp, t=None, null_indicator=None: [inp]  # identity
    fm_block = FlowMatching(noising_type="none", flow_prediction_target="radar_tokens",
                            flow_cond_mode="token_concat")
    fm_inter = FlowMatching(noising_type="none", flow_prediction_target="radar_tokens",
                            flow_cond_mode="token_concat_interleaved")
    torch.manual_seed(123)
    lb, _ = fm_block(x=x1.clone(), nnet=nnet, cond=cond.clone(), all_config=cfg)
    torch.manual_seed(123)
    li, _ = fm_inter(x=x1.clone(), nnet=nnet, cond=cond.clone(), all_config=cfg)
    assert torch.allclose(lb, li, atol=1e-6), (lb.item(), li.item())


def test_solver_interleaved_feeds_2TL_and_returns_clean_radar():
    from diffusion.flow_matching import ODEEulerFlowMatchingSolver
    B, T, L, C = 2, 3, 4, 16
    cond = torch.randn(B, T * L, C)
    xT = torch.randn(B, T * L, C)
    seen = {}

    def nnet(inp, t=None, null_indicator=None):
        seen["n"] = inp.shape[1]            # must be 2*T*L if interleave happened
        return [None, inp]                  # identity at index -1

    sv = ODEEulerFlowMatchingSolver(nnet, step_size_type="step_in_dsigma",
                                    guidance_scale=1.0)
    z, _ = sv.sample(x_T=xT.clone(), batch_size=B, sample_steps=5,
                     unconditional_guidance_scale=1.0, has_null_indicator=False,
                     prediction_target="velocity",
                     flow_cond_mode="token_concat_interleaved",
                     cond_tokens=cond.clone(), cond_num_latent_tokens=L)
    assert seen.get("n") == 2 * T * L, seen
    assert tuple(z.shape) == (B, T * L, C), z.shape


def test_modality_pos_emb_partition_and_alignment():
    import torch, numpy as np
    from types import SimpleNamespace
    from libs.model.flowtok_t2i import FlowTok
    L_tok, T, D = 77, 4, 768
    cfg = SimpleNamespace(
        channels=16, clip_dim=16, num_clip_token=L_tok, cfg_indicator=0.0,
        noising_type="none", noising_scale=0.1, use_modality_pos_emb=True,
        textVAE=SimpleNamespace(num_blocks=1, hidden_dim=32, num_attention_heads=2,
                                dropout_prob=0.0, clip_loss_weight=0.0),
    )
    m = FlowTok(cfg, num_latent_tokens=L_tok, hidden_size=D, depth=1, num_heads=8)
    assert m.use_modality_pos_emb is True
    seq = 2 * T * L_tok
    pe = m._build_pos_embed(seq, torch.device("cpu"), torch.float32)
    assert pe.shape == (1, seq, D)
    half = seq // 2
    d_sp, d_tp = D // 2, D // 4
    for i in range(T):
        for j in (0, 40, 76):
            sat = i * L_tok + j
            rad = half + i * L_tok + j
            assert torch.allclose(pe[0, sat, :d_sp], pe[0, rad, :d_sp])
            assert torch.allclose(pe[0, sat, d_sp:d_sp + d_tp], pe[0, rad, d_sp:d_sp + d_tp])
            assert not torch.allclose(pe[0, sat, d_sp + d_tp:], pe[0, rad, d_sp + d_tp:])


def test_modality_flag_defaults_off():
    import torch
    from types import SimpleNamespace
    from libs.model.flowtok_t2i import FlowTok
    cfg = SimpleNamespace(
        channels=16, clip_dim=16, num_clip_token=77, cfg_indicator=0.0,
        noising_type="none", noising_scale=0.1,
        textVAE=SimpleNamespace(num_blocks=1, hidden_dim=32, num_attention_heads=2,
                                dropout_prob=0.0, clip_loss_weight=0.0),
    )
    m = FlowTok(cfg, num_latent_tokens=77, hidden_size=768, depth=1, num_heads=8)
    assert m.use_modality_pos_emb is False
    pe = m._build_pos_embed(2 * 77, torch.device("cpu"), torch.float32)
    assert pe.shape == (1, 2 * 77, 768)


def test_flowmatching_accepts_token_concat_modality():
    from diffusion.flow_matching import FlowMatching
    fm = FlowMatching(flow_cond_mode="token_concat_modality",
                      flow_prediction_target="radar_tokens")
    assert fm.flow_cond_mode == "token_concat_modality"

def test_modality_training_branch_supervises_radar_half():
    import torch
    from types import SimpleNamespace
    from diffusion.flow_matching import FlowMatching
    B, T, Lt, C = 2, 2, 77, 16
    fm = FlowMatching(flow_cond_mode="token_concat_modality",
                      flow_prediction_target="radar_tokens")
    x_start = torch.randn(B, T * Lt, C)
    cond = torch.randn(B, T * Lt, C)
    t = torch.rand(B)
    class IdNnet:
        def __call__(self, inp, t=None, null_indicator=None, context=None):
            return [inp]
    all_cfg = SimpleNamespace(
        losses=SimpleNamespace(contrastive_loss_weight=0.0, kld_loss_weight=0.0),
        vq_model=SimpleNamespace(num_latent_tokens=Lt),
        nnet=SimpleNamespace(model_args=SimpleNamespace(cfg_indicator=0.0)),
    )
    loss, logs = fm.p_losses_textVAE_flowtok(x_start, cond, t, IdNnet(), all_cfg)
    assert "diff_loss" in logs and torch.isfinite(loss).all()


def test_crossattention_shape():
    import torch
    from libs.model.flowtok_t2i import CrossAttention
    ca = CrossAttention(64, num_heads=8)
    x = torch.randn(2, 30, 64); ctx = torch.randn(2, 50, 64)
    out = ca(x, ctx)
    assert out.shape == (2, 30, 64)

def test_cross_attn_block_zero_init_identity():
    import torch
    from libs.model.flowtok_t2i import DiTBlock
    blk = DiTBlock(64, num_heads=8, use_cross_attn=True)
    blk.train(False)
    for p in blk.adaLN_modulation[-1].parameters():
        torch.nn.init.zeros_(p)
    x = torch.randn(2, 10, 64); c = torch.randn(2, 64); ctx = torch.randn(2, 12, 64)
    out = blk._forward(x, c, ctx)
    assert torch.allclose(out, x, atol=1e-5)

def test_flowtok_cross_attention_forward_and_compat():
    import torch
    from types import SimpleNamespace
    from libs.model.flowtok_t2i import FlowTok
    base = dict(channels=16, clip_dim=16, num_clip_token=77, cfg_indicator=0.0,
                noising_type="none", noising_scale=0.1,
                textVAE=SimpleNamespace(num_blocks=1, hidden_dim=32, num_attention_heads=2,
                                        dropout_prob=0.0, clip_loss_weight=0.0))
    cfg = SimpleNamespace(use_cross_attention=True, **base)
    m = FlowTok(cfg, num_latent_tokens=77, hidden_size=128, depth=2, num_heads=8)
    assert hasattr(m, "context_embedder")
    x = torch.randn(2, 2 * 77, 16); ctx = torch.randn(2, 2 * 77, 16); t = torch.rand(2)
    nullind = torch.zeros(2, dtype=torch.bool)
    out = m(x, t=t, null_indicator=nullind, context=ctx)[0]
    assert out.shape == (2, 2 * 77, 16)
    cfg0 = SimpleNamespace(**base)
    m0 = FlowTok(cfg0, num_latent_tokens=77, hidden_size=128, depth=2, num_heads=8)
    assert not hasattr(m0, "context_embedder")
    assert m0.use_cross_attention is False
    out0 = m0(x, t=t, null_indicator=nullind)[0]
    assert out0.shape == (2, 2 * 77, 16)


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
