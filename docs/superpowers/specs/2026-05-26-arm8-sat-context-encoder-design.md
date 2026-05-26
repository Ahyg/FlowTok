# Arm 8 — Satellite Context Self-Attention Encoder (design)

**Date:** 2026-05-26
**Status:** design approved, pending spec review → implementation plan
**Branch:** `v2v-7arm-comparison` (both repos must stay on this branch — cluster reads the live tree)

## 1. Motivation

Arm 7 (cross-attention, `flow_cond_mode="cross_attention"`) conditions the radar
DiT on satellite tokens used as cross-attention K/V. But those sat tokens reach
the K/V **un-refined**: in `_forward` they only pass through a single linear
projection (`context_embedder`, 16→768) plus an additive positional embedding,
then are fed *unchanged* as K/V to every block. There is **no self-attention
among the sat tokens** inside the learnable conditioning path — in particular no
cross-frame mixing, so each block attends to a static, shallow sat
representation.

This is exactly the step that every strong conditioning system keeps:
Stable Diffusion / PixArt run text through a deep T5 self-attention encoder
before cross-attention; IP-Adapter runs the image through a CLIP self-attention
ViT. Arm 7 is the degenerate "no context encoder" case.

**Arm 8 adds the missing context encoder:** a deterministic, timestep-independent
self-attention stack that refines the sat tokens before they become K/V. It
isolates a single variable vs arm 7 — "does refining the sat K/V with
self-attention help?" — holding the cross-attention injection, dataset, steps,
and tokenizer fixed.

Note: arm 8 uses **full self-attention over all sat tokens** (spatiotemporal,
not axial), so it carries **no** TiTok same-index-across-frames correspondence
assumption. (That assumption belongs to the deferred **arm 9** factorized /
axial-temporal design.)

## 2. Scope

In scope:
- A new `SatContextEncoder` module in `libs/model/flowtok_t2i.py`.
- A new config flag `sat_context_encoder_layers` (default 0 = arm 7 behavior).
- One insertion point in `FlowTok._forward` (ctx branch only).
- New config / PBS train / PBS test / tiny-gate scripts cloned from arm 7 (xattn-B).
- Unit tests for backward-compat and shape/behavior.

Out of scope (explicitly):
- Arm 9 (factorized frame-local + axial temporal attention) — deferred.
- Touching the main token stream (`x_embedder` path) — unchanged.
- Touching legacy `context_encoder` (FlowEncoder/textVAE) or `context_projector`
  — they belong to the dormant T2I text path, are never called in v2v
  sat-conditioning, and must remain bit-identical to preserve state_dict
  compatibility for arms 1–7 and the running XL big-runs.
- Classifier-free guidance changes (null_indicator stays all-False, as arm 7).

## 3. Architecture

### 3.1 New module (pre-norm ViT encoder, reusing timm `Attention`/`Mlp`)

```python
# libs/model/flowtok_t2i.py  (Attention, Mlp already imported from timm L6)

class SatEncoderLayer(nn.Module):
    """One pre-norm self-attention + MLP block. No adaLN, no timestep cond,
    no cross-attn — the sat context is clean input, independent of the
    diffusion timestep."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn  = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp   = Mlp(in_features=hidden_size,
                         hidden_features=int(hidden_size * mlp_ratio),
                         act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SatContextEncoder(nn.Module):
    """Timestep-independent self-attention encoder refining satellite K/V
    tokens before per-block cross-attention (Arm 8). Full self-attention over
    the whole sat sequence (T*L tokens) => spatiotemporal, not axial."""
    def __init__(self, hidden_size, num_heads, depth=6, mlp_ratio=4.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [SatEncoderLayer(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)  # final norm stabilizes K/V
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
```

Design choices:
- **Pre-norm, self-written** (not reusing `FlowEncoder`): `FlowEncoder` is
  post-norm, carries a VAE head, and was built at width 16. A small pre-norm
  stack at width 768 matches `DiTBlock`'s style and is more controllable.
  (Reusing `FlowEncoder` at d=768 with the VAE head stripped is a viable
  alternative but rejected for style/clarity.)
- **No adaLN / no timestep**: the satellite input is the clean condition,
  independent of the denoising step, mirroring T5/CLIP encoders.
- **Runs once**, before the DiT block loop; output reused by all blocks'
  cross-attention.

### 3.2 Construction (in `FlowTok.__init__`, inside the `use_cross_attention` block)

```python
if self.use_cross_attention:
    self.context_embedder = nn.Linear(config.channels, hidden_size, bias=True)  # arm 7
    self.sat_ctx_layers = getattr(config, "sat_context_encoder_layers", 0)      # NEW
    self.use_sat_context_encoder = self.sat_ctx_layers > 0
    if self.use_sat_context_encoder:
        self.sat_context_encoder = SatContextEncoder(
            hidden_size, num_heads, depth=self.sat_ctx_layers, mlp_ratio=mlp_ratio)
```

`num_heads`, `hidden_size`, `mlp_ratio` are already in `__init__` scope (used at
the `DiTBlock` construction). When `use_cross_attention` is False the attribute
`use_sat_context_encoder` is never set; guard reads with
`getattr(self, "use_sat_context_encoder", False)`.

### 3.3 Data flow (`FlowTok._forward`, ctx branch only)

```python
ctx = None
if self.use_cross_attention and context is not None:
    ctx = self.context_embedder(context)                       # 16 -> 768   (arm 7)
    ctx = ctx + self._build_pos_embed(seq_len=ctx.shape[1],
                                      device=ctx.device, dtype=ctx.dtype)  # 768 pos-emb (arm 7)
    if getattr(self, "use_sat_context_encoder", False):        # NEW (arm 8)
        ctx = self.sat_context_encoder(ctx)                    # N=6 self-attn @ 768
# blocks consume ctx unchanged (arm 7 cross-attn), main stream x untouched
```

The encoder receives the sat tokens **already at width 768 with the existing
768-d positional embedding added** (spatial sincos 384 ⊕ temporal sincos 384),
so it is position-aware with no new pos-emb code.

## 4. Locked parameters

| Parameter | Value | Rationale |
|---|---|---|
| Encoder depth `N` | **6** | user decision |
| Encoder width | **768** (= hidden_size) | attention works at model width; reuses existing 768 pos-emb; full-capacity heads |
| Variational head | **none** | deterministic refinement; not encoding into a VAE latent |
| Timestep conditioning | **none** | sat condition is timestep-independent |
| Attention scope | **full** over T*L=1232 sat tokens | spatiotemporal; avoids axial/TiTok-correspondence assumption |
| Backbone size | **B** | matches the 7-arm comparison family |
| TiTok correspondence diagnostic | **skipped** | user decision; full self-attn has no axial assumption to validate |

Param cost: ~12·D² per layer = ~7.1M @ D=768; N=6 ≈ **+42M**. arm8-B ≈ 179.5M
(arm7) + 42M ≈ **~221M**.

## 5. Backward compatibility (hard constraint)

- New flag `sat_context_encoder_layers` defaults to **0** → module not
  constructed, forward branch not taken → **bit-identical to arm 7**.
- Arms 1–7 configs and checkpoints, and the in-flight XL/H big-runs, are
  unaffected: no change to the main stream, no change to legacy modules, no new
  required config keys.
- Guarded attribute reads (`getattr(..., False)`) so non-cross-attn models never
  reference arm-8 attributes.

## 6. Experiment setup (clone the established arm-7 / capacity-ladder flow)

1. **Config** `configs/Sat2Radar-v2v-cmp-satenc-B-2021summer_gadi.py`: clone
   `Sat2Radar-v2v-cmp-xattn-B-2021summer_gadi.py`; add
   `sat_context_encoder_layers=6` to the `model` args dict (next to
   `use_cross_attention=True`); change `name`/workdir to `...satenc...`.
   Everything else identical: cross_attention, radar_tokens, 1445-clip
   `dataset_filelist_v2v_baseline_2021summer.pkl`, 60k steps, bs=8, run1 AEs.
2. **Tiny gate** `configs/Sat2Radar-v2v-cmp-satenc-tiny_gadi.py` +
   `train_v2v_cmp_satenc_tiny_gadi.sh`: arm 8 is new architecture, so follow the
   arm 6/7 protocol — tiny overfit gate (pass = diff_loss < 0.3 & step ≥ 11900)
   that auto-qsubs the full job on pass.
3. **Train** `train_v2v_cmp_satenc_B_gadi.sh`: clone
   `train_v2v_cmp_xattn_B...`; TARGET=60000, gpuhopper, self-resubmit;
   auto-qsub the holdout test on reaching 60k.
4. **Test** `holdout_test_v2v_cmp_satenc_gadi.sh`: clone
   `holdout_test_v2v_cmp_xattn_gadi.sh`; **gen-metrics ON** (FVD/KVD/TC), full
   2901-clip test set, seed 42. Joins the same comparison table.

Workdir: `Experiments/sat2radar_flowtok_v2v_cmp_satenc_B`.

## 7. Testing

Unit tests (`tests/test_ablation_units.py`, following the arm 6/7 pattern):
- **Backward-compat**: a cross-attn model with `sat_context_encoder_layers` unset
  / 0 has no `sat_context_encoder` module and produces output bit-identical to
  the pre-change arm-7 path (same seed, same input).
- **Construction**: with `sat_context_encoder_layers=6`, the module exists with
  6 `SatEncoderLayer`s and a final norm.
- **Shape**: `_forward` output shape unchanged vs arm 7 for the same input.
- **Forward smoke**: end-to-end forward runs for both i2i (L=77) and v2v
  (L=T*77=1232) token lengths.
- **Param-count sanity**: arm8-B param count ≈ arm7-B + ~42M.

Empirical validation:
- Tiny overfit gate must pass before the full run (diff_loss < 0.3).
- Holdout: arm8 vs arm7 on the standard 2901-clip 2024-07~2025-07 test set;
  primary metric weighted FSS, plus FVD/KVD/TC.

## 8. Risks / open notes

- **+42M params on 1445 clips** → overfit risk; the tiny gate + holdout FSS-vs-arm7
  decide. If arm8 underperforms arm7, the encoder depth (N) or dataset size is the
  suspect, not the mechanism.
- Final-norm placement (post-stack) is a deliberate choice; if K/V scale drifts,
  revisit.
