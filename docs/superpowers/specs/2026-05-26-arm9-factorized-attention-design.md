# Arm 9 — Factorized (frame-local + axial-temporal) Attention (design)

**Date:** 2026-05-26
**Status:** design approved, pending spec review → implementation plan
**Branch:** `v2v-7arm-comparison` (cluster reads the live tree; do NOT switch branches / merge while jobs run)

## 1. Motivation

The current FlowTok v2v DiT (arms 3-8) runs **full self-attention over the flattened
T·L = 16·77 = 1232 radar-token sequence** — space and time mixed in one global
attention, the same camp as CogVideoX/Sora. Arm 9 replaces that single full
self-attention with **factorized divided space-time attention** (TimeSformer-style):
a **frame-local spatial** self-attention (each frame's 77 tokens attend only within
the frame) followed by an **axial temporal** self-attention (token index *j* attends
across the 16 frames). The cross-attention to satellite K/V and the MLP stay exactly
as arm 7.

**Hypothesis:** on the small 1445-clip set, a stronger structural prior (factorization)
may regularize better than full attention's freedom — i.e. beat arm 7 by *reducing
overfitting*, not by adding capacity. (SOTA video DiTs went the other way — full 3D
attention — for quality at scale; this experiment probes the small-data regime.)

**Axial caveat (accepted):** axial temporal attention assumes token index *j* corresponds
across frames. FlowTiTok produces 1D (non-grid) tokens, so this correspondence is not
guaranteed. The user chose to proceed axial-first and **skip** a separate TiTok
correspondence diagnostic; if arm 9 underperforms, this assumption is a prime suspect.

## 2. Scope

In scope:
- New `FactorizedDiTBlock` class in `libs/model/flowtok_t2i.py` (DiTBlock untouched).
- New config flag `use_factorized_attn` (default False).
- Block-list construction switch in `FlowTok.__init__`.
- Config / PBS train / tiny-gate / holdout scripts cloned from arm 7 (xattn-B).
- Unit tests (backward-compat, construction, forward shapes, reshape correctness).

Out of scope:
- **Modifying `DiTBlock`** — explicitly NOT touched (Approach B). The running XL/H/m8
  jobs and arms 1-8 import `DiTBlock` from the live tree; leaving it byte-identical
  makes "flag off = bit-identical" a structural guarantee, not a test result.
- The main token-stream `x_embedder`/pos-emb path, legacy `context_encoder`/
  `context_projector`, `diffusion/flow_matching.py`, `scripts/test_sat2radar_v2v.py`.
- arm 8's `SatContextEncoder` (orthogonal; not combined here).
- Non-axial / full cross-frame temporal attention (deferred; axial-first per user).
- CFG changes (null_indicator stays all-False, as arm 7).

## 3. Architecture

### 3.1 New block (Approach B: separate class; reuses timm `Attention`, existing `CrossAttention`/`Mlp`/`modulate`)

Mirrors the existing cross-attn `DiTBlock` style exactly: `nn.LayerNorm(..., elementwise_affine=False, eps=1e-6)` (affine comes from adaLN `modulate`), checkpointed `_forward`, `adaLN_modulation = Sequential(SiLU, Linear(D, n_mod·D))`. Here `n_mod = 12` (4 sub-layers × shift/scale/gate).

```python
class FactorizedDiTBlock(nn.Module):
    """Arm 9: factorized divided space-time attention with adaLN-Zero.
    Replaces DiTBlock's single full self-attention with frame-local spatial SA +
    axial temporal SA; keeps arm7's cross-attention (sat K/V) + MLP. 4 sub-layers,
    adaLN chunk(12). Full self-attn is NOT present."""
    def __init__(self, hidden_size, num_heads, n_per_frame, mlp_ratio=4.0):
        super().__init__()
        self.n_per_frame = n_per_frame
        self.norm_sp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_sp = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm_tp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_tp = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm_ca = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio),
                       act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 12 * hidden_size, bias=True))

    def forward(self, x, c, context=None):
        return torch.utils.checkpoint.checkpoint(self._forward, x, c, context, use_reentrant=False)

    def _frame_local(self, attn, x):
        # [B, T*L, D] -> per-frame groups [B*T, L, D] -> attn within frame -> back
        B, S, D = x.shape; L = self.n_per_frame; T = S // L
        x = x.reshape(B * T, L, D)
        x = attn(x)
        return x.reshape(B, S, D)

    def _axial_temporal(self, attn, x):
        # [B, T*L, D] -> per-index groups [B*L, T, D] -> attn across frames -> back
        B, S, D = x.shape; L = self.n_per_frame; T = S // L
        x = x.reshape(B, T, L, D).transpose(1, 2).reshape(B * L, T, D)
        x = attn(x)
        return x.reshape(B, L, T, D).transpose(1, 2).reshape(B, S, D)

    def _forward(self, x, c, context=None):
        assert x.shape[1] % self.n_per_frame == 0, "seq_len must be divisible by n_per_frame"
        (sh_sp, sc_sp, g_sp,
         sh_tp, sc_tp, g_tp,
         sh_ca, sc_ca, g_ca,
         sh_mlp, sc_mlp, g_mlp) = self.adaLN_modulation(c).chunk(12, dim=1)
        x = x + g_sp.unsqueeze(1) * self._frame_local(self.attn_sp, modulate(self.norm_sp(x), sh_sp, sc_sp))
        x = x + g_tp.unsqueeze(1) * self._axial_temporal(self.attn_tp, modulate(self.norm_tp(x), sh_tp, sc_tp))
        x = x + g_ca.unsqueeze(1) * self.cross_attn(modulate(self.norm_ca(x), sh_ca, sc_ca), context)
        x = x + g_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sh_mlp, sc_mlp))
        return x
```

Token order is frame-major (pos-emb temporal = `arange // n_per_frame` confirms it), so
`reshape(B, T, L, D)` correctly maps `[b, t, l] → token t*L + l`.

### 3.2 Block-list construction (`FlowTok.__init__`)

Replace the single block-list construction with a flag switch (DiTBlock branch unchanged):

```python
        if getattr(config, "use_factorized_attn", False):
            self.blocks = nn.ModuleList([
                FactorizedDiTBlock(hidden_size, num_heads,
                                   n_per_frame=num_latent_tokens, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                         use_cross_attn=self.use_cross_attention)
                for _ in range(depth)
            ])
```

`n_per_frame = num_latent_tokens` (= 77; NOT `pos_n_per_frame`/2L — arm 9 is not interleaved). arm 9 requires `use_cross_attention=True` so `context_embedder` is built and `ctx` is passed; the config sets both flags True.

### 3.3 Data flow / init / forward
- `_forward` (model) is UNCHANGED: `x = block(x, c, ctx) if self.use_cross_attention else block(x, c)`. With `use_cross_attention=True`, `FactorizedDiTBlock.forward(x, c, context=ctx)` is called.
- `initialize_weights()` zeros `block.adaLN_modulation[-1]` for every block in `self.blocks` (it does not reference `block.attn`/`norm1`), so all 12 modulation outputs start at 0 → every sub-layer gated to identity at init (adaLN-Zero), same stable start as arm 7.
- Main token stream, pos-emb, cross-attn (global → all sat K/V), MLP: all as arm 7.

### 3.4 i2i graceful degradation
For i2i (`L=77`, `T=1`): frame-local = full self-attn over the 77 tokens; axial temporal reshapes to `[B*L, 1, D]` → attention over a length-1 sequence = a value/proj transform (no cross-frame mixing possible). Forward stays well-defined.

## 4. Locked parameters

| Parameter | Value | Rationale |
|---|---|---|
| Factorization | **replace** full self-attn with [frame-local spatial] + [axial temporal] | user decision (TimeSformer divided attn) |
| Temporal mode | **axial** (same token index across frames) | user decision (axial-first) |
| TiTok correspondence diagnostic | **skipped** | user decision |
| Cross-attn to sat | **global, unchanged from arm 7** | arm 9 = arm7 + factorized self-attn only |
| adaLN | chunk **12** (4 sub-layers), all gates zero-init | adaLN-Zero, matches DiTBlock convention |
| Backbone | **B** (hidden 768, depth 12, heads 12) | matches comparison family |
| Implementation | **Approach B**: separate `FactorizedDiTBlock`, DiTBlock untouched | running jobs read live tree → structural backward-compat |

Param cost: per block vs arm7-cross-block, +1 full attention (≈4·D²) + adaLN 9→12 (+3·D²) = +7·D² ≈ +4.13M @ D=768; ×12 ≈ **+49M** → arm9-B ≈ **~229M** (verify in impl).

## 5. Backward compatibility (hard constraint)
- `use_factorized_attn` defaults **False** → `else` branch → existing `DiTBlock` construction, **bit-identical** to arms 1-8.
- `DiTBlock` and all other modules are **not modified** → in-flight XL/H/m8 jobs and arms 1-8 unaffected even on resume.
- Guarded read `getattr(config, "use_factorized_attn", False)` so configs without the key behave exactly as before.

## 6. Experiment setup (clone arm-7/arm-8 flow; name `fact`)
1. **Config** `configs/Sat2Radar-v2v-cmp-fact-B-2021summer_gadi.py`: clone xattn-B; add `use_factorized_attn=True` to the `model` Args dict (alongside `use_cross_attention=True`); workdir → `sat2radar_flowtok_v2v_cmp_fact_B`. Everything else identical (cross_attention, radar_tokens, 1445-clip 2021summer, 60k, bs=8, run1 AEs).
2. **Tiny gate** `configs/Sat2Radar-v2v-cmp-fact-tiny_gadi.py` + `train_v2v_cmp_fact_tiny_gadi.sh`: new architecture → overfit gate (pass = diff_loss<0.3 & step≥11900) auto-qsubs full. Reuse the existing `..._xattn_tiny/dataset_filelist.pkl` (arm-agnostic overfit clips).
3. **Train** `train_v2v_cmp_fact_full_gadi.sh`: clone xattn-full; TARGET=60000, gpuhopper, self-resubmit; auto-qsub the holdout test at 60k.
4. **Test** `holdout_test_v2v_cmp_fact_gadi.sh`: clone xattn holdout; **gen-metrics ON** (FVD/KVD/TC), full 2901-clip test set, seed 42.

Workdir `Experiments/sat2radar_flowtok_v2v_cmp_fact_B`.

## 7. Testing
Unit tests in `tests/test_ablation_units.py`:
- **Backward-compat:** model without `use_factorized_attn` builds `DiTBlock`s (not `FactorizedDiTBlock`); a `use_factorized_attn=False` model's `state_dict` keys == the pre-change arm7 model's keys.
- **Construction:** `use_factorized_attn=True` (+ `use_cross_attention=True`) → `self.blocks` are `FactorizedDiTBlock`, count == depth, each with `attn_sp`/`attn_tp`/`cross_attn` and adaLN Linear out-dim == 12·hidden.
- **Forward shape:** output shape preserved for v2v (`L=77`, `T≥2`) and i2i (`T=1`); finite.
- **Reshape correctness:** (a) round-trip — `_frame_local`/`_axial_temporal` with identity attn return the input unchanged; (b) grouping — encode token value = (frame_id, token_idx); confirm `_frame_local`'s internal `[B*T, L, D]` groups by frame and `_axial_temporal`'s `[B*L, T, D]` groups by token index (e.g. via a mean-pool attn that makes within-group tokens identical).
- **Param-count sanity:** arm9-B param count ≈ arm7-B + ~49M.
- **Full suite:** all pre-existing arm1-8 tests still pass (no regression).

Empirical: tiny gate must pass before full; holdout arm9 vs arm7 (primary wFSS + FVD/KVD/TC).

## 8. Risks / notes
- **Axial assumption on 1D TiTok tokens** (§1) — the main scientific risk; accepted by user.
- **+49M on 1445 clips** — overfit risk; tiny gate + holdout decide.
- Two reshape/transpose helpers are the error-prone part → covered by the reshape-correctness tests (§7).
