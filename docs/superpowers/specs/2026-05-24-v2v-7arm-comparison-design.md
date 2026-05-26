# V2V 7-Arm Method Comparison on the 2021-Summer Small Set

**Date:** 2026-05-24
**Status:** Approved (design), pending implementation plan
**Repos touched:** `FlowTok/` (arms 3–7, new code), `Diffi2i-shrimp-proj2/` (arms 1–2, scripts only), shared filelists under `/g/data/kl02/yh0308/Data/71/filelists/`

## 1. Goal

Reproduce the i2i **M-series ablation methodology** for v2v: train 7 satellite→radar
methods on **one shared small dataset** (summer-2021) so they are directly comparable,
and decide which v2v conditioning method is best. This mirrors the i2i M-series, which
compared FlowTok arms m1–m8/m11 plus Diffi2i baseu (m9) / diu (m10) on
`dataset_filelist_i2i_baseline_2021summer_merged.pkl`.

The point of the small set is **fast iteration** — this is a method comparison, not the
200k-step XL full runs (those m8/m8align XL jobs are running separately and are NOT part
of this work; they must not be disturbed).

## 2. Dataset (shared by all 7 arms)

- **Source of truth for the date range:** `dataset_filelist_i2i_baseline_2021summer_merged.pkl`
  (the exact summer-2021 range the i2i M-series used).
- **Build approach:** filter the full v2v train pkl
  `dataset_filelist_v2v_train_201906_202312.pkl` down to that date range, rather than
  rebuilding clips from raw `.npy`. Both FlowTok and Diffi2i consume the same
  `(train, val, test)` v2v filelist format (the existing `train_v2v_baseu_gadi.sh` /
  `train_v2v_diu_gadi.sh` already `cp` such a pkl into the model dir).
- **Output:** `/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_baseline_2021summer.pkl`
- **Clip params:** `num_frames=16`, `frame_stride=1`, `crop_size=128`, `use_lightning=True`,
  `ir_band_indices=None` (all 10 IR bands → 11ch input with lightning). Matches the m8 v2v config.
- **Verification:** confirm the date range read off the i2i pkl matches the filtered v2v clips;
  print clip count; assert >0 train clips.

## 3. The seven arms

All FlowTok arms use **FlowTok-B** (depth 12, hidden 768, 12 heads), **~60k steps**, **bs=8**,
run1 AEs (sat10ch run1 + radar run1, 77 tokens, 16-dim) — same recipe as the i2i M-series,
just v2v (16 frames).

| # | Arm | Backbone | New code | Mechanism |
|---|-----|----------|----------|-----------|
| 1 | baseu v2v | Diffi2i BaseU3d | no | `baseu_training.py --num-frames 16`, in_dim 11, hilburn loss |
| 2 | diu v2v | Diffi2i DiT3d | no | `diff_training.py --num-frames 16`, in_dim 11 |
| 3 | flowtok direct | FlowTok-B | no | `flow_cond_mode="none"`, velocity, sat→radar direct token flow (legacy reference recipe) |
| 4 | block-M8 | FlowTok-B | no | `flow_cond_mode="token_concat"` + predict x1; inp `[sat(16·77) \| noisy_radar(16·77)]`, seq=2464 |
| 5 | aligned | FlowTok-B | no | `flow_cond_mode="token_concat_interleaved"`, per-frame `[sat_i \| radar_i]`, seq=2464 |
| 6 | block + modality pos-emb | FlowTok-B | **yes** | block layout + 3-axis sincos pos-emb (spatial/temporal/**modality**) |
| 7 | cross-attention | FlowTok-B | **yes** | radar tokens self-attend; sat tokens injected via per-block cross-attn |

baseu/diu (arms 1–2) keep their native model sizes and training loops. Their step budgets
are **not** directly comparable to the token-flow arms (pixel diffusion vs token flow) —
the controlled comparison is *within* arms 3–7; arms 1–2 are reference baselines.

## 4. New code (FlowTok) — backward-compatible, opt-in

Both new features are gated by config flags defaulting to legacy behavior, so the in-flight
m8/m8align XL jobs (which resume from ckpts) reconstruct the **identical** old model. This
follows the workspace backward-compat rule (new args default = legacy).

### 4.1 Arm 6 — modality positional embedding

- New `flow_cond_mode = "token_concat_modality"`.
- New model flag `use_modality_pos_emb` (default `False`), threaded from
  `config.nnet.model_args` into `FlowTok.__init__`.
- **Data path is the unchanged block `token_concat` path** — input is
  `[cond(T·L) | noisy_radar(T·L)]`, last T·L tokens supervised, predict x1. Only the
  positional embedding changes.
- **`_build_pos_embed` 3-axis variant** (hidden=768, modality boundary = `seq_len//2`,
  `L=num_latent_tokens=77`, `T=16`):
  ```
  spatial  = idx % 77              -> sincos dim 384 (D/2)
  temporal = (idx % (T*77)) // 77  -> sincos dim 192 (D/4)   # 0..15, sat_i & radar_i ALIGNED
  modality = idx // (T*77)         -> sincos dim 192 (D/4)   # 0=sat, 1=radar
  pos = concat([spatial_e, temporal_e, modality_e], -1)      # [L, 768]
  ```
  All three dims are even (sincos requirement). No new learnable params.
- **Difference from arm 5 (interleaved):** arm 5 encodes modality implicitly via a spatial
  offset (`pos_n_per_frame=2L`); arm 6 keeps the block layout but adds an *explicit* modality
  axis and aligns the temporal index so sat-frame-i and radar-frame-i share temporal pos i.

### 4.2 Arm 7 — cross-attention conditioning

- New `flow_cond_mode = "cross_attention"`.
- New model flag `use_cross_attention` (default `False`), threaded from
  `config.nnet.model_args`.
- New `CrossAttention(nn.Module)`: query from radar stream, K/V from sat context;
  multi-head, qkv_bias, same head count as self-attn.
- `DiTBlock(use_cross_attn=True)` variant:
  ```
  shift/scale/gate × 3 = adaLN_modulation(c).chunk(9)   # widened 6 -> 9 for this variant
  x = x + gate_msa * self_attn(modulate(norm1(x)))      # radar <-> radar
  x = x + gate_ca  * cross_attn(modulate(norm_ca(x)), sat_ctx)   # radar -> sat (K,V)
  x = x + gate_mlp * mlp(modulate(norm2(x)))
  ```
  Only constructed when `use_cross_attention=True` (legacy blocks stay 6-wide).
- **Cross-attn gate zero-initialized** (adaLN-zero style) so at init the model output equals
  the self-attn-only path → stable training start.
- `context_embedder = nn.Linear(cond_in_channels, hidden_size)` + spatial/temporal pos-emb
  on the sat context. Only constructed when `use_cross_attention=True`.
- `FlowTok._forward(x, t, null_indicator, context=None)`: when cross-attn is on, embed
  `context` (sat tokens) once and pass to every block's cross-attn.
- **Data path:** main sequence = noisy radar tokens (T·L=1232), NOT concatenated; sat tokens
  (1232) are cross-attn context. Predict x1 on the full radar stream (no slicing).
- `diffusion/flow_matching.py`: new `cross_attention` branch in both the training loss and
  the ODE Euler solver — pass `cond` as `context=`, do not concat, no de-slice.
- `scripts/train_sat2radar_v2v.py`: route sat tokens to `context` for this mode in the
  inference sample call.

## 5. Tests first (TDD)

Add to `FlowTok/tests/test_ablation_units.py` (no pytest; run `python3 tests/test_ablation_units.py`):

**Arm 6:**
- `_build_pos_embed` with `use_modality_pos_emb=True` returns `[1, L, 768]`.
- Dim partition is 384 (spatial) + 192 (temporal) + 192 (modality).
- sat-frame-i token and radar-frame-i token share the temporal sub-vector.
- The two modality halves differ only in the modality sub-vector (spatial/temporal equal for
  aligned positions).

**Arm 7:**
- `CrossAttention(q=[B,Lq,D], kv=[B,Lk,D])` → `[B,Lq,D]`.
- `DiTBlock(use_cross_attn=True)` forward → correct shape with a context arg.
- FlowMatching `cross_attention` training branch: `prediction.shape == radar_tokens.shape`
  (T·L=1232), not 2·T·L.
- Zero-init cross-attn gate ⇒ block output == self-attn-only output at init (identity check).
- ODE solver `cross_attention` path feeds T·L and returns clean radar (T·L), passes context.

All existing tests must stay green (the block `token_concat` / interleaved paths are untouched).

## 6. Files

**Dataset:**
- build script (one-off) → `dataset_filelist_v2v_baseline_2021summer.pkl`

**FlowTok configs** (B + 2021summer; naming `Sat2Radar-v2v-cmp-<arm>-B-2021summer_gadi.py`):
- `direct`, `m8block`, `m8align`, `m8mod` (arm 6), `xattn` (arm 7) full configs
- `m8mod` + `xattn` tiny configs (32-clip overfit)

**FlowTok PBS scripts** (`train_v2v_cmp_<arm>_2021summer_gadi.sh`):
- 5 full scripts (arms 3–7) + 2 tiny scripts (arms 6,7) with the gate→full→self-resubmit contract

**Diffi2i PBS scripts:**
- `train_v2v_cmp_baseu_2021summer_gadi.sh`, `train_v2v_cmp_diu_2021summer_gadi.sh`

**FlowTok code:**
- `libs/model/flowtok_t2i.py` — modality pos-emb branch, `CrossAttention`, cross-attn DiTBlock,
  context embedder, `use_modality_pos_emb` / `use_cross_attention` flags
- `diffusion/flow_matching.py` — `token_concat_modality` (pos-emb only) + `cross_attention` branches
- `scripts/train_sat2radar_v2v.py` — wire pos-emb override + cross-attn context for new modes
- `tests/test_ablation_units.py` — new tests

## 7. Run plan

1. Build dataset pkl; CPU shape-sanity for arms 6 & 7 (forward pass, both flags on/off).
2. Tiny-overfit (32 clips) for arms 6 & 7 → gate `diff_loss<0.3 & step≥11900 & not-NaN` → auto-qsub full.
3. Launch arms 1–5 full directly on the 2021summer set.
4. Self-resubmit (`timeout 47h` + resume) until 60k (FlowTok) / target (baseu/diu).
5. Holdout test (v2v test pkl `dataset_filelist_v2v_test_202407_202507.pkl`) wired after full runs land ckpts — FSS/SSIM/PSNR/MAE, `--seed 42`, same pkl across arms.

## 8. Non-goals / risks

- Not touching the running m8/m8align XL full jobs. Backward-compat flags guarantee this.
- baseu/diu cross-family comparability is qualitative only.
- If a tiny arm fails the gate, write `TINY_FAIL` and do NOT auto-submit full (diagnose first).
- FlowTok-B may underfit v2v vs XL; acceptable for a method-ranking ablation (the winner can be
  re-run at XL later if desired).
