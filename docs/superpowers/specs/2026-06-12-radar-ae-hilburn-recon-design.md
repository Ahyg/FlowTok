# Radar AE v4 "hilGAN" — Hilburn high-dBZ recon + guarded GAN

**Date:** 2026-06-12
**Author:** yh0308 (+ Claude)
**Status:** design approved, pending spec review → implementation plan
**Repo:** FlowTok (1d-tokenizer AE family)

## Problem

The radar bl128 AE GAN-continuation 300k→350k (`run4ftgan2_cond3nan1`) **catastrophically
regressed at high dBZ** while every aggregate metric looked flat — a GAN-overshoot collapse
hidden by mean-regression:

| per-thr FSS | 300k | 350k |
|---|--:|--:|
| 60 dBZ | 0.376 | **0.196** (halved) |
| 55 dBZ | 0.711 | 0.614 |
| 50 dBZ | 0.848 | 0.809 |
| avg_fss | 0.895 | 0.870 (−3% only) |
| PSNR | 50.6 | 51.6 (+1.9%) |

Root cause (from TensorBoard `train/*`): recon (0.0004) and perceptual (0.0066) stayed flat
the whole window — pixel fidelity was fine. The **adversarial branch drifted**: `gan_loss`
0.13→2.0 (15×), `lecam_loss` 0.0004→0.0075 (18×), `logits_fake`→−2.0, inflection at step
320k. The generator chased "looks-real" texture and sacrificed the rare convective cores that
dominate high-threshold FSS but are invisible to MSE/perceptual.

## Goal

Restart the radar AE from the healthy **300k** checkpoint and make reconstruction explicitly
prioritize high-dBZ pixels, so the convective extremes survive (and ideally improve) instead
of being smoothed away — while keeping GAN's sharpness benefit under guardrails that prevent
a repeat of the 320k drift.

Success = per-threshold holdout FSS at 55/60 dBZ recovers toward or above the 300k baseline,
with **no** regression at low/mid thresholds (0–45 dBZ) and no avg_fss loss.

## Design

### 1. Hilburn reconstruction loss (opt-in, backward-compatible)

Add a third reconstruction type `"hilburn"` to `modeling/modules/losses.py`
`ReconstructionLoss_Stage2._compute_recon_loss`, alongside the existing `l1`/`l2`.

Port Diffi2i's element-wise `Hilburn_Loss` (`Diffi2i-shrimp-proj2/src/utils.py`):

```
weight = exp(b * y_true**c)          # y_true = GT radar, normalized [0,1]
loss   = mean( weight * 0.5 * (y_pred - y_true)**2 )
```

Defaults `b=5.0`, `c=3.0` (Diffi2i-proven). At normalized radar: y=1.0 (60 dBZ) → weight
e⁵≈148×, y=0.5 (30 dBZ) → ≈1.9×, y=0 → 1×.

**New `config.losses` keys** (all read via `loss_config.get(...)`, legacy default = current
behavior so the sat config and every other job are untouched):
- `reconstruction_loss: hilburn`  (default stays `l2` when absent)
- `hilburn_b: 5.0`
- `hilburn_c: 3.0`

`reconstruction_weight` stays **1.0** — the cubic-exp self-rebalances (rare high-dBZ errors
lift the recon mean above 0.0004 on their own). Applies to the radar channel only by virtue
of the radar (1-channel) config; sat never sets `reconstruction_loss: hilburn`.

**NaN-safety (verified):** `data/dataset.py:160` `scale_radar_img` fills `nan→0.0` before
clip+normalize, so all-NaN radar regions become 0 dBZ → `exp(5·0³)=1` (weight-1 background).
No NaN poisoning; fill regions are not upweighted; only real cores get amplified.

### 2. GAN guardrails (the 320k-drift fix)

Keep GAN on (`discriminator_start=300000`, `discriminator_weight=0.001` — sweep winner) but:
- **Discriminator LR decay**: cosine `1e-4 → 1e-5` across 300k→350k, so the disc stops
  outpacing the generator (root cause of the drift). Generator LR floor stays 1e-5
  (`force_lr_floor_on_resume: true`).
- **Save final only** (user choice 2026-06-13): `save_every = 50000` so the only new save is
  the final **350k** checkpoint — no intermediate clutter. Tradeoff (accepted): no pre-collapse
  fallback ckpt, so the **disc-LR decay carries the anti-drift safety** (plus Hilburn anchoring);
  the good 300k is always retained regardless. The drift signatures (`lecam_loss`, `gan_loss`)
  are still logged to TB every step for live monitoring (log-only, no auto-abort).
- **No mid-run dataset switch**: dataset fixed = cond3nan1 (71full) for the whole run.

### 3. Restart, data, dirs

- Resume weights from `radar_..._run4ftgan_cond1_gadi/checkpoint-300000` (healthy
  pre-collapse) into a **new** dir `radar_..._run4hilgan_cond3nan1_gadi` (symlink the 300k in,
  same pattern as run4ftgan2). The archived `run4ftgan2` dir is left untouched for A/B.
- Dataset = cond3nan1 full i2i train (71,040 frames),
  `dataset_filelist_i2i_train_201906_202406_cond3nan1_clip16_p005_seed42.pkl`.
- **Target step 350000** (50k window). Generator LR floor 1e-5 throughout.

### 4. Synergy rationale

Hilburn anchors the high-dBZ cores via the recon gradient *while* GAN adds sharpness — the
recon term now actively fights to preserve the convective extremes the GAN was destroying.
That coupling is the bet: GAN for texture, Hilburn for not throwing away the cores.

## Files to create / modify

| Path | Change |
|---|---|
| `modeling/modules/losses.py` | add `hilburn` branch + `hilburn_b`/`hilburn_c` to `_compute_recon_loss` (opt-in, legacy default l2) |
| `configs/radar_..._run4hilgan_cond3nan1_gadi.yaml` | clone of run4ftgan2 radar yaml; `reconstruction_loss: hilburn`, `hilburn_b/c`, disc-LR cosine 1e-4→1e-5, `save_every: 10000`, output_dir→new dir, max_train_steps=350000 |
| `train_radar_..._run4hilgan_cond3nan1_gadi.sh` | PBS clone pointing at the new config |
| (disc-LR decay) | wire cosine disc-LR-decay in the training loop / optimizer setup if not already a config knob — TBD in plan |
| (early-stop monitor) | lecam/gan threshold check — TBD in plan (config flag vs log-only) |

## Verification

1. **NaN check** — DONE (radar nan→0 before loss; Hilburn safe).
2. **Tiny overfit gate** (per gadi-ml-workflow): 32-clip overfit, ~10k steps, vis-freq ~500.
   Pass = Hilburn term finite & decreasing, no NaN, high-dBZ pixels memorized (recon↓ on cores).
   Only then submit full.
3. **Full run** 300k→350k from the new dir, gpuhopper, walltime-aware.
4. **Holdout eval** — same 46,416-frame nofilter clip16 test set, seed 42, per-threshold FSS,
   every 10k ckpt vs radar 300k. Success = 55/60 dBZ FSS recovers toward/above 300k with no
   0–45 dBZ regression and no avg_fss loss.

## Open items (confirmed defaults)

- Target step: **350k** (confirmed).
- Disc-LR decay shape: **cosine 1e-4→1e-5** (confirmed).

## Out of scope

- Sat AE (its 350k is a net win; not touched here).
- Boosting `reconstruction_weight` or spatially-weighting the perceptual loss (user chose the
  minimal-change loss; revisit only if the tiny gate / holdout shows Hilburn is out-competed
  by the perceptual term).
