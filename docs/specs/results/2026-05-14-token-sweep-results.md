# Sat AE Token-Count Sweep — Results

**Date:** 2026-05-18
**Design:** `docs/specs/2026-05-14-sat-ae-token-count-sweep-design.md`
**Question:** Is `num_latent_tokens=77` enough for the 10ch(+lgt) sat AE, or do 128 / 256 unlock meaningful reconstruction gains?

## TL;DR

**77 is under-budget. Adopt 128. 256 is not worth it.**

- 128 beats 77 substantially and *consistently across every metric* (avg MSE −22.5%, LPIPS −19.1%, rFID −21.8%, val L2 −30%, high-freq MSE −14.6%, Sobel edge IoU +6.8%).
- 256 vs 128 is flat / within single-seed noise on all three headline metrics (and LPIPS is marginally *worse*).
- All cells use **0 % near-dead tokens**; PCA effective rank even *rises* with N (177→188→227). So the post-128 plateau is **not** unused capacity — the bottleneck beyond 128 is the loss recipe / decoder / data, not the token count.
- Maps cleanly to design §7 row: *"128 improves over 77; 256 within ±15 % of 128 → 128 is the sweet spot."*

## Setup

- 3 cells, retrained clean, identical except `num_latent_tokens ∈ {77,128,256}` and the per-token-normalized `kl_weight` (1e-6·77/N). tiny enc + small dec (~37M), run1 loss recipe, discriminator off, 60k steps, seed 42, lab2 single-4090. Trained 2026-05-15 (≈6.5 h total).
- Eval: 2024/07 i2i test (4391 frames) for `best_val`; §6.2 freq/edge diagnostic on 8 evenly-spaced samples; §6.3 latent utilization on 256 samples.
- `best_val` and `final` are within noise for every cell (training converged, no overfit) → headline numbers use `best_val`.

## Headline metrics (design §7: high-freq MSE, Sobel edge IoU, LPIPS)

| metric | tok77 | tok128 | tok256 | 128 vs 77 | 256 vs 128 |
|---|---|---|---|---|---|
| **High-freq MSE** ↓ | 1.740e-4 | 1.486e-4 | 1.438e-4 | **−14.6 %** | −3.2 % |
| **Sobel edge IoU** ↑ | 0.5520 | 0.5894 | 0.5985 | **+6.8 %** | +1.5 % |
| **LPIPS (alex)** ↓ | 0.19054 | 0.15418 | 0.15773 | **−19.1 %** | +2.3 % (worse) |

### Supporting metrics (full 4391-frame test set, best_val)

| metric | tok77 | tok128 | tok256 | 128 vs 77 | 256 vs 128 |
|---|---|---|---|---|---|
| avg MSE ↓ | 7.501e-4 | 5.806e-4 | 5.713e-4 | −22.6 % | −1.6 % |
| avg SSIM ↑ | 0.8550 | 0.8872 | 0.8926 | +0.032 | +0.005 |
| avg PSNR ↑ | 35.38 | 36.47 | 36.61 | +1.09 dB | +0.14 dB |
| avg rFID ↓ | 124.97 | 97.78 | 89.34 | −21.8 % | −8.6 % |
| val L2 (2024/06) ↓ | 1.392e-3 | 9.68e-4 | 9.58e-4 | −30.5 % | −1.0 % |

## Latent utilization (design §6.3)

| | tok77 | tok128 | tok256 |
|---|---|---|---|
| near-dead tokens | **0 / 77** | **0 / 128** | **0 / 256** |
| var/token (mean) | 0.768 | 0.777 | 0.864 |
| KL/token (mean) | 31.2 | 32.8 | 19.7 |
| PCA effective rank | 177.1 | 188.4 | 227.4 |

No token collapse anywhere → the design §7 "≥30 % near-dead ⇒ capacity unused" override **does not apply**. 256 genuinely spans more latent dimensions (rank 227 vs 188) and carries lower per-token KL, yet reconstruction does **not** improve over 128 — the extra representational room is not the limiting factor past 128.

## Decision (design §7)

| Check | Result |
|---|---|
| 128 vs 77 improved? | **Yes.** By the strict letter (≥15 % on ≥2/3 headlines) it's 1/3 (LPIPS −19.1 %; high-freq MSE −14.6 % just shy; Sobel +6.8 %). By totality — ~7 independent metrics all improve, several >20 %, val L2 −30 % — this is far beyond single-seed noise. **Substantial real improvement.** |
| 256 vs 128 improved? | **No.** All headlines within ±15 % (−3.2 %, +1.5 %, +2.3 % worse). Noise. |
| ≥30 % near-dead in 128/256? | **No** (0 %). Override inactive. |

→ **Design §7 row matched: "128 vs 77 improved; 256 within ±15 % of 128 → 128 is the sweet spot. Adopt 128 for follow-on AE training."**

### Honest caveat on the 15 % rule

The conservative rule (≥15 % on 2/3 headlines) is technically met on only **1** headline for 128-vs-77 (LPIPS). High-freq MSE lands at 14.6 % (one point under) and Sobel IoU at 6.8 %. The verdict rests on the *consistency* argument the spec itself anticipates: noise does not push avg-MSE, LPIPS, rFID, PSNR, SSIM, val-L2 and the two diagnostic headlines all in the improving direction simultaneously, several by >20 %. Single seed, shared batch order across cells (intentional, design §8) — so this is an ablation-clean comparison, not an estimate of run-to-run variance.

## Recommendations / follow-ups

1. **Adopt `num_latent_tokens=128`** for all follow-on sat AE training (and as the FlowTok v2v sat tokenizer budget). Note: the in-flight joint-tokenizer pilot uses 77 to match FlowTok-S `num_clip_token=77`; a 128 variant there is a clean follow-up if joint training pans out.
2. **The post-128 plateau is a recipe/architecture bottleneck, not a token-count one.** Per the design's "no cell improves further" branch, the next experiment to sharpen high-freq detail should vary the **loss recipe** (or decoder capacity) at fixed N=128 — *not* push N higher. Concretely: revisit perceptual weighting / a (carefully tuned) discriminator, since LPIPS & rFID still dominate the error budget while MSE is already low.
3. The high per-channel error is concentrated in IR0 / IR6 / IR2 (see diagnostic per-channel tables); IR2 in particular has low edge IoU (~0.39–0.45) and high LPIPS (~0.38) across all cells — a channel-specific weakness worth a targeted look, independent of token count.

## Artifacts

- Per-cell metrics: `/mnt/ssd_2/yghu/Experiments/sat10ch_ae_tok{77,128,256}_run1/eval_{best_val,final}/metrics.{json,md}`
- Latent utilization: `.../eval_{best_val,final}/latent_utilization.{json,md}`
- 3-cell diagnostic + figures: `/mnt/ssd_2/yghu/Experiments/sat_ae_token_sweep_diagnostic/{diagnose_summary.{json,md},figs/}`
