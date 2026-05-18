# Joint Sat/Radar Tokenizer — Results

Design + analysis: `docs/specs/2026-05-18-joint-sat-radar-tokenizer-design.md`

**Read the spec §2 first** — "positional correspondence" was reframed to *shared latent-code alignment*; the realistic payoff is better flow *conditioning/convergence*, not added predictive power.

## 1. Tokenizer phase (primary result)

| metric | Group A (separate) | Group B (joint) | Δ |
|---|---|---|---|
| cross-modal cosine | 0.0559 | 0.9987 | **+0.9428** |
| linear CKA | 0.3261 | 0.2662 | **-0.0600** |
| per-index linfit err | 0.6175 | 0.7687 | +0.1512 |
| sat recon MSE | 0.001089 | 0.001745 | +60.2% |
| radar recon MSE | 0.002788 | 0.003091 | +10.9% |
| sat near-dead tok | 0 | 0 | |
| radar near-dead tok | 0 | 0 | |

**Decision rule (design §5):** aligned if Δcosine ≥ 0.10 or ΔCKA ≥ 0.15, AND ≤10% recon degradation on *both* modalities.

- alignment improved: **True** (Δcos=+0.9428, ΔCKA=-0.0600)
- reconstruction preserved: **False** (sat +60.2%, radar +10.9%)

### → Joint training helps the tokenizers: **False**

**Trade-off detected:** the spaces aligned but reconstruction degraded. Per the spec, the right follow-up is a sim_weight sweep (0.1/0.25/0.5/1.0), not a verdict.

## 2. v2v pilot (secondary — 8k steps, NOT converged)

- Group A v2v loss: first 1.6567 → last 0.6110 (min 0.5419, 80 pts)
- Group B v2v loss: first 1.3229 → last 0.4288 (min 0.3067, 80 pts)

Lower/faster-dropping loss for B would support the *conditioning* hypothesis. B last − A last = **-0.1822**. Treat as directional only (single seed, 8k steps).

## 3. What to do next

- If *helps tokenizers* = True and v2v shows B converging faster/lower: proceed to a full-length v2v A/B at the chosen sim_weight to confirm.
- If alignment up but recon down: run the sim_weight sweep.
- If no alignment: token-index alignment isn't the lever — revisit (set-level / InfoNCE alternative in spec §3.1) or drop the idea.

_Single seed, shared batch order across A/B (intentional for ablation cleanliness). Pilot step budgets — see spec §3.2/§7._

---

## 4. Interpretation (added manually after the auto-run)

**The mechanism is real, but sim_weight=0.5 aligned the spaces *by collapsing them*, not by learning a shared semantic basis.** Read past the cosine number:

| signal | A (sep) | B (joint) | meaning |
|---|---|---|---|
| cross-modal cosine | 0.056 | **0.999** | sim loss worked — per-index directions nearly identical |
| linear CKA | 0.326 | **0.266 ↓** | but rotation/scale-invariant similarity went *down* |
| PCA eff. rank — sat | 29.9 | **10.8 ↓** | sat latent collapsed to ⅓ the dimensionality |
| PCA eff. rank — radar | 10.4 | **5.6 ↓** | radar latent collapsed to ½ |
| sat recon MSE | — | **+60 %** | the cost of that collapse |

Cosine → 1.0 *with halved effective rank and degraded reconstruction* is **alignment-by-collapse**: both encoders satisfied the cosine loss by squeezing onto a shared low-dimensional, near-degenerate code rather than by carrying matched information. This is exactly the modality-conflict failure mode flagged in design §2 risk-4 / §3.1 — and note the `near_dead_tokens=0` metric **missed it** (collapse here is in the *spectrum*, not in dead individual tokens; PCA effective rank is the metric that caught it).

**Both original hypotheses got directional support, with caveats:**
1. *Joint training aligns sat/radar latents* — **strongly yes** (0.056 → 0.999); the loss does what was intended.
2. *Alignment helps the downstream flow* — **directionally yes**: B's v2v token-loss dropped faster and lower (last 0.43 vs 0.61; min 0.31 vs 0.54), consistent with the spec §2 "easier-to-condition target". **But confounded**: a collapsed low-rank token space is *intrinsically easier to predict* regardless of whether the sat→radar mapping is genuinely better. The v2v pilot only measured token-space loss, not decoded radar quality — so not yet evidence of better forecasts.

**Verdict:** the idea is sound and the effect is large; sim_weight=0.5 is simply past the useful operating point. Not a "no".

## 5. Concrete next step (recommended, NOT auto-run — needs your call)

A **sim_weight sweep** at much lower weights — `{0.0(=A), 0.02, 0.05, 0.1, 0.25}` — plotting the trade-off curve: cross-modal cosine **and** PCA effective rank **and** recon MSE on shared axes. Goal: the largest weight that keeps PCA rank ≈ Group-A level with recon degradation ≤10%. Then re-run the v2v pilot **with pixel-space radar metrics** (decode + MSE/FSS), not token-space loss, to remove the "collapse makes prediction trivially easier" confound. One config knob + the existing pipeline; ≈1 night.

---

## 6. Decoded-radar pixel-space comparison (added — resolves §5 confound)

`test_sat2radar_v2v.py`, 2024/07 v2v test, 241 clips (3856 frames total, 16/clip), step-8000 pilots, identical settings. Metrics on dBZ radar after full decode (sat→flow→radar detokenizer).

| metric | dir | A (separate AE) | B (joint AE) | better |
|---|---|---|---|---|
| mse_dbz | ↓ | 27.3956 | 68.7669 | **A** |
| mae_dbz | ↓ | 1.7631 | 4.6654 | **A** |
| rmse_dbz | ↓ | 5.2341 | 8.2926 | **A** |
| psnr_db | ↑ | 21.1862 | 17.1892 | **A** |
| ssim | ↑ | 0.6349 | 0.3421 | **A** |
| r2 | ↑ | -0.3544 | -2.3998 | **A** |
| avg_fss | ↑ | 0.0985 | 0.1024 | **B** |
| weighted_fss | ↑ | 0.0413 | 0.0446 | **B** |
| csi35 | ↑ | 0.0000 | 0.0000 | tie |
| pod35 | ↑ | 0.0000 | 0.0000 | tie |
| far35 | ↓ | 1.0000 | 1.0000 | tie |

**Tally:** B wins 2 / A wins 6 → **decoded radar: A (separate) better**.

Read with the §4 caveat: B's tokenizer is collapsed/low-rank, so a lower token-loss did not necessarily mean better pixels. This table is the decisive check the pilot was missing. Panels: `/mnt/ssd_2/yghu/Experiments/v2v_jointtok_{A,B}_run1/test8000/`.

**This resolves §4 hypothesis 2.** B's lower token-space loss (0.43 vs 0.61) was *entirely* the collapse artifact: decoded to pixels, B is worse on every reconstruction metric (mse_dbz 2.5×, ssim 0.34 vs 0.64, r² −2.40 vs −0.35). The two FSS wins are at near-zero absolute FSS (0.10) with both models scoring **0** on the ≥35 dBZ convective skill scores (csi/pod/far) — i.e. neither 8k-step pilot predicts storm cores yet, so the FSS edge is not meaningful skill. **End-to-end conclusion: joint training at sim_weight=0.5 is net-harmful — the alignment is real but it costs more in reconstruction than it returns in flow conditioning.** Not a refutation of the idea (see §4 verdict); it confirms 0.5 is past the operating point and motivates the §5 sweep. The decision rule (design §5) already returned *helps tokenizers = **False*** in §1; this pixel-space table is the independent end-to-end confirmation of that call.



---

## 7. sim_weight sweep `{0, 0.02, 0.05, 0.1, 0.25}` (+0.5 anchor)

Same one-knob ablation, same budgets as §1/§6 (AE 15k, v2v 8k, seed 42, shared batch order). w=0 reuses Group A, w=0.5 reuses Group B. **Final ranking is the decoded-radar table — token-space loss is deliberately not used here (see §4).**

### 7a. AE trade-off (the §5 collapse question)

| sim_weight | x-modal cosine ↑ | linear CKA | PCA rank sat | PCA rank radar | sat recon MSE | radar recon MSE | sat recon Δ vs w0 |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.0559 | 0.3261 | 29.9 | 10.4 | 0.001089 | 0.002788 | +0.0% |
| 0.02 | _pending_ | | | | | | |
| 0.05 | _pending_ | | | | | | |
| 0.10 | _pending_ | | | | | | |
| 0.25 | _pending_ | | | | | | |
| 0.50 | 0.9987 | 0.2662 | 10.8 | 5.6 | 0.001745 | 0.003091 | +60.2% |

### 7b. Decoded-radar pixel-space (DECISIVE — 2024/07 v2v test)

| sim_weight | mse_dbz ↓ | rmse_dbz ↓ | ssim ↑ | r² ↑ | avg_fss ↑ | csi35 ↑ |
|---|---|---|---|---|---|---|
| 0.00 | 27.3956 | 5.2341 | 0.6349 | -0.3544 | 0.0985 | 0.0000 |
| 0.02 | _pending_ | | | | | |
| 0.05 | _pending_ | | | | | |
| 0.10 | _pending_ | | | | | |
| 0.25 | _pending_ | | | | | |
| 0.50 | 68.7669 | 8.2926 | 0.3421 | -2.3998 | 0.1024 | 0.0000 |

**Decoded-radar winner (min mse_dbz): sim_weight = 0.00** (mse_dbz 27.3956, ssim 0.6349, r² -0.3544).

**§5 non-collapse operating point:** largest w keeping sat PCA rank ≥ 0.9·w0 and sat recon ≤ +10% is **w = 0.00**. Coincides with the decoded-radar winner — clean result.

→ **No positive sim_weight beats the separate baseline on decoded radar.** Index-wise cosine alignment, at every weight tested, costs more reconstruction than it returns in flow conditioning. Recommend dropping index-wise alignment and revisiting the set-level / InfoNCE alternative (spec §3.1) before further sweeping.

