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

### 6.1 Training-trajectory diagnosis (A vs B, samples 2k→8k)

Reading the `samples/{2000,4000,6000,8000}_sat_lgt_gt_pred.png` arc for both
groups separates the two failure modes:

- **A (separate AE):** 2k sparse speckle → 4k patchy fill → 6k coherent blobs
  + some mid-reflectivity → 8k localized blobs that *track GT location* (no
  cores yet). Monotonically gaining structure, **still moving at 8k**, trending
  *toward* GT. ⇒ healthy tokenizer, **flow merely undertrained**.
- **B (joint AE @0.5):** 2k near-empty → 4k thin edge fragments → 6k broad
  smear → 8k large diffuse over-spread wash that does *not* track GT and is
  getting **blurrier/broader**, not sharper. Converging toward a degenerate
  low-rank field, *away* from GT. ⇒ **flow undertrained AND tokenizer
  collapsed**.

The two groups head to **different attractors** (A sharpens toward GT; B
smears away) — this is the visible fingerprint of §4 alignment-by-collapse:
B's radar latent is rank-halved (PCA eff. rank sat 29.9→10.8, radar
10.4→5.6; 0 near-dead ⇒ spectral, not per-token), so the detokenizer can only
render a low-dimensional blur regardless of flow quality. A's latent is intact
(recon MSE ~2× better relative to B's degradation; decoded r² −0.35 vs B
−2.40), so A is flow-budget-limited, not tokenizer-limited.

**Consequences:** (1) §6's "A > B" is mechanistically explained — not that A's
flow is good (r²<0) but that B is *doubly* handicapped; the gap should persist
/ widen with budget since B's ceiling is structurally lower. (2) 8k judged
neither on quality — both mid-trajectory; the big-budget rerun is necessary.
(3) At w=0.5 the damage is to the **tokenizer ceiling**, not just the flow —
so the §7 low-weight cells {0.05, 0.25} **+ the new pure-AE recon panels** are
the right instrument: they show, per weight, whether alignment can be gained
*without* dropping PCA rank / raising recon MSE (= without lowering the
ceiling). A's own 8k tokenizer ceiling is unknown (recon image was pruned);
the §7 w000 cell (separate, 25k AE / 40k v2v + recon panel) settles it.

### 6.2 Big-budget sweep — operational post-mortem + the verdict it still yields

The reduced big-budget sweep ran 2026-05-18 11:25→17:45 and reported "ALL
DONE", but **§7b is empty (0/3 decoded)**. Three compounding failures, none of
which invalidate the AE-phase result that *did* survive:

1. **Launcher ckpt-test bug (mine).** v2v writes the checkpoint as a *directory*
   `…/ckpts/40000.ckpt/` (a TrainState dir; `test_sat2radar_v2v.py:437`). The
   launcher gated DECODE on `[ -f "${CK}" ]` — a *regular-file* test, false for a
   directory. Both **w000 and w005 v2v fully completed all 40000 steps**
   (`output.log`: "Finish fitting, step=40000", "Save checkpoint 40000…"), but
   DECODE was skipped on the false `-f`.
2. **Unconditional end-of-cell prune.** `rm -rf "${V2V}/ckpts"` (and the AE
   `checkpoint-*` dirs) runs every cell regardless of whether DECODE consumed
   them — so the completed 40k v2v ckpt *and* the 25k AE ckpts were deleted.
   Nothing is checkpoint-recoverable; a re-run is required to get §7b.
3. **Shared-GPU OOM on w025.** The GPU-free wait loop runs *once* at sweep
   start (11:25, GPU idle). At 17:44 two foreign processes (user `yxma`,
   `openpi-sdvla`, 2×7.75 GiB) seized GPU 0 mid-sweep; w025 AE OOM'd at step ~0.
   lab2 GPU 0 is shared and the loop never re-checked per cell.

**What survived is the AE phase for w000/w005 — and it already settles w=0.05**
(see 7a; the decoded table would only re-confirm). Numbers `eval`'s 7a table
omits, pulled from `align_{w000,w005}.json`:

| signal | w000 (separate) | w005 (joint 0.05) | read |
|---|---|---|---|
| x-modal cosine | −0.031 | **0.990** | alignment loss "works" |
| linear CKA | 0.387 | **0.204 ↓** | rotation-invariant sim *down* |
| PCA rank sat | 53.1 | **34.8 (−34%)** | sat latent collapsed |
| sat recon MSE | 0.000851 | **+46.3%** | ≫ design-§5 +10% gate |
| radar recon MSE | 0.002250 | **+23.2%** | also ≫ +10% |
| **near-dead tok / 77 — sat** | **0** | **57** | **74% of codebook dead** |
| **near-dead tok / 77 — radar** | **0** | **65** | **84% of codebook dead** |

This is *worse* than the §4 sim_weight=0.5 pilot, not better: at 0.5/15k the
collapse was purely spectral (near_dead=0, PCA-rank only). At the **smaller
weight 0.05 with the bigger 25k budget** the collapse is so severe it kills
74–84 % of individual tokens outright. More training at low weight does not
soften alignment-by-collapse — it *deepens* it.

**Tokenizer-ceiling panels (the reference the user asked for; survived):**
`joint_tok_align/recon_{w000,w005}.png` — pure encode→decode, no flow, the
best radar the detokenizer can ever render.

- **w000 (separate):** radar recon tracks the GT storm cells (mild blur, cores
  roughly placed). Healthy ceiling — if its v2v is poor, blame the flow budget.
- **w005 (joint 0.05):** radar recon is a diffuse blue smear with **no
  convective structure**, even with zero flow error. The collapsed latent
  *physically cannot* render cores; any downstream v2v decode is capped at this
  smear.

**Verdict (does not need §7b):** index-wise cosine alignment is **refuted as
implemented**. At the smallest swept weight and the big budget it collapses
~80 % of the codebook and destroys the radar tokenizer ceiling; it fails the
design-§5 recon gate by 4–5×. w025 (higher weight) can only be worse. The
decoded-radar table is now *confirmatory*, not decisive. Recommended next
lever: drop index-wise cosine; try the **set-level / InfoNCE** alternative
(spec §3.1), which aligns the *code set* without forcing per-index direction
equality (the mechanism that drives the collapse). §7 below stays as the
auto-managed sweep tracker; if a confirmatory decoded run is wanted, the
launcher fix is in `scripts/launch_jointtok_sweep.sh` (`-f`→`-e`, prune-after-
decode, per-cell GPU re-check).

---

## 7. sim_weight sweep — reduced set, big budget

`w ∈ {0.0 separate, 0.05, 0.25}`, **AE 25k / v2v 40k** (the §6 8k pilot was visually unconverged — neg R², CSI35=0, low-freq blobs; those 8k A/B numbers stay in §4/§6 and are *not* mixed into this table). One-knob ablation, seed 42, shared batch order. **Ranking = the decoded-radar table 7b** — token-space loss is deliberately excluded (§4 collapse confound). Tokenizer-ceiling recon panels (pure encode→decode, no flow): `joint_tok_align/recon_{w000,w005,w025}.png` (also copied to each `joint_ae_sweep_*_run1/recon_test.png`) — the decoded radar in 7b can never beat that ceiling.

### 7a. AE trade-off + tokenizer ceiling

| sim_weight | x-modal cosine ↑ | linear CKA | PCA rank sat | PCA rank radar | sat recon MSE | radar recon MSE | sat recon Δ vs w0 |
|---|---|---|---|---|---|---|---|
| 0.00 | -0.0310 | 0.3874 | 53.1 | 26.7 | 0.000851 | 0.002250 | +0.0% |
| 0.05 | 0.9903 | 0.2043 | 34.8 | 24.9 | 0.001245 | 0.002771 | +46.3% |
| 0.25 | _pending_ | | | | | | |

### 7b. Decoded-radar pixel-space (DECISIVE — 2024/07 v2v test)

| sim_weight | mse_dbz ↓ | rmse_dbz ↓ | ssim ↑ | r² ↑ | avg_fss ↑ | csi35 ↑ |
|---|---|---|---|---|---|---|
| 0.00 | _pending_ | | | | | |
| 0.05 | _pending_ | | | | | |
| 0.25 | _pending_ | | | | | |

_Sweep running — 7b fills in per cell (~3.3 h/cell: AE 25k + v2v 40k + decode)._

