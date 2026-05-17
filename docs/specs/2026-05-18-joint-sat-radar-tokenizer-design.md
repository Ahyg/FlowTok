# Joint Sat/Radar Tokenizer Training — Design + Critical Analysis

**Date:** 2026-05-18
**Owner:** yghu
**Status:** Autonomous overnight pilot (user asleep — decisions made + flagged for morning review)

## 0. TL;DR for the morning

Your idea is **sound, but the framing needs one correction** and the expected payoff is narrower than "helps the flow learn." Read §2 (analysis) — it changes how to interpret results. Everything was scaled to a single 4090 + the small lab2 dataset and run as a **staged pilot** (tokenizer phase = the real scientific result; v2v phase = a short downstream sanity check, explicitly not a converged model). Results: see `docs/specs/results/2026-05-18-joint-tokenizer-results.md`.

## 1. Your hypothesis (restated)

Jointly training the sat tokenizer/detokenizer and the radar tokenizer/detokenizer with a similarity loss between sat tokens and radar tokens makes them learn the **same positional correspondence**, which helps the downstream FlowTok v2v (sat→radar).

## 2. Critical analysis — is this sound?

**Verdict: the mechanism is real and worth testing, but "positional correspondence" is the wrong mental model, and the realistic benefit is *optimization/conditioning*, not *added predictive power*.**

1. **The tokens are NOT spatial.** FlowTiTok uses 1D *unordered learned-query* latent tokens (`libs/flowtitok.py:79-82`, `nn.Parameter(randn(num_latent_tokens, width))`). Token index *k* has **no pixel/spatial meaning**, and for two *independently* trained tokenizers the per-index basis is **arbitrary and mutually unrelated**. So "same positional correspondence" does not exist to be learned in the spatial sense.

2. **Correct reframing: shared latent *code* alignment.** What the similarity loss actually does — when applied to *time-paired* sat & radar frames — is force the two tokenizers to adopt a **common latent layout**: token *k* encodes "the same scene factor" in both modalities. The flow (sat-token-seq → radar-token-seq) then becomes a **smaller, closer-to-identity per-index displacement** instead of a map between two arbitrary unrelated spaces. With a small DiT and little data, an easier target geometry can genuinely help.

3. **Key caveat — alignment ≠ information.** Sat (cloud-top IR + lightning) and radar (surface precip reflectivity) carry *different physical content*. Aligning latent spaces injects **no precipitation information the sat input lacks**. Sat→radar is fundamentally a *prediction* problem. So the honest upside is "the flow trains faster / better-conditioned / needs less data," **not** "fundamentally better forecasts." Measure both, attribute correctly.

4. **Risk — modality conflict vs reconstruction.** A too-strong similarity weight fights each tokenizer's own reconstruction (forcing radar tokens to look like sat tokens destroys radar detail). This is the central tension. The similarity weight is **the one knob**; reconstruction of *both* modalities is monitored as the guardrail.

5. **Pairing requirement.** The similarity loss needs **same-timestamp paired sat+radar frames** → the joint AE trains on the paired v2v filelist (single-frame mode), not the sat-only i2i set.

**Conclusion:** proceed, but the experiment's primary question is reframed to: *Does joint training measurably align the two latent spaces, and at what reconstruction cost? Then, does that alignment translate into a better-conditioned / faster / better v2v flow?*

## 3. Experiment design

**Clean ablation via one knob.** A single training script trains a sat AE + radar AE on the *same* paired data, *same* arch/seed/steps/order. `sim_weight = 0` ⇒ Group A (separate baseline). `sim_weight > 0` ⇒ Group B (joint). This removes every confound except the similarity loss — the cleanest possible A/B (param groups are disjoint; with `sim_weight=0` no gradient crosses modalities, so it is equivalent to independent training but with identical data/order).

| | Group A (separate) | Group B (joint) |
|---|---|---|
| Sat AE | tiny enc + small dec, 11ch, 77 tok | same |
| Radar AE | tiny enc + small dec, 1ch, 77 tok | same |
| Similarity loss | **off** (`sim_weight=0`) | **on** (`sim_weight=0.5`, index-wise cosine) |
| Data / seed / steps | paired v2v filelist, single-frame, seed 42, 15k | identical |
| Downstream | FlowTok-S v2v, frozen A tokenizers, 8k | FlowTok-S v2v, frozen B tokenizers, 8k |

### 3.1 Similarity loss (the net-new detail)

For a paired (sat_frame, radar_frame) at the same timestamp, encode both, take the **posterior means** S, R ∈ ℝ^[B,N,D] (pre-sampling, for stability). Primary:

```
L_sim = mean over tokens k of  (1 − cos( S[:,k,:], R[:,k,:] ))      # index-wise cosine
L_total = L_ae_sat + L_ae_radar + sim_weight · L_sim
```

`L_ae_*` = run1 recipe (L2 recon w=1.0, LPIPS-convnext_s w=1.1, KL w=1e-6, no discriminator). Index-wise (not pooled) because the flow operates per-token-index — aligning the exact axis the flow uses. Reconstruction terms prevent representational collapse (decoders must still reconstruct both modalities).

**Alternatives considered (documented, not used in pilot):** (b) per-index InfoNCE across the batch (anti-collapse, more knobs); (c) global pooled CLIP-style InfoNCE like `diffusion/flow_matching.py:257-274` (aligns global content, *not* per-index — weaker for this flow). The existing "contrastive" flavor is **not** this: it aligns textVAE-encoded radar vs projected sat *inside v2v flow training*, with frozen tokenizers — orthogonal to this experiment.

### 3.2 Scaling for lab2 small dataset (explicitly requested)

- AE: `vit_enc=tiny, vit_dec=small`, token_size 16, 77 tokens (~37M, reused from the just-finished token sweep — proven to train cleanly on this data).
- v2v DiT: **FlowTok-S** (~30M, the existing `Sat2Radar-v2v-uni-...-FlowTiTok-S.py`), frozen tokenizers, 16-frame clips batch 2.
- Steps cut for an overnight pilot: AE 15k (token-sweep best-val was effectively reached well before 60k on this data), v2v 8k (**pilot — not converged; reads as a conditioning/convergence signal + rough recon, not a final model**).

## 4. Compute & schedule (single GPU 0; 1/2/3 held by other users)

Sequential on GPU 0, started by an orchestrator that **waits for the token-sweep eval to free GPU 0**, then:

1. A: sat AE 15k (~50 min) → 2. A: radar AE 15k (~50 min) → 3. B: joint AE 15k (~90 min)
4. **AE-phase analysis written immediately** (this is the primary result; survives even if v2v doesn't finish)
5. A: v2v-S 8k (~90 min) → 6. B: v2v-S 8k (~90 min) → 7. final analysis + results doc

Total ≈ 7–8 h after GPU frees. Runs in **tmux session `joint_tok`** (survives disconnect). Output → `/mnt/ssd_2/yghu/Experiments/` (ssd_1 at 95%). 3-slot rolling ckpts (latest/best_val/final), ~same disk scheme as token sweep.

## 5. Evaluation

**Tokenizer phase (primary):**
- Reconstruction (A vs B, both modalities): MSE/PSNR/SSIM/LPIPS on 2024/07 test single-frame. *Did joint training cost reconstruction?*
- Latent alignment (A vs B): on paired test frames — mean index-wise cosine(S_k,R_k); centered-kernel-alignment (CKA) between sat & radar latents; error of a per-index linear sat→radar fit (lower ⇒ flow target closer to conditioning). *Did joint training actually align the spaces?*
- Latent utilization (reuse `scripts/latent_utilization.py`): near-dead token count, PCA effective rank — does alignment collapse capacity?

**v2v phase (secondary, pilot):**
- Flow training loss curve A vs B (conditioning/convergence speed at equal steps).
- Radar reconstruction from predicted tokens on val_small: MSE/SSIM + radar CSI@20/35 dBZ if cheap.

**Decision rule:** Joint training "helps the tokenizers" if B raises cross-modal token cosine / CKA by a clear margin (≥0.1 abs cosine or ≥0.15 CKA) **with ≤10% relative reconstruction degradation on both modalities**. It "helps the flow" only if, additionally, B's v2v reaches the same loss in fewer steps OR a lower val radar MSE at 8k. Disagreement (alignment up but recon or flow worse) ⇒ report the trade-off; the right follow-up is a similarity-weight sweep, not a verdict.

## 6. Deliverables

1. `scripts/train_joint_sat_radar_ae.py` — net-new joint AE trainer (sim_weight knob).
2. `configs/joint_ae_sep_lab2.yaml`, `configs/joint_ae_joint_lab2.yaml` — Group A / B AE configs.
3. `configs/Sat2Radar-v2v-jointtok-{A,B}-FlowTiTok-S.py` — v2v configs pointing at A/B tokenizers (AE arch overridden to tiny+small to match — guards against the strict=False silent-mismatch trap).
4. `scripts/launch_joint_tok_experiment.sh` — staged tmux orchestrator (waits for GPU 0, runs 1–7, writes results).
5. `scripts/eval_joint_tokenizer.py` — alignment metrics (cosine/CKA/linear-fit) A vs B.
6. `docs/specs/results/2026-05-18-joint-tokenizer-results.md` — analyzed write-up (the morning deliverable).

## 7. Assumptions made autonomously (flag for morning review)

- Reframed objective from "spatial position" to "shared latent-code alignment" (§2) — **the key conceptual change; please confirm this matches intent.**
- `sim_weight=0.5`, index-wise cosine, on posterior means — single defensible default; not swept (one night). If alignment shows up but recon degrades, the obvious next run is a 0.1/0.25/0.5/1.0 weight sweep.
- 77 tokens (matches FlowTok-S `num_clip_token=77`; avoids touching the DiT). Token-sweep hinted 128>77 for sat recon — orthogonal axis, deferred.
- v2v at 8k steps is a **pilot signal**, not a converged comparison. Loss-curve/convergence is the trustworthy v2v read; absolute forecast quality is not.
- AE 15k steps assumed sufficient for a *comparative* pilot on this small dataset (token-sweep evidence). If A/B both look under-trained, rerun longer.
