# Sat/Radar AE Loss-Recipe Sweep — Design Spec

**Date:** 2026-05-20
**Owner:** yghu
**Status:** Draft → Awaiting user review
**Lineage:** follows `2026-05-14-sat-ae-token-count-sweep-design.md` (capacity axis, settled at 128) → this spec sweeps the **loss recipe** at fixed `num_latent_tokens=128`.

## 1. Question

On Gadi, the FlowTiTok AE was trained from scratch three times (radar / sat-3ch / sat10ch each):

| | recipe | radar 200k MSE / SSIM / FSS | outcome |
|---|---|---|---|
| **run1** | `disc_start=400000` (GAN never fires), `perceptual_weight=1.1`, `perceptual_per_channel=false`, `kl_weight=1e-6`, `disc_weight=0.1`, `lecam=0.001`, l2, lr 1e-4 cos | **5.6e-4 / 0.952 / 0.705** | **best**, but (P1) visibly **blurry** (L2-mean-seeking, no GAN) and (P2) **sparse lightning channel structurally weak** (looks fine on full-set MSE — an empty-pixel artifact — but lgt-rich SSIM is only ~0.38, exposed by the overfit-lgt probe) |
| run2 | `disc_start 400k→60k`, `perceptual 1.1→0.01`, `kl 1e-6→1e-4`, all at once | 9.2e-3 / 0.65 / 0.10 (collapsed) | diverged: total loss went negative one logging step after GAN onset at 60k |
| run3 | `disc_start→100000`, `perceptual→1.0 + perceptual_per_channel:true`, `kl→1e-5`, `lecam→0.01`, `disc_lr→5e-5` | 1.4e-3 / 0.887 / **0.52** | less blurry but **hallucinates wrong detail** — radar FSS collapses 0.705→0.52, MSE ~2.5× worse; the 2026-05-14 study independently measured low-freq MSE +131–640%/band and Sobel edge IoU −23–31% |

**Goal:** find the loss-recipe hyperparameter combination that **matches or exceeds run1's reconstruction quality while reducing blur (P1) and fixing the sparse lightning channel (P2) — *without* introducing run3's hallucinated false detail.** Screen locally on lab2 at reduced budget; emit one Gadi-ready recipe.

### 1.1 Evidence-grounded causal read of run3 (drives the variant set)

Looking at run3 **at 50k, before its GAN engaged** (`disc_start=100000`): radar MSE 1.57e-3 vs run1-50k 8.7e-4, FSS 0.493 vs 0.582 — run3 was already ~1.8× worse **pre-GAN**, and did **not** degrade further after the GAN turned on at 100k (FSS drifted 0.50→0.52). Therefore the wrong-detail cause is **mostly not the GAN**. Ranked suspects:

1. **`kl_weight` 1e-6 → 1e-5 (×10)** — higher KL pulls the VAE posterior toward the prior; the decoder fills the information gap by inventing prior-plausible structure. Present from step 0 (consistent with run3 being bad at 50k). *The headline non-perceptual cause; promoted to a primary lever.*
2. **`perceptual_per_channel: true`** — independent LPIPS pressure per band (incl. the near-empty lightning channel) drives per-channel texture synthesis. Distinct mechanism from raw perceptual magnitude.
3. **The GAN** — run3's evidence actively *exonerates* a late gentle GAN (no post-100k degradation). Still tested explicitly in Stage 3, not assumed.
4. Coupled/minor: `lecam` only bites with GAN on; run2/run3 *resumed from checkpoint-198063* — a provenance confound this spec controls by training all local cells clean from scratch.

## 2. Scope

- **In scope:** local from-scratch AE screening for **radar (1ch)** and **sat10ch (11ch = 10 IR + lightning)**; loss-recipe sweep at fixed 128 tokens; produce one recommended Gadi recipe.
- **Out of scope:** sat-3ch (redundant with sat10ch for the lightning question); token-count axis (settled at 128 by the 2026-05-14 sweep); downstream sat→radar FlowTok effect; architecture variation (held at the 2026-05-14 regime for transfer validity). Each is a separate follow-on.

## 3. Variables

### 3.1 Held identical across all cells (the inherited, pre-validated regime)

| Knob | Setting | Source |
|---|---|---|
| Architecture | `vit_enc_model_size` tiny + `vit_dec_model_size` small (~37M) | 2026-05-14 §3 — pre-validated as recipe-sensitive (cleanly ranked tok77<128 and reproduced run1's blur signature) and fast (~2 h/cell) |
| `token_size` | 16 | run1 |
| `num_latent_tokens` | **128** (fixed) | 2026-05-14 result: 128 is the sweet spot; +capacity is itself a P1 de-blur lever |
| `kl_weight` baseline | **`6.0e-7` = `1e-6 × 77/128`** (per-token-normalized; KL is sum-over-(C×N), mean-over-batch — verified `quantizer.py:164-170`, `losses.py:407-408`) | 2026-05-14 §3 / Risks |
| Quantizer | `quantize_mode: vae` | run1 |
| Recon loss | `l2`, `reconstruction_weight=1.0` (Stage 2 overrides lgt channel only) | run1 |
| Perceptual | `lpips-convnext_s-1.0-0.1` | run1 |
| Discriminator | **OFF** for Stages 0–2 (`discriminator_start > max_train_steps`); ON only in Stage 3 | strategy decision: GAN deferred to confirm |
| Optimizer | adamw, lr 1e-4, betas (0.9,0.999), wd 1e-4, cosine, warmup 10% of horizon, end_lr 1e-5 | run1 / 2026-05-14 |
| Precision | `mixed_precision: no` (fp32) | run1 / 2026-05-14 |
| Batch | `per_gpu_batch_size=32`, grad_accum=1 (smoke-test 4090 fit; fallback 16, same across all cells) | 2026-05-14 §3 |
| Data | train: `71_3m` 2021/05–10 ct005 (~26k); val: 2024/06 w1; test: 2024/07 i2i (4391) | 2026-05-14 §3 |
| Preprocessing | `resize_shorter_edge=128`, `crop_size=128`, `random_crop=true`, `random_flip=true`, `res_ratio_filtering=true` | run1 |
| Init / seed | from scratch, `seed=42`, identical batch order (intentional for ablation cleanliness; controls the run3 resume confound) | this spec |
| Screening horizon | **40k steps** (user-chosen proxy; 2026-05-14 showed this regime converges by ~60k with no overfit, so 40k is sufficient for *ranking*) | user |
| Confirm horizon | **100k steps** (so a fraction-scaled late GAN actually engages) | this spec |

### 3.2 The swept axes

Baseline `B0` = run1 recipe in the §3.1 regime (perceptual 1.1 global, `perceptual_per_channel=false`, KL 6.0e-7, GAN off). All Stage-1 cells change **exactly one** knob vs `B0` (the explicit antidote to run2's all-at-once failure).

| Axis | Values (baseline **bold**) |
|---|---|
| `kl_weight` (×normalized base 6.0e-7) | 0.1× (6.0e-8) · **1× (6.0e-7)** · 10× (6.0e-6 ≈ run3's 1e-5 regime) |
| `perceptual_weight` (global) | 0.6 · **1.1** · 1.6 |
| `perceptual_per_channel` | **false** · true |
| lightning per-channel (sat10ch, ch 10) | **off** · 3× · 5× — `recon: l1`, `recon_weight ×k`, `perceptual_per_channel_weights[10]=k` (overfit-proven; 5× lifted lgt-rich SSIM +0.19 at +8.3% IR-MSE cost, 3× = milder) |
| late gentle GAN (Stage 3 only) | off · `disc_start≈0.55×horizon`, `disc_weight 0.005`, `disc_lr 5e-5`, `lecam 0.01` (run3's *only* correct ingredient) |

## 4. Experiment Matrix

### Stage 0 — Baselines (GAN off, 40k, run1 recipe @128)
- `B0-radar` — radar. **New.** Working baseline (radar).
- `B0-s10` — sat10ch. **= the validated 2026-05-14 `tok128` cell**: reuse its trained ckpt + metrics as the anchor; re-run only if a §3.1 knob differs from that cell (verify config hash — see Risks).
- Proxy validity is **pre-established** by the 2026-05-14 sweep (it reproduced run1's blur signature and ranked tok77<tok128 cleanly in this exact regime); no separate 77-token control is run.

### Stage 1 — Loss-recipe levers (GAN off, 40k, radar + sat10ch, each = B0 + one change)
| Cell | Change vs B0 | Probes |
|---|---|---|
| `S1-kl05` | `kl_weight ×0.1` → 6.0e-8 | KL-as-fidelity, faithful-latent direction |
| `S1-kl10` | `kl_weight ×10` → 6.0e-6 | reproduces run3 suspect #1 in isolation |
| `S1-pc` | `perceptual_per_channel: true` | run3 suspect #2 in isolation |
| `S1-p06` | `perceptual_weight 1.1→0.6` | bounded magnitude ↓ |
| `S1-p16` | `perceptual_weight 1.1→1.6` | bounded magnitude ↑ (find ceiling before false detail) |

5 cells × 2 modalities = **10 runs**.

### Stage 2 — Lightning lever (GAN off, 40k, **sat10ch only**, on best Stage-1 sat10ch recipe; fallback B0-s10)
- `S2-w3` — lgt ch10 → L1 + 3× recon & perceptual weight.
- `S2-w5` — lgt ch10 → L1 + 5× recon & perceptual weight.

**2 runs.**

### Stage 3 — Combine + late gentle GAN + confirm (100k, radar + sat10ch), paired GAN-free twins
Final candidate is built **per modality**: `S3-radar` = radar's own best Stage-1 recipe; `S3-s10` = sat10ch's own best Stage-1 recipe **+** the best Stage-2 lightning setting. (Stage-1 winners may differ between radar and sat10ch — each modality carries forward its own.)
- `S3-radar-gan`, `S3-s10-gan` — candidate + late gentle GAN (§3.2).
- `S3-radar-nogan`, `S3-s10-nogan` — identical recipe, GAN off, same 100k horizon → isolates the GAN's effect at the confirm horizon (explicit safety test; run3 predicts safe, we verify).

**4 runs.** Grand total ≈ **16–17 training runs** (`B0-s10` reused).

## 5. Compute & Orchestration

- **Platform:** lab2. GPU 0 free now; GPUs 1–3 occupied by other users' jobs. Strategy: launch on GPU 0, **opportunistically dispatch queued cells to GPUs 1–3** when `nvidia-smi` shows them free (memory below a threshold), never preempting other users; sequential fallback if only GPU 0.
- **Wallclock:** ~37M params, batch 32, 40k steps ≈ ~1.5–2 h/cell (radar cheaper than sat10ch); 100k confirm ≈ ~4–5 h/cell. Total ≈ 1–2 days depending on how many of GPUs 1–3 free up.
- **Output disk:** `/mnt/ssd_2/yghu/Experiments/ae_recipe_sweep/<cell>/` (ssd_1 near full per 2026-05-14 §8). Layout `<cell>/{ckpts/,samples/,output.log}`.
- **Conda env:** confirm in planning — Gadi AE scripts use `1d-tokenizer`; the 2026-05-14 local sweep used `flowtok`. Pin the verified one in the launcher.

## 6. Ckpt Policy

Per cell, 3 rolling slots (~150–250 MB each at ~37M fp32 + EMA + AdamW): `latest.ckpt` (every `save_every`, resume only), `best_val.ckpt` (atomic-rename on val-L2 improvement, 2024/06 w1, pure L2), `final.ckpt` (once at end). Headline = `best_val`; `final` secondary. ≈ 17 cells × 3 ≈ < 6 GB.

## 7. Evaluation Protocol (reuse the 2026-05-14 harness for cross-comparability)

For `best_val` + `final` of every cell:

1. **Standard recon** (`scripts/test_flowtitok_ae.py`, 2024/07 i2i test, 4391 frames): MSE/MAE per-ch+mean, PSNR, SSIM, LPIPS(alex), rFID, FSS (pysteps, thr 0–60/5, scale 1–10), CSI/POD/**FAR**.
2. **Freq/edge diagnostic** (the 2026-05-14 §6.2 port — the blur-vs-hallucination detector): low-freq MSE (σ=2), **high-freq MSE**, 8×8 patch-MSE heatmap, **Sobel edge IoU @0.1**, radial power spectrum.
3. **Latent utilization** (2026-05-14 §6.3): near-dead token count, KL/token, PCA effective rank — guards against recipe-induced latent collapse.
4. **Lightning-rich held-out subset** (new — the *only* honest P2 metric; full-set lgt MSE is an empty-pixel artifact): build a filelist of lgt-positive frames from 2024/07 test (larger analog of `configs/tiny_filelist_16samples_lgt_ae.pkl`); report lgt-channel SSIM / FSS / edge IoU there, plus a visual contact sheet (catch hallucination by eye).

Rolled up in `docs/specs/results/2026-05-20-ae-recipe-sweep-results.md`.

## 8. Decision Rule — fidelity-first, false-detail-guarded

Vs the matched **128 baseline** (`B0-radar` / `B0-s10`):

**Hard constraints (cell rejected if any violated; all evaluated beyond single-seed noise ≈ the 2026-05-14 ~15% floor, so noise alone never trips a constraint):**
1. avg MSE not worse beyond **+5%**.
2. **High-threshold FSS** (≥35 dBZ radar; upper IR & lgt bins sat10ch) **not decreased**.
3. **FAR not increased** (direct "reconstructed wrong detail" metric).
4. **Not the run3 hallucination signature**: Sobel edge IoU must not rise *while* high-freq MSE also rises (sharper-but-wrong edges) — and low-freq MSE must not regress.

**Then** rank surviving cells by targeted improvement:
- **P1 blur:** high-freq MSE ↓ / LPIPS ↓ / Sobel IoU ↑ — with FAR & high-thr FSS flat-or-better.
- **P2 lightning:** lgt-rich-subset SSIM / FSS / edge IoU ↑.

"Improved" threshold = **≥15% relative on the targeted metric (≥2 of its sub-metrics same sign)**, matching the 2026-05-14 single-seed-noise floor.

**The "best hyperparameter combination"** = the Stage-3 confirmed recipe that (a) passes **all** hard constraints on **both** modalities and (b) maximizes P1+P2 improvement. Reported as:
- a full **YAML diff vs run1**, and
- the **schedule rescaled to the Gadi regime** (200k steps / 250k examples): GAN-start by the same training-fraction, KL re-normalized for the Gadi token count (recommend 128 there too → `1e-6×77/128`; if Gadi keeps 77, `1e-6`).

If no cell passes all hard constraints on both modalities: report the closest per-modality recipes + the causal attribution and recommend the next experiment (do **not** ship a constraint-violating recipe).

## 9. Deliverable the user sees

1. One results table: all ~17 runs × (recon + freq/edge + lgt-rich) metrics × **hard-constraint pass/fail** vs baseline.
2. The causal attribution of run3's wrong detail — which of {`kl ×10`, `perceptual_per_channel`, gentle GAN} caused it — read directly from `S1-kl10` / `S1-pc` / the Stage-3 GAN-vs-noGAN twins.
3. **The single recommended Gadi-ready config** (YAML diff vs run1 + rescaled full-schedule values).

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Local tiny+small (~37M) recipe ranking may not transfer to Gadi base+large (~280M). | The 2026-05-14 sweep showed this regime *does* reproduce the run1-vs-run3 qualitative split (blur vs hallucination) — recipe-direction transfer is the validated use. Stage-3 winner is a *candidate* to confirm at full scale on Gadi, not a final claim. |
| Reusing the 2026-05-14 `tok128` cell as `B0-s10` is invalid if any §3.1 knob differs (e.g. its loss recipe, eval set, or kl normalization). | Before reuse, diff that cell's saved `config.yaml` against `B0-s10`'s intended config; re-run `B0-s10` if not byte-identical on the recipe + data + eval. |
| 40k < the ~60k convergence point seen in 2026-05-14 → ranking noisier. | 40k is for *ranking*, not final numbers; Stage-3 confirm runs 100k. Report best_val (converged-ish) not final. If Stage-1 ranking is within noise, extend the top-2 to 60k before Stage 2. |
| Per-token KL normalization wrong. | **Resolved** in 2026-05-14 Risks: KL = sum-over-(C×N) per sample, mean over batch (`quantizer.py:164-170`, `losses.py:407-408`); `kl_weight × 77/N` is correct. |
| `train_flowtitok_ae.py` may not read `num_latent_tokens` / per-channel / scaled `disc_start` at train time. | Verified `recon_loss_per_channel*` + `perceptual_per_channel*` + `discriminator_start` in `modeling/modules/losses.py`. Confirm `num_latent_tokens` plumbing + a 100-step smoke test per new config family before full launch. |
| batch 32 OOM on 4090 at 128px / 37M. | 100-step smoke test; fallback batch 16 applied **uniformly** to all cells (consistency > absolute value for relative ranking). |
| lgt-rich subset too small → noisy P2 read. | Target a few hundred lgt-positive frames (not 16 like the overfit probe); report n and per-frame variance. |
| Cross-device read (ssd_1 train) + write (ssd_2 ckpt). | `iostat` during smoke test; stage train shard to ssd_2 if I/O-bound (per 2026-05-14 §8). |
| Shared `seed=42` → cells not independent samples. | **Intentional** for ablation cleanliness (difference attributable to the one changed knob); documented so the reader does not read it as variance. |

## 11. Open Items Resolved by User

- Objective = **best AE overall**: maintain run1's quality *while* fixing P1 (blur) + P2 (sparse lightning) — not a trade-off. ✓
- Target modalities: **radar + sat10ch** (skip sat-3ch). ✓
- GPU: GPU 0 now, opportunistically grab 1–3 as they free. ✓
- Screening budget: **~40k steps/cell**. ✓
- Strategy: **A — one-lever-at-a-time ablation** (not factorial, not auto-HPO). ✓
- GAN: **deferred to the Stage-3 confirm** (screen GAN-free). ✓
- `num_latent_tokens`: **128** for all local cells. ✓
- run3 lesson: less blur but **wrong/false detail** — explicit fidelity guardrail (FAR, high-thr FSS, Sobel-vs-highfreq signature); and the non-perceptual causes (KL, per-channel-perceptual) decomposed as first-class Stage-1 levers. ✓
