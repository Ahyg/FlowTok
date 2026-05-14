# Sat AE Token-Count Sweep — Design Spec

**Date:** 2026-05-14
**Owner:** yghu
**Status:** Draft → Awaiting user review

## 1. Question

Is `num_latent_tokens=77` enough capacity for the 10ch sat AE (+lightning, so 11ch total) on the lab2-local 71_3m dataset, or does increasing it to 128 / 256 unlock meaningful reconstruction-quality gains?

Background: `sat10ch_…_run1_gadi` (base enc + large dec, 77 tokens, trained on the full 71 multi-year dataset on Gadi) reconstructs accurately on low-frequency content but is visibly blurred on high-frequency detail. `run3` (different loss recipe) is sharper but hallucinates structure (low-freq MSE +131-640% per IR band, Sobel edge IoU drops 23-31%). It is unclear whether the blur is a **token-budget** bottleneck or a **loss-recipe** bottleneck. This experiment isolates the token-budget axis.

## 2. Scope

- **In scope:** train 3 sat AEs (tok77, tok128, tok256), evaluate reconstruction on 2024/07 test, compare.
- **Out of scope:** downstream sat→radar (FlowTok) effect; loss-recipe variation; radar AE; per-token-dim variation. Each is a separate follow-on if this sweep produces a useful finding.

## 3. Variables

| Axis | Setting |
|---|---|
| **Varied** | `model.vq_model.num_latent_tokens ∈ {77, 128, 256}` |
| Architecture | `vit_enc_model_size: tiny`, `vit_dec_model_size: small` — total ~27M. (run1 was `base+large` ≈ 280M, so this is ~10× shrink, matched to the ~26k-frame train set.) The framework's named-size table is the source of truth: if "tiny" is not a registered size in `modeling/titok.py`, fall back to `small + small` and document the actual pair used. |
| `token_size` | 16 (per-token dim; fixed) |
| `in_channels` / `out_channels` | 11 / 11 (10 IR + 1 lightning; matches run1) |
| Loss recipe | run1's: `reconstruction_loss=l2`, `reconstruction_weight=1.0`, `perceptual_loss=lpips-convnext_s-1.0-0.1`, `perceptual_weight=1.1`, `lecam_regularization_weight=0.001`, `quantize_mode=vae`. **Discriminator disabled** (`discriminator_start > max_train_steps`, matches run1's effective behaviour). |
| **KL weight** | Normalized per-token: `kl_weight(N) = 1e-6 × (77 / N)` → `1e-6` for N=77, `6.02e-7` for N=128, `3.01e-7` for N=256. Reason: KL is summed over latent dims; without this, the larger-token cells are over-penalized. |
| Data | train: `71_3m` 2021/05-10 ct005-filtered (~26k frames); val: 2024/06 first week (val_small); test: 2024/07 (4391 frames). i2i (per-frame) mode. |
| Preprocessing | matches run1: `resize_shorter_edge=128`, `crop_size=128`, `random_crop=true`, `random_flip=true`, `res_ratio_filtering=true`. |
| Optimizer | adamw, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4, cosine schedule, `warmup_steps=6000` (10% of 60k), `end_lr=1e-5`. |
| Training | `max_train_steps=60000`, `per_gpu_batch_size=32` (down from run1's 64 because 4090 is smaller than Gadi V100; tune if VRAM allows higher), `gradient_accumulation_steps=1`, `mixed_precision='no'` (fp32, matches run1), `use_ema=true`, `seed=42`, `max_grad_norm=1.0`. |

**Held identical across cells:** every above setting except `num_latent_tokens` and the proportional `kl_weight`. Same data shards, same seed, same step count.

## 4. Compute

- **Platform:** lab2 (local), 4090×2 available (GPUs 0 and 2; GPUs 1 and 3 are occupied by other users' inference servers).
- **Orchestration:** launch tok77 on GPU 0 and tok128 on GPU 2 in parallel. tok256 starts on whichever GPU frees first. No DDP (1 GPU per cell).
- **Wallclock estimate:** ~12-18h per cell on single 4090 at 27M params, batch=32, 60k steps. Total: ~24-36h.
- **Output disk:** `/mnt/ssd_2/yghu/Experiments/sat10ch_ae_tok{77,128,256}_run1/` (avoid ssd_1 which is at 95% capacity).
- **Conda env:** `flowtok`.

## 5. Ckpt Policy (disk-budget critical)

Per cell, keep at most these three rolling slots — each ~500 MB (model fp32 + EMA fp32 + AdamW state):

| Slot | When written |
|---|---|
| `latest.ckpt` | Overwrites every `save_every=5000` steps. Resume use only. |
| `best_val.ckpt` | Overwrites when **val L2 recon loss** (on 2024/06 val_small, i2i, no perceptual/KL term) improves. Logged every 2000 steps. Pure L2 because LPIPS-weighted "best" is sensitive to perceptual scoring noise on a small val set. |
| `final.ckpt` | Written once at training end. |

Total: 3 cells × 3 slots × ~500 MB ≈ **4.5 GB**.

Evaluation reads `best_val.ckpt` for the headline number. `final.ckpt` is the secondary point for comparison (some models keep improving past best-val on train-distribution metrics but degrade on val; both are informative).

## 6. Evaluation Protocol

Run on 2024/07 i2i test set (4391 frames) for `best_val.ckpt` and `final.ckpt` of each cell, via the existing `scripts/test_flowtitok_ae.py`. For each ckpt produce a `metrics.json` and a `metrics.md` next to the ckpt.

### 6.1 Standard reconstruction metrics (existing)
- MSE / MAE (per channel + mean across the 11 channels)
- PSNR
- SSIM (per channel + mean)
- LPIPS (alex backbone)
- rFID

### 6.2 Frequency / spatial / edge diagnostic (port from `/tmp/diagnose_run1_vs_run3.py`)
- Low-freq MSE (Gaussian σ=2 blur, then MSE)
- High-freq MSE (residual after low-freq subtraction)
- 8×8 patch MSE map (visualize as heatmap)
- Sobel edge IoU at threshold 0.1 (binarize, IoU recon vs ground truth)
- Radial power spectrum comparison plot

Output: per-cell `diagnostic.md` with the 8 evenly-spaced sample comparison figures, plus a top-level `Code/FlowTok/docs/specs/results/2026-05-14-token-sweep-results.md` rolling up cell-by-cell.

### 6.3 Latent utilization (new, ~30 lines numpy)
On 256 random test samples per cell, compute the latent code statistics:
- Per-token activation variance across the batch → count "near-dead" tokens (variance < 1% of mean).
- Per-token mean KL divergence from prior.
- PCA effective rank of the flattened `[B, N×D]` latents.

The diagnostic question: does the 256-token cell actually *use* the extra tokens, or are most of them collapsed?

## 7. Decision Rule

After all three cells finish and 6.1-6.3 are computed:

Headline metrics for the decision: **high-freq MSE, Sobel edge IoU, LPIPS** — the three most sensitive to the blur failure mode of run1. Threshold for "improved": **≥15% relative change on the headline metric, with the same sign on at least 2 of 3**. Below 15% is treated as noise on a single-seed comparison.

| Pattern | Conclusion |
|---|---|
| **128 vs 77: improved (≥15%, 2/3 headlines); 256 vs 128: within ±15%** | 128 is the sweet spot. Adopt for follow-on AE training. |
| **256 vs 128: improved (≥15%, 2/3 headlines)** | Strong non-linearity; adopt 256, consider following up with 384. |
| **No cell improves over 77 by ≥15% on 2/3 headlines** | Token count is not the bottleneck. Loss recipe or architecture capacity is. Next experiment: vary loss recipe at fixed N=77. |
| **128/256 cells have ≥30% near-dead tokens (§6.3)** | Capacity is unused regardless of N. Same conclusion as the previous row: not a token-count problem. |
| **Headline metrics disagree (e.g., LPIPS improves but Sobel IoU regresses)** | Report and discuss; defer follow-on decision until a downstream FlowTok pilot at the leading candidate. |

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `vit_enc_model_size=small` and `=base` already exist in `modeling/`; a true "tiny" may not. If "tiny" isn't a registered size, fall back to `small + small`. | Inspect `modeling/titok.py` (and the size-registry in TA-TiTok) before writing configs. Pick the smallest pair that matches the ~27M target; record the actual pair used in the cell config and in this spec's §3 row. |
| ~~Per-token KL normalization is wrong if KL is mean-over-tokens~~ | **Resolved.** Verified in `modeling/quantizer/quantizer.py:164-170` (`torch.sum(..., dim=[1,2])` over `[B,C,N]`) and `modeling/modules/losses.py:407-408` (sum-over-batch then divide by B). KL is **sum-over-(C×N) per sample, mean over batch**. §3 normalization `kl_weight × 77/N` is correct. |
| `per_gpu_batch_size=32` on 4090 may OOM if attention sequence length 256+small-dec is heavier than expected. | Smoke-test 100 steps per cell before kicking off full run. If OOM, drop to batch=16 (same across cells). |
| Reading from `/mnt/ssd_1` while writing checkpoints to `/mnt/ssd_2` crosses devices — verify dataloader throughput isn't bottlenecked. | Run `iostat` during the 100-step smoke test. If I/O is the bottleneck, stage the 71_3m train shard to `/mnt/ssd_2` first. |
| `best_val.ckpt` rolling overwrite races if val happens during ckpt write. | Use atomic rename (write `tmp_best_val.ckpt` then `os.replace`). Standard pattern in `train_flowtitok_ae.py`; verify it's there. |
| Three runs share `seed=42` → identical batch ordering means cells aren't independent samples. | This is *intentional* for ablation cleanliness — we want the difference to be attributable to the only changed knob. Document this in the results write-up so the reader doesn't confuse it for hidden variance. |

## 9. Deliverables

1. Three training configs: `configs/sat10ch_ae_tok{77,128,256}_lab2.yaml`.
2. One launch script: `scripts/launch_sat_ae_token_sweep_lab2.sh` — handles GPU 0+2 parallel scheduling and tok256 hand-off.
3. One eval script: `scripts/eval_sat_ae_token_sweep.sh` — runs §6.1 + §6.2 + §6.3 over all six ckpts (3 cells × {best_val, final}).
4. One diagnostic write-up: `docs/specs/results/2026-05-14-token-sweep-results.md`.

## 10. Open Items Resolved by User

- AE-only judge (no downstream FlowTok in this sweep). ✓
- 3 cells, all retrained from scratch, no run1 reuse. ✓
- run1 loss recipe (low KL, no perceptual_per_channel, LPIPS-only, no discriminator). ✓
- Architecture: tiny + small ~27M (substitute `small + small` if framework lacks "tiny"). ✓
- Local lab2, 2 GPUs available, no DDP. ✓
- 60k steps, cosine, 6k warmup. ✓
- Output disk → ssd_2. ✓
- Ckpt rolling slots (latest / best_val / final). ✓
- per-token KL normalization (kl_weight × 77/N). ✓
