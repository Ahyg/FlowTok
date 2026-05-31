# AE Loss-Recipe Sweep — Results

Design: `docs/specs/2026-05-20-ae-loss-recipe-sweep-design.md`
Plan / orchestrator: `scripts/run_recipe_sweep_queue.sh`
Eval harness: `scripts/eval_recipe_sweep.sh` (reuses `scripts/test_flowtitok_ae.py`)
Rollup: `Experiments/ae_recipe_sweep/rollup_eval_metrics.py` (runtime, not committed)

**Scope:** 21 GAN-free 40k-step screening cells (10 radar 1ch + 11 sat 11ch) at the FlowTiTok tiny-enc/small-dec / token_size=16 / num_latent_tokens=128 starting point, plus 3 sat token-count cells (sat10ch_ae_tok{77,128,256}_run1; 60k steps, separate dir). All test metrics on the 2024/07 held-out i2i test set (4391 samples).

Eval ran clean 21/21, 0 failed, finished 2026-05-22 02:34.

---

## TL;DR

1. **`patch_size=16 → 8` is the single biggest win on BOTH modalities.** Radar mse **−25.7%**, fss 0.50 → 0.59; sat fss_excl_lgt 0.76 → **0.84**. *The one architectural change to ship.*
2. **`kl_weight` baseline is too high. ÷10 wins (`kl05`), ×10 catastrophic (`kl10`)** — for sat10, +67% mse. `kl05` is the lowest weight swept; **not at floor** — a `÷100/÷1000` probe is warranted before locking the recipe.
3. **`perceptual_weight` preferences are OPPOSITE across modalities.** Radar wants LOW (0.6 helps, 1.6 hurts +10%); sat wants HIGH (1.6 best rfid; 0.6 hurts +16%).
4. **`perceptual_per_channel=true` for sat is a pixel-vs-perceptual TRADEOFF, not a loss.** Pixel mse +22.5%, but rfid_excl_lgt 173 → 54, lpips 0.28 → 0.11, fss 0.76 → 0.83 — best perceptual quality in the sweep. Choose based on downstream objective.
5. **Lightning is NOT the per_channel mse culprit — IR-band texture synthesis is.** Removing lightning from per_channel (`s10_s1_pc_nolgt`) made mse WORSE, not better. Boosting lightning *recon* weight (`s10_lgt10/20`) is a dead lever — ch10_ssim ≈ 0.995 is already saturated, extra weight only steals capacity from IR.
6. **`num_latent_tokens=128` is the sweet spot, both modalities** (radar test: 64/77/256 all worse than 128; sat token sweep: 128 ≈ 256 ≫ 77).

---

## Recommended Recipes (apply to your bigger AE / bigger dataset)

> **All recipes verified one lever at a time vs baseline. Combining wins has not been jointly tested.** Expect some redundancy / capacity competition between levers — verify on a small budget before committing to a long run. Numbers in `★` lines flag changes from `radar_b0` / `s10_b0`.

### Radar

```yaml
model:
  vq_model:
    quantize_mode: vae
    token_size: 16
    vit_enc_model_size: tiny
    vit_dec_model_size: small
    vit_enc_patch_size: 8         # ★ was 16  (mse −25.7%, fss 0.50→0.59, rfid 24.6→15.6)
    vit_dec_patch_size: 8         # ★ was 16
    num_latent_tokens: 128        # verified sweet spot (64/77/256 all worse)
    in_channels: 1
    out_channels: 1
losses:
  kl_weight: 6.02e-08             # ★ was 6.02e-7 (÷10; mse −11.5%) — may not be at floor
  perceptual_weight: 0.6          # ★ was 1.1  (modest −5.8%, close to single-seed noise)
  reconstruction_loss: l2
  reconstruction_weight: 1.0
  perceptual_per_channel: false   #   no-op for 1ch
```

### Sat10 — variant A: optimize pixel MSE / overall accuracy

```yaml
model:
  vq_model:
    vit_enc_patch_size: 8         # ★ fss_xlgt 0.76→0.84, rfid 114→107
    vit_dec_patch_size: 8         # ★
    num_latent_tokens: 128        # sweet spot (60k sat-token sweep confirms)
    in_channels: 11
    out_channels: 11
losses:
  kl_weight: 6.02e-08             # ★ was 6.02e-7  (÷10; mse −6.4%)
  perceptual_weight: 1.6          # ★ was 1.1  (rfid 114→102 best, mse −2.5%)
  perceptual_per_channel: false   #   off — per_channel hurts pixel mse
training:
  per_gpu_batch_size: 32
```

### Sat10 — variant B: optimize perceptual / structural fidelity (for downstream flow-matching, FID/CSI-style objectives)

```yaml
model:
  vq_model:
    vit_enc_patch_size: 8         # ★
    vit_dec_patch_size: 8         # ★
    num_latent_tokens: 128
    in_channels: 11
    out_channels: 11
losses:
  kl_weight: 6.02e-08             # ★
  perceptual_weight: 1.1          # or 1.6
  perceptual_per_channel: true                                          # ★ rfid 173→54, lpips 0.28→0.11, fss 0.76→0.83
  perceptual_per_channel_weights: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]     # ★ keep lightning ch10 wt=1; setting it to 0 made mse worse, not better
training:
  per_gpu_batch_size: 8           # ★ per_channel × 11ch OOMs at 32 and 16 on 24GB
```

**Pick A or B by downstream goal:**

- field-MSE / r² / numerical accuracy → **A**
- tokenizer for flow-matching / rFID / CSI-style structure → **B**

---

## Full Tables

### Radar — 10 cells, baseline `radar_b0`

| cell | mse | Δmse | ssim | psnr | lpips | rfid | fss |
|---|---|---|---|---|---|---|---|
| radar_b0 | 0.001739 | +0.0% | 0.864 | 36.49 | 0.060 | 24.6 | 0.503 |
| radar_s1_kl05 (kl÷10) | 0.001538 | **−11.5%** | 0.877 | 37.08 | 0.058 | 24.7 | 0.525 |
| radar_s1_kl10 (kl×10) | 0.001924 | +10.6% | 0.855 | 36.30 | 0.067 | 23.9 | 0.478 |
| radar_s1_p06 (perc 0.6) | 0.001638 | −5.8% | 0.870 | 36.96 | 0.058 | 25.2 | 0.522 |
| radar_s1_p16 (perc 1.6) | 0.001912 | +10.0% | 0.858 | 36.28 | 0.064 | 27.8 | 0.469 |
| radar_s1_pc (per_channel) | 0.001665 | −4.3% | 0.868 | 36.74 | 0.059 | 24.2 | 0.525 |
| **radar_b0_p8 (patch=8)** | **0.001291** | **−25.7%** | **0.892** | **37.94** | **0.035** | **15.6** | **0.593** |
| radar_tok64 | 0.001956 | +12.5% | 0.855 | 36.31 | 0.060 | 25.8 | 0.495 |
| radar_tok77 | 0.001987 | +14.3% | 0.855 | 36.12 | 0.062 | 24.6 | 0.492 |
| radar_tok256 | 0.001860 | +7.0% | 0.860 | 36.17 | 0.061 | 24.2 | 0.494 |

### Sat10ch — 11 cells

baseline `s10_b0` (batch=32); per_channel and lgt cells with batch=8 compare against `s10_b0_b8` for clean single-lever attribution.

| cell | mse | mse_xlgt | fss_xlgt | rfid_xlgt | lpips_xlgt | ch10_mse | ch10_ssim |
|---|---|---|---|---|---|---|---|
| s10_b0 | 0.000607 | 0.000667 | 0.763 | 114.0 | 0.177 | 2.92e-6 | 0.9951 |
| s10_b0_b8 (batch=8 ctrl) | 0.000768 | 0.000844 | 0.761 | 172.7 | 0.277 | 2.97e-6 | 0.9948 |
| **s10_b0_p8 (patch=8)** | **0.000591** | **0.000650** | **0.838** | **107.0** | **0.159** | 2.92e-6 | 0.9951 |
| s10_s1_kl05 (kl÷10) | **0.000568** | 0.000624 | 0.771 | 107.6 | 0.175 | 2.94e-6 | 0.9950 |
| s10_s1_kl10 (kl×10) | 0.001015 | 0.001116 | 0.753 | 173.0 | 0.270 | 2.92e-6 | 0.9951 |
| s10_s1_p06 (perc 0.6) | 0.000705 | 0.000775 | 0.817 | 128.4 | 0.208 | 2.97e-6 | 0.9949 |
| s10_s1_p16 (perc 1.6) | 0.000592 | 0.000651 | 0.763 | **101.8** | 0.185 | 2.91e-6 | 0.9951 |
| s10_s1_pc (per_channel, b=8) | 0.000941 | 0.001035 | **0.833** | **54.2** | **0.111** | **2.48e-6** | **0.9992** |
| s10_s1_pc_nolgt (b=8, lgt wt=0) | 0.001094 | 0.001203 | 0.835 | 47.9 | 0.128 | 2.88e-6 | 0.9954 |
| s10_lgt10 (lgt recon ×10) | 0.000714 | 0.000785 | 0.759 | 122.6 | 0.212 | 2.93e-6 | 0.9949 |
| s10_lgt20 (lgt recon ×20) | 0.000721 | 0.000793 | 0.767 | 122.0 | 0.215 | 2.91e-6 | 0.9950 |

### Sat token sweep — 3 cells (separate dir, 60k steps)

**Not directly comparable to the 40k recipe cells**; usable for directional token-count signal only.

| cell | mse | mse_xlgt | fss_xlgt | rfid_xlgt | lpips_xlgt |
|---|---|---|---|---|---|
| sat10ch_ae_tok77_run1 | 0.000750 | 0.000825 | 0.820 | 137.3 | 0.209 |
| sat10ch_ae_tok128_run1 | 0.000581 | 0.000638 | 0.763 | 107.4 | 0.169 |
| sat10ch_ae_tok256_run1 | 0.000571 | 0.000628 | 0.765 | 98.1 | 0.173 |

128 vs 256: −1.7% mse, marginal. 77 → 128: −22.5% mse. Conclusion: 128 is the elbow; doubling tokens to 256 is no gain.

---

## Findings (mechanism / interpretation)

### F1. patch=8 is universally the biggest win

Halving patch size pushes the ViT input sequence from `(128/16)² = 64` to `(128/8)² = 256` patches; `num_latent_tokens` stays 128 because TiTok uses learned query tokens that *summarize* the patch sequence, not direct pixel tokens. The encoder/decoder gets ~4× more spatial detail to compress while the bottleneck stays the same — net gain in pixel and structural metrics on both modalities. Expected cost: ViT compute/memory ~4×; observed: `s10_b0_p8` ran at batch=32 without OOM at 24 GB single-GPU, so on Gadi H100/A100 it's a free lunch.

### F2. KL is too high; ÷10 wins and floor is not located

Both radar (−11.5%) and sat (−6.4%) prefer `kl_weight ÷ 10`. At `× 10`, sat collapses (+67% mse) — a strong sign the baseline 6.02e-7 already sits on the steep side. `kl05` is the lowest swept; no information whether ÷100 / ÷1000 keep winning or eventually hurt regularization. **Add a `kl_w007` (6.02e-9) probe before locking the recipe.**

### F3. Perceptual weight preferences are opposite

| | perc 0.6 | perc 1.1 (baseline) | perc 1.6 |
|---|---|---|---|
| radar mse | **−5.8%** ✓ | — | +10.0% ✗ |
| sat mse | +16% ✗ | — | **−2.5%** ✓ |

Most likely explanation: radar is single-channel, sparse-structure, low-frequency-dominated — a heavy perceptual loss (LPIPS+ConvNeXt trained on natural images) over-textures it. Sat10ch has rich texture across IR bands; the same perceptual loss helps it organize high-frequency content. **Per-modality perceptual tuning required — do not share `perceptual_weight` across configs.**

### F4. per_channel is a tradeoff, not a loss

Comparison vs matched-batch baseline `s10_b0_b8`:

| | mse | rfid_xlgt | lpips_xlgt | fss_xlgt | ch10_ssim |
|---|---|---|---|---|---|
| s10_b0_b8 | 0.000768 | 172.7 | 0.277 | 0.761 | 0.9948 |
| s10_s1_pc | 0.000941 (+22.5%) | **54.2** | **0.111** | **0.833** | **0.9992** |

Pixel MSE worsens, but every perceptual / structural metric improves massively, and the lightning channel improves SSIM 0.995 → 0.999. Mechanism: per_channel LPIPS runs on each channel independently (channel-split path in `modeling/modules/perceptual_loss.py`), so the network is forced to *synthesize plausible texture per channel* rather than blending channels in a way that hurts perceptual but helps pixel error. **For a tokenizer feeding flow-matching, this is the better setting — perceptual ceiling > pixel exactness.**

### F5. The lightning question — resolved

The hypothesis was: per_channel hurts sat10 mse because the sparse lightning channel (ch10) is OOD for ImageNet-trained LPIPS. The diagnostic `s10_s1_pc_nolgt` (per_channel with `perceptual_per_channel_weights[10]=0`) tests it.

| | mse | ch10_ssim |
|---|---|---|
| s10_b0_b8 (no per_channel) | 0.000768 | 0.9948 |
| s10_s1_pc (per_channel all 11) | 0.000941 | **0.9992** |
| s10_s1_pc_nolgt (lgt wt=0) | **0.001094** (worse than pc) | 0.9954 (≈ baseline) |

**Conclusions:**

- Removing lightning made overall mse WORSE — so lightning was not the culprit.
- The pixel-mse degradation comes from **IR-band texture synthesis** (10 IR channels each getting independent perceptual gradients → more aggressive per-band texture generation).
- Giving lightning an independent perceptual signal IMPROVES ch10 ssim 0.9951 → 0.9992. So *if running per_channel, keep lightning weight = 1*, do not zero it.

Boosting lightning *recon* weight (`s10_lgt10` = ×10, `s10_lgt20` = ×20) did **nothing** for ch10 (mse 2.92e-6 → 2.91~2.93e-6, ssim 0.9951 → 0.9949~0.9950) and slightly hurt overall mse. Lightning is sparse, mostly zero, *already trivially reconstructed* — extra weight only steals capacity from IR. **Dead lever, do not use.**

### F6. Token count: 128 is the elbow

Radar test (single-lever vs `radar_b0=128`): tok64 +12.5%, tok77 +14.3%, tok256 +7.0% — all worse, monotone-ish convex around 128. Sat token sweep (60k, separate dir): tok77 → 128 mse −22.5%, 128 → 256 −1.7% (within noise). Both modalities point to 128 as the capacity elbow for the current tiny/small + token_size=16 architecture. **Re-check this when scaling up enc/dec size** — the elbow likely moves.

---

## Cross-checks & caveats (apply weight to recommendations accordingly)

1. **Single seed**. Radar noise floor ≈ 5% from prior runs (see `radar_s1_pc` at −4.3% which is structurally a no-op for 1ch). Any "win" under ±5% on radar mse is suspect. Patch=8 (−25.7%), kl05 (−11.5%) clearly beat the floor; perceptual=0.6 (−5.8%) is borderline.
2. **One-lever ablation**. Combining patch=8 + kl05 + perceptual-tuned + per_channel has not been jointly tested. Levers compete for capacity — linear extrapolation will likely *over-estimate* combined gains. Run a 1–2 cell combo verification on a small budget before scaling.
3. **kl05 may not be at floor** (F2). If you have one extra slot, probe `kl_weight=6.02e-9` (÷100) on both modalities.
4. **Sat token sweep is 60k vs 40k recipe** — cross-comparable only directionally on token count; absolute numbers not comparable to recipe-sweep sat10 cells.
5. **patch=8 ViT cost ~4×**. Single-GPU 24 GB ran sat10 batch=32 patch=8 cleanly. With per_channel (variant B) the joint memory cost has not been measured — retest at H100/A100; batch=8 is the safe floor.
6. **Test set is i2i 2024/07 only** — held out from training-period (2021 Jul/Aug/Oct training). Held-out month, but a single month — seasonality on a different month not tested. (Memory: `dataset_split_season_match`.)

---

## Recommended next moves

- **Ship patch=8 + kl05 to the next baseline** — both clear wins, both modalities, both well above noise.
- **Per-modality perceptual tuning**: radar 0.6, sat 1.6.
- **One combo-verification cell per modality** (e.g. `radar_p8_kl05_p06`, `s10_p8_kl05_p16`) before the long bigger-data run, to flag any non-additive interaction.
- **One `kl_w007` probe** (÷100) on each modality.
- **For sat-only**: choose variant A (pixel) or B (perceptual) by downstream objective. If the AE feeds flow-matching for nowcasting, B's massive rfid/lpips/fss improvement is likely the right pick despite +22.5% pixel mse.
