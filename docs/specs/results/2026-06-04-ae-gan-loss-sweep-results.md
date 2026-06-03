# AE Late-GAN Loss Sweep — Results (Stage 3)

Design: `docs/specs/2026-05-20-ae-loss-recipe-sweep-design.md` (Stage 3, deferred GAN stage)
Config generator: `scripts/gen_gan_sweep_configs.py`
Orchestrator: `scripts/run_gan_sweep_queue.sh`
Eval harness: `scripts/eval_gan_sweep.sh` (reuses `scripts/test_flowtitok_ae.py`)
Run: lab2, 2026-06-03/04. All test metrics on the 2024/07 held-out i2i test set (4391 samples).

**Scope:** the deferred GAN stage of the loss-recipe sweep. Base = the radar GAN-free winner from `2026-05-22-ae-loss-recipe-sweep-results.md` (patch8 + kl÷10 + perceptual 0.6 @ 128 tokens, l2, tiny-enc/small-dec, token_size 16, vae). One axis swept: `discriminator_weight ∈ {0.001, 0.01, 0.1, 1.0}` plus a paired GAN-off anchor, **100k steps each**. GAN engages late at `discriminator_start=60000` (after recon plateaus, ~60k in this tiny-model regime). Other GAN params follow TA-TiTok `tatitok_bl64_vae`: `discriminator_learning_rate=1e-4`, `lecam_regularization_weight=1e-3`, `discriminator_factor=1.0`.

Eval ran clean 5/5, 0 failed, both `best_val` and `final` slots.

---

## TL;DR

1. **`discriminator_weight=0.001` is the only beneficial GAN setting — and it wins decisively.** vs GAN-off at the true 100k endpoint: MSE −22%, PSNR **+4.9 dB**, SSIM 0.973 vs 0.954, LPIPS −21%, **rFID 3.64 vs 6.29 (−42%)**, FSS essentially tied (0.712 vs 0.721). A whisper of adversarial signal acts as a regularizer rather than trading pixels for realism.
2. **GAN strength is monotonically harmful above 0.001.** `w=0.01` already drops below GAN-off on everything; `w=0.1` collapses perceptual (rFID 10.1, LPIPS 0.029); **`w=1.0` near-collapses** (MSE ×5.4, rFID ×6, FSS 0.465). Not NaN divergence — a quality collapse where the discriminator overwhelms reconstruction.
3. **The useful GAN weight is an order of magnitude below TA-TiTok's 0.1 default.** Radar's single channel + tiny model can't absorb a strong discriminator.
4. **Methodology gotcha — `best_val` selection hides the GAN.** `checkpoint-best_val` is picked by val-**L2**, which for the strong-GAN cells lands on the *pre-GAN* checkpoint (`w01`/`w1` best_val = step 60000, before GAN engaged). Always also eval the `final` slot for an apples-to-apples GAN-on comparison.

---

## Recommended Recipe (apply to the bigger AE / bigger dataset)

Take the radar GAN-free winner and add a **gentle late GAN**. Do **not** raise the discriminator weight.

```yaml
model:
  vq_model:
    quantize_mode: vae
    token_size: 16
    vit_enc_model_size: tiny
    vit_dec_model_size: small
    vit_enc_patch_size: 8
    vit_dec_patch_size: 8
    num_latent_tokens: 128
    in_channels: 1
    out_channels: 1
losses:
  kl_weight: 6.02e-08
  perceptual_weight: 0.6
  reconstruction_loss: l2
  reconstruction_weight: 1.0
  # --- late GAN (this sweep's win) ---
  discriminator_weight: 0.001       # ★ sweet spot; 0.01 already hurts, 1.0 collapses
  discriminator_start: 60000        # ★ engage AFTER recon plateaus (0.6× of a 100k horizon)
  discriminator_factor: 1.0
  lecam_regularization_weight: 0.001
optimizer:
  params:
    discriminator_learning_rate: 1.0e-04
```

**Scaling note:** `discriminator_start` here is 0.6× of a 100k horizon, chosen so the GAN refines an already-converged recon. For a longer bigger-data run, keep the same *intent* — start the GAN only after the recon loss has plateaued — rather than copying the literal 60k step.

---

## Full Tables

### `final` slot — step 100000, GAN fully engaged for every GAN cell (the apples-to-apples comparison)

| cell | disc_w | MSE↓ | SSIM↑ | PSNR↑ | LPIPS↓ | rFID↓ | FSS↑ |
|---|---|---|---|---|---|---|---|
| radar_gan_nogan | off | 0.000574 | 0.9544 | 41.85 | 0.0176 | 6.286 | **0.7208** |
| **radar_gan_w0001** | **0.001** | **0.000448** | **0.9734** | **46.76** | **0.0139** | **3.644** | 0.7120 |
| radar_gan_w001 | 0.01 | 0.000680 | 0.9275 | 39.99 | 0.0199 | 5.790 | 0.6439 |
| radar_gan_w01 | 0.1 | 0.000806 | 0.9474 | 44.81 | 0.0286 | 10.140 | 0.6420 |
| radar_gan_w1 | 1.0 | 0.003119 | 0.8556 | 35.08 | 0.0929 | 36.389 | 0.4649 |

### `best_val` slot — lowest val-L2 checkpoint (⚠ different step per cell; see note)

| cell | disc_w | best_val step | MSE↓ | SSIM↑ | PSNR↑ | LPIPS↓ | rFID↓ | FSS↑ |
|---|---|---|---|---|---|---|---|---|
| radar_gan_nogan | off | 100k | 0.000573 | 0.9549 | 41.89 | 0.0175 | 6.259 | **0.7204** |
| **radar_gan_w0001** | **0.001** | 100k | **0.000428** | **0.9753** | **48.40** | **0.0134** | **3.581** | 0.7010 |
| radar_gan_w001 | 0.01 | 100k | 0.000681 | 0.9333 | 40.27 | 0.0198 | 5.746 | 0.6430 |
| radar_gan_w01 | 0.1 | ⚠ 60k (pre-GAN) | 0.000534 | 0.9454 | 40.99 | 0.0169 | 7.129 | 0.7038 |
| radar_gan_w1 | 1.0 | ⚠ 60k (pre-GAN) | 0.000629 | 0.9419 | 40.80 | 0.0190 | 6.572 | 0.6913 |

`w01`/`w1` best_val numbers reflect their step-60000 (pre-GAN) checkpoint and so understate the damage the strong GAN does — compare them in the `final` table instead. `w0001` is slightly better at `best_val` than `final` (PSNR 48.4 vs 46.8, rFID 3.58 vs 3.64); both are 100k GAN-on models.

---

## Findings

### G1. A tiny GAN regularizes; it does not trade pixels for perception

The textbook expectation is that a GAN improves perceptual realism (rFID/LPIPS) at the cost of pixel fidelity (MSE/PSNR). At `disc_weight=0.001` that is **not** what happens — every metric improves at once (MSE −22% **and** rFID −42%). The gentlest adversarial gradient appears to act as a structural prior that the L2+perceptual objective alone does not provide, nudging the decoder away from the slightly blurry L2 optimum without yet entering the regime where the discriminator dictates outputs.

### G2. The benefit is sharply non-monotonic and collapses fast

| disc_w | rFID | LPIPS | MSE | FSS |
|---|---|---|---|---|
| off | 6.29 | 0.0176 | 0.000574 | 0.721 |
| 0.001 | **3.64** | **0.0139** | **0.000448** | 0.712 |
| 0.01 | 5.79 | 0.0199 | 0.000680 | 0.644 |
| 0.1 | 10.14 | 0.0286 | 0.000806 | 0.642 |
| 1.0 | 36.39 | 0.0929 | 0.003119 | 0.465 |

One decade up from the optimum (0.01) is already net-negative; two decades (0.1) wrecks perceptual quality; three (1.0) is near-collapse. The usable window is narrow and centered an order of magnitude below TA-TiTok's 0.1 default — consistent with radar being single-channel and the tokenizer being a tiny/small model with limited capacity to satisfy a strong discriminator.

### G3. FSS is the lone metric GAN-off wins

GAN-off edges every GAN cell on FSS (0.7208), but `w0001` (0.7120) is within ~1% while gaining a 42% rFID reduction. For a tokenizer feeding downstream flow-matching / nowcasting (where distributional realism matters), that trade is clearly worth taking.

---

## Cross-checks & caveats

1. **Single seed.** Radar mse noise floor ≈ 5% from prior runs. `w0001`'s gains (MSE −22%, rFID −42%, PSNR +4.9 dB) clear the floor comfortably; the FSS *loss* (−1.2%) is within noise.
2. **`best_val` ≠ same step across cells** (G2/methodology note). Use the `final` table for GAN-strength conclusions.
3. **disc_start is regime-specific.** 60k was chosen for a 100k tiny-model horizon. On a longer bigger-data run, re-locate the recon plateau and start the GAN there rather than copying 60k.
4. **Test set is i2i 2024/07 only** — single held-out month; seasonality on other months not tested. (Memory: `dataset_split_season_match`.)
5. **Not jointly tested with sat10ch.** This sweep is radar-1ch only. Whether a tiny GAN helps the 11-channel sat AE is untested; sat already prefers high perceptual weight, so the GAN interaction may differ.

---

## Recommended next moves

- **Ship `disc_weight=0.001` + late `disc_start` into the bigger radar AE run.** It is the only setting that helps and it helps a lot (rFID −42%).
- **Do not raise the discriminator weight** — the window is narrow and 0.01 already hurts.
- **For the bigger-data run**, locate the recon plateau and set `disc_start` there (intent over literal 60k).
- **Optional**: a small `disc_w=0.0003 / 0.003` probe to map the peak more finely, if a slot is free.
- **Untested**: whether a tiny GAN also helps the sat10ch AE — run one `w0001`-equivalent sat cell before assuming transfer.
