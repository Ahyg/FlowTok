# Sat→Radar i2i: 4-Model Controlled Ablation — Design

Date: 2026-05-19
Repo: `FlowTok/` (NCI GADI, project `kl02`)
Status: design for approval

## 1. Goal

Train **four i2i (single-frame satellite → radar) models** as a controlled A/B
study. All four share one baseline recipe; each non-baseline model changes
**exactly one** variable so any metric delta is attributable to that change.

| # | Model | The single change vs baseline |
|---|---|---|
| **M1** | baseline | none (i2i analog of run1 `sat10ch-direct`) |
| **M2** | + single-stream textVAE dual-branch projector | `use_text_vae_encoder=True` (+ contrastive/KLD aux losses) |
| **M3** | generation algorithm → diffusion | flow-matching replaced by Diffi2i-style DDIM (token space) |
| **M4** | flow-matching predicts radar tokens | network predicts x̂1 instead of velocity (math-equivalent reparam) |

## 2. Fixed components (identical across M1–M4)

- **Tokenizers (AE)**: run1 self-trained, frozen, no retrain:
  - sat: `Experiments/sat10ch_flowtitok_ae_bl77_vae_scratch_run1_gadi/checkpoint-200000/ema_model/pytorch_model.bin` (11ch in/out, 128px, 77 tok, 16-dim)
  - radar: `Experiments/radar_flowtitok_ae_bl77_vae_scratch_run1_gadi/checkpoint-200000/ema_model/pytorch_model.bin` (1ch, 128px, 77 tok, 16-dim)
- **DiT backbone**: `FlowTok-B` (hidden 768, depth 12, heads 12).
- **Optim**: AdamW, lr 4e-4, wd 0.03, betas (0.9, 0.95), `customized` scheduler, warmup 5000 (= run1).
- **Data path/aug**: `num_frames=1`, crop 128, all 10 IR + lightning (11ch sat, `cond_use_sat_lightning_tokens=False`), hflip/vflip on.
- **Budget**: 60k steps, batch 64, gpuhopper, 1 GPU/job, walltime-aware self-resubmit.
- **Eval**: holdout test on 2024-07 nofilter, `--seed 42`, FSS/wFSS/SSIM/PSNR/MAE.

## 3. Dataset (build once, shared)

Training script reads `train`/`val` slots from a **single** `filelist_path`;
test read separately. Build 3 sub-filelists then merge into one 3-slot
filelist (same pattern as existing `merge_dataset_v2v_cpu_gadi.sh`).

| Slot | Date range | Filter | build args |
|---|---|---|---|
| train | 2021-05-01 … 2021-10-31 | ct005 (same as before) | `--coverage-threshold 0.05 --split-ratio 1,0,0` |
| val | 2024-06-01 … 2024-06-30 | none | `--coverage-threshold 0.0 --split-ratio 0,1,0` |
| test | 2024-07-01 … 2024-07-31 | none | `--coverage-threshold 0.0 --split-ratio 0,0,1` |

Output filelist: `/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_baseline_2021summer_merged.pkl`
(`.pkl` is this codebase's pre-existing filelist format, produced/consumed only
by the project's own `build_dataset.py`/dataset loader — not external input).

New scripts:
- `build_dataset_i2i_baseline_cpu_gadi.sh` — `qsub -v SPLIT=train|val|test` (3 runs, `--split-mode days`).
- `merge_dataset_i2i_baseline_cpu_gadi.sh` — merge train/val/test slots into the merged filelist.

## 4. Model specifications

### M1 — baseline
New config `configs/Sat2Radar-i2i-b-baseline-2021summer_gadi.py`: i2i clone of
`Sat2Radar-v2v-sat10ch-direct-FlowTiTok-XL_gadi.py` with
`nnet.name="flowtok-b"`, `num_frames=1`, run1 AE paths, the merged filelist,
`use_text_vae_encoder=False`, velocity prediction, `noising_type="constant"`,
`noising_scale=0.1`, `cfg_indicator=0.0`, `n_steps=60_000`, `batch_size=64`,
`save_interval=10_000`, `eval_interval=1_000`. **No code change.**

### M2 — single-stream textVAE dual-branch projector
Config = M1 + `use_text_vae_encoder=True`,
`losses.contrastive_loss_weight=1.0`, `losses.kld_loss_weight=1e-4`,
`model.textVAE = Args(num_blocks=6, hidden_dim=256, num_attention_heads=4, dropout_prob=0.1, clip_loss_weight=0.0, align_quantized=False, use_pretrained=False, tokenizer_checkpoint="", freeze_encoder=False)`.
Everything else (noising, CFG, AE) identical to M1. Uses the existing
`_text_encoder` (FlowEncoder → x0/mu/logvar branch) + `_text_projector`
(contrastive branch) inside `libs/model/flowtok_t2i.py`. **No code change**
(existing path, exercised by `..._textVAE` experiment before).

### M3 — generation algorithm → diffusion (token space)
Operate the **Diffi2i diffusion scheme on radar tokens**, sat tokens as condition.

- Forward q: `z_τ = α_τ·z1 + σ_τ·ε`, `ε~N(0,I)`; **linear** β schedule
  (`linspace(1e-4, 2e-2, 1001)`), `α_τ=√∏(1−β)`, `σ_τ=√(1−α²)`; 1000 train timesteps.
- Conditioning: **channel-concat** sat tokens `z_sat∈[B,L,16]` with noisy radar
  `z_τ∈[B,L,16]` → DiT input `[B,L,32]` (faithful to Diffi2i `cat((x_t,cond),1)`).
  FlowTok-B `x_embedder` in_channels 16→32 via new flag (see §5).
- Target: **pred_x0** — network predicts `ẑ1` (clean radar tokens);
  loss = `0.5·‖ẑ1 − z1‖²` (l2).
- Sampling: **DDIM, 500 steps** (γ=0). Per step from `ẑ1` derive
  `ε̂=(z_τ−α_τ·ẑ1)/σ_τ`, then `z_{τ'} = α_{τ'}·ẑ1 + σ_{τ'}·ε̂`; subseq
  `linspace(1000,0,501).round()`.
- Decode `ẑ1` → radar via the frozen radar run1 AE decoder (same as baseline).
- New module `diffusion/token_diffusion.py` (`TokenDiffusion`: `loss`,
  `ddim_sample`), gated by `config.generation_algorithm="diffusion"`.

### M4 — flow-matching predicts radar tokens (x1 reparam)
Keep the existing flow path and **multi-step Euler ODE** unchanged; only the
network's prediction target and the train/sample reparameterization change.

- Path (unchanged): `x_t = ψ(t) = (t·(σmin/σmax−1)+1)·x0 + t·x1`,
  true velocity `v = (σmin/σmax−1)·x0 + x1` (constant on the straight path);
  `x0` = sat-derived start, `x1` = radar tokens.
- Training target = `x1` (radar tokens): loss = `0.5·‖x̂1 − x1‖²` (replaces the
  `‖v̂ − target_velocity‖²` term; aux losses unchanged).
- Sampling (multi-step Euler kept): store the fixed start `x0 = x_T`. Each step
  the net predicts `x̂1`; convert `v̂ = (σmin/σmax−1)·x0 + x̂1`; existing
  `step_in_dsigma` Euler update `x ← x + v̂·Δσ`. Mathematically equivalent to
  v-prediction, only reparameterized.
- New flag `config.flow_prediction_target ∈ {"velocity"(default), "radar_tokens"}`
  in `diffusion/flow_matching.py` + `ODEEulerFlowMatchingSolver`.

## 5. Backward-compatible code changes (must not break in-flight jobs)

All new behavior is **opt-in with legacy defaults** (gadi-ml-workflow §6).
In-flight v2v jobs and old ckpts keep working.

- `config.generation_algorithm` — default `"flow_matching"`, `| "diffusion"`.
- `config.flow_prediction_target` — default `"velocity"`, `| "radar_tokens"`.
- `config.diffusion = d(schedule="linear", target="pred_x0", train_timesteps=1000, sample_steps=500, gamma="ddim")` — only read when `generation_algorithm="diffusion"`.
- `config.cond_concat_channels` (bool, default False) — when True, DiT
  `x_embedder` input = 2× token dim (M3 channel-concat). Constructed from
  `config` so old configs build the legacy 1× input.
- `scripts/train_sat2radar_v2v.py`: branch on `generation_algorithm`
  (FlowMatching vs TokenDiffusion); pass `flow_prediction_target`.
- `scripts/validate_sat2radar_v2v.py` / `test_sat2radar_v2v.py`: read all new
  keys via `.get(key, legacy_default)`.
- Sanity import + 1 forward/backward per new path before any `qsub`.

## 6. Tiny-overfit gate (required for M2/M3/M4)

M2/M3/M4 touch model/loss/generation paths → before the 60k full run, each
runs a **32-sample i2i overfit (~5k steps)** on gpuhopper (~1–3h) using
`scripts/make_overfit_filelist.py` (first 32 train entries → 3-slot filelist).
Success: main loss collapses (fm/diff < ~0.005) and vis memorizes the 32
samples. M1 has no code change → straight to full. A failed gate blocks that
model's 60k submission until root-caused.

## 7. Compute & artifacts

Four independent walltime-aware self-resubmitting PBS scripts
(`#PBS -P kl02 -q gpuhopper`, ngpus=1, mem=90GB, `-m abe`,
`storage=gdata/kl02+scratch/kl02`), each: detect latest ckpt → train toward
60k → re-`qsub` if `<60k`. Job logs in `/scratch/kl02/$USER/Projv2v/job_logs/`.

Artifact dirs (preserve, never delete; `_old_<reason>` on redo):
- `Experiments/sat2radar_flowtok_i2i_b_baseline_2021summer/`
- `Experiments/sat2radar_flowtok_i2i_b_textvae_2021summer/`
- `Experiments/sat2radar_flowtok_i2i_b_diffusion_2021summer/`
- `Experiments/sat2radar_flowtok_i2i_b_xpred_2021summer/`

## 8. Evaluation protocol

After all four reach 60k: `scripts/test_sat2radar_v2v.py` on the merged
filelist's **test** slot (2024-07 nofilter), `--seed 42`, batch 8, identical
sampling budget per family (flow: 20 Euler steps; M3: 500 DDIM steps), metrics
FSS/wFSS/SSIM/PSNR/MAE → one comparison table. Also test the matching mid
checkpoints (e.g. 30k) for a convergence-speed view.

## 9. Out of scope (YAGNI)

No AE retraining; no v2v; no new tokenizer; no CFG sweep; no multi-GPU;
no architectural search beyond the one change per model.

## 10. Open risks

- M3 token-space diffusion with pred_x0 + linear schedule is unvalidated here
  → the tiny gate is the guard.
- M2 KLD on tokens can NaN (gadi-ml-workflow known_failures) → watch first 1k
  steps; KLD weight kept low (1e-4) per textVAE precedent.
- 6-month filtered i2i sample count unknown until build; if very small, 60k may
  overfit — val-on-2024-06 curve is the early-stop signal.
