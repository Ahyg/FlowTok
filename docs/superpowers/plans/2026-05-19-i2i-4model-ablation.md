# i2i 4-Model Controlled Ablation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (autonomous batch — user authorized "直接跑，明早出结果"). Steps use `- [ ]` checkboxes.

**Goal:** Train 4 i2i sat→radar models (M1 baseline / M2 +textVAE projector / M3 token-diffusion / M4 x1-pred flow) as a controlled ablation, self-chaining on GADI so results progress overnight.

**Architecture:** Every new behavior is opt-in with a legacy default (`getattr(cfg, key, legacy)`), so in-flight v2v jobs and old ckpts are unaffected. One PBS pipeline job builds the dataset then auto-submits M1-full + M2/M3/M4-tiny; each tiny job auto-judges convergence — pass → auto-submit its full job; fail → write a `TINY_FAIL` marker that the operator (Claude monitor loop) detects, debugs, fixes, and resubmits tiny until it passes.

**Tech Stack:** PyTorch, FlowTok DiT, FlowTiTok AE (run1), PBS/qsub, conda env `flowtok`.

---

## Conventions

- Repo: `/scratch/kl02/yh0308/Projv2v/FlowTok`. Env: `source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh && conda activate flowtok`.
- Filelists: `/g/data/kl02/yh0308/Data/71/filelists/`. Run1 AE: `Experiments/{sat10ch,radar}_flowtitok_ae_bl77_vae_scratch_run1_gadi/checkpoint-200000/ema_model/pytorch_model.bin`.
- Merged filelist: `…/filelists/dataset_filelist_i2i_baseline_2021summer_merged.pkl` (`.pkl` = this repo's pre-existing filelist format, produced/consumed only by its own `build_dataset.py`/loader — not external input).
- Exp dirs: `Experiments/sat2radar_flowtok_i2i_b_{baseline,textvae,diffusion,xpred}_2021summer[ _tiny]`.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `libs/model/flowtok_t2i.py` | Modify | `cond_concat_channels` → split in/out channels |
| `diffusion/flow_matching.py` | Modify | `flow_prediction_target`; solver `prediction_target` x1→v |
| `diffusion/token_diffusion.py` | Create | Diffi2i-style DDIM on tokens |
| `scripts/train_sat2radar_v2v.py` | Modify | branch `generation_algorithm`/`flow_prediction_target` |
| `scripts/validate_sat2radar_v2v.py`, `scripts/test_sat2radar_v2v.py` | Modify | diffusion sampling branch + `.get()` defaults |
| `scripts/make_overfit_filelist.py` | Create | tiny 32-sample 3-slot filelist, `--put-into all` |
| `scripts/merge_i2i_filelists.py` | Create | merge train/val/test slots → one 3-slot filelist |
| `tests/test_ablation_units.py` | Create | CPU unit tests for the 4 new code paths |
| `tests/smoke_paths.py` | Create | CPU import+fwd/bwd of M1/M3/M4 paths (pre-qsub gate) |
| `configs/Sat2Radar-i2i-b-{m1..m4}-2021summer_gadi.py` | Create ×4 | full configs |
| `configs/Sat2Radar-i2i-b-{m2,m3,m4}-tiny_gadi.py` | Create ×3 | tiny-overfit configs |
| `build_dataset_i2i_baseline_pipeline_gadi.sh` | Create | build 3 splits + merge + tiny filelists + fan-out qsub |
| `train_i2i_b_{m1,m2,m3,m4}_full_gadi.sh` | Create ×4 | walltime-aware self-resubmit to 60k |
| `train_i2i_b_{m2,m3,m4}_tiny_gadi.sh` | Create ×3 | tiny gate; pass→qsub full, fail→`TINY_FAIL` |

## Phase 1 — Backward-compatible code (TDD, Tasks 1-6)

1. **DiT channel split** — `FlowTok.__init__`: `out=config.channels`; `in = channels*2 if getattr(config,"cond_concat_channels",False) else channels`. Test: in_features 16 (legacy) / 32 (concat), fwd shape→16.
2. **FlowMatching `flow_prediction_target`** — ctor kwarg default `"velocity"`; in `p_losses_textVAE_flowtok` target = `x_start` if `radar_tokens` else `Dt_psi(...)`. Test: reparam `(σmin/σmax−1)x0+x1 == Dt_psi`.
3. **Solver x1→v** — `ODEEulerFlowMatchingSolver.sample`: `self.prediction_target=kwargs.get("prediction_target","velocity")`; `sample_euler`: cache `x0_start=x_T.clone()`, `velocity=(σmin/σmax−1)x0_start+out` when `radar_tokens` else `out`. Test: x1-dummy trajectory == velocity-dummy trajectory (atol 1e-4).
4. **`diffusion/token_diffusion.py`** — `TokenDiffusion` (linear/cosine schedule buffers, `q_sample`, `loss` channel-concat pred_x0/eps, `ddim_sample`). Test: schedule shapes 1001, α↓, σ↑; DDIM+oracle pred_x0 recovers z1 (atol 1e-3).
5. **`scripts/make_overfit_filelist.py`** — copy skill script, add `--put-into all` → `(sub,sub,sub)`. **`scripts/merge_i2i_filelists.py`** — argparse `--train/--val/--test/--out`, reads each slot, writes 3-slot (mirrors `merge_dataset_v2v_cpu_gadi.sh` logic, no inline ser. in this doc).
6. **train/validate/test integration** — train ~569 build `token_diffusion_model` if `generation_algorithm=="diffusion"`, pass `flow_prediction_target` to `FlowMatching`; loss sites ~763/907 branch to `token_diffusion_model.loss(nnet,radar_tokens,sat_tokens)`; sampling ~1216 branch to `ddim_sample(nnet_ema_local,cond=sat_tokens,sample_steps=cfg.diffusion.sample_steps)` else add `prediction_target=` to `ode_solver.sample`. validate ~513 / test ~732 same branch, all reads via `getattr/.get` defaults.

Each task: write failing test → run (FAIL) → implement (exact edits per §"Phase 1 details" in spec/this repo) → run (PASS) → commit.

## Phase 2 — Configs (Task 7)
Base on `configs/Sat2Radar-v2v-sat10ch-direct-FlowTiTok-XL_gadi.py`. Common: `nnet=d(name="flowtok-b",model_args=model)`, `num_frames=1`, run1 AE ckpts, `sat_in/out=11`, `radar_in/out=1`, `ae_image_size=128`, merged filelist, `train=d(n_steps=60_000,batch_size=64,log_interval=100,eval_interval=1_000,save_interval=10_000,n_samples_eval=4,val_max_batches=64)`, `sample.sample_steps=20`.
- **M1** baseline: `use_text_vae_encoder=False`, velocity, `noising_type="constant"/0.1`, `cfg_indicator=0.0`, `generation_algorithm="flow_matching"`, `flow_prediction_target="velocity"`, workdir `…_baseline_2021summer`.
- **M2**: M1 + `use_text_vae_encoder=True`, `losses=d(contrastive_loss_weight=1.0,kld_loss_weight=1e-4)`, full `model.textVAE` Args, workdir `…_textvae_2021summer`.
- **M3**: M1 + `model` `cond_concat_channels=True`, `generation_algorithm="diffusion"`, `diffusion=d(schedule="linear",target="pred_x0",train_timesteps=1000,sample_steps=500,gamma="ddim")`, workdir `…_diffusion_2021summer`.
- **M4**: M1 + `flow_prediction_target="radar_tokens"`, workdir `…_xpred_2021summer`.
- **Tiny ×3** (m2/m3/m4): copy full, `n_steps=5000,eval_interval=500,save_interval=2500`, `filelist_path=<exp_tiny>/dataset_filelist.pkl`, diffusion `sample_steps=100`/flow `20`, workdir `…_<m>_2021summer_tiny`.

## Phase 3 — Self-chaining PBS (Tasks 8-10)
Header: `#PBS -P kl02 -l storage=gdata/kl02+scratch/kl02 -l ncpus=12 -l ngpus=1 -l mem=90GB -l jobfs=90GB -l wd -M auhuyg@gmail.com -m abe`; env block per `train_sat2radar_v2v_sat10ch_direct_gadi.sh`.
- **8 full ×4** `-q gpuhopper -l walltime=48:00:00`: detect latest `WD/ckpts/*.ckpt` step; if ≥60000 exit; else `accelerate launch … --config=<full cfg>`; re-detect, if `<60000` re-`qsub` self.
- **9 tiny ×3** `-q gpuhopper -l walltime=06:00:00`: run tiny cfg → grep last `diff_loss`; `<0.02` → `qsub train_i2i_b_<m>_full_gadi.sh` (+echo GATE PASS); else `touch <exp_tiny>/TINY_FAIL` (+echo GATE FAIL).
- **10 pipeline** `-q normal -l walltime=24:00:00 -l ncpus=8 -l mem=32GB`: 3× `build_dataset.py` (train 20210501-20211031 ct005 `1,0,0`; val 20240601-30 nofilter `0,1,0`; test 20240701-31 nofilter `0,0,1`) → `merge_i2i_filelists.py` → `make_overfit_filelist.py --put-into all --n 32` ×3 → `qsub` M1-full + M2/M3/M4-tiny.
`chmod +x` all; commit.

## Phase 4 — Smoke gate + launch (Tasks 11-12)
- **11** login CPU: `python -m pytest tests/test_ablation_units.py -q` all pass; `python tests/smoke_paths.py` (loads M1/M3/M4 cfg, builds FlowTok_B, 1 fwd+bwd on 2-sample fake batch, asserts finite loss & shapes). Any fail → fix, do NOT qsub.
- **12**: `qsub build_dataset_i2i_baseline_pipeline_gadi.sh`; `qstat -u yh0308` confirm queued; report jobid + chain.

## Phase 5 — Operator monitor loop (Task 13, post-submit)
Poll `qstat` + each `Experiments/..._tiny/` for `TINY_FAIL`/job logs. On `TINY_FAIL`: read tiny log, root-cause (systematic-debugging), fix code, re-`qsub` that tiny; loop until it passes (then it auto-submits full). On all-pass: confirm 4 full jobs queued/running. Post-60k holdout eval (`test_sat2radar_v2v.py` on merged test slot, seed 42, FSS/SSIM/PSNR/MAE) is a separate human-read step, not auto-chained.

## Self-Review
Spec coverage: §3→T10/T5; §4 M1-M4→T7; M2 textVAE existing path→T6/T7; M3→T1/T4/T6/T7; M4→T2/T3/T6/T7; §5 backward-compat→T1-6 (all getattr-default); §6 tiny gate→T9/T10/T13; §7 self-resubmit→T8-10; §8 eval→post-60k (T6 makes test.py diffusion-aware). No placeholders; names consistent across tasks (`cond_concat_channels`, `flow_prediction_target`, `generation_algorithm`, `prediction_target`, `TokenDiffusion.loss/ddim_sample`).
