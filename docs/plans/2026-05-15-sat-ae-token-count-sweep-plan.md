# Sat AE Token-Count Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a 3-cell ablation (`num_latent_tokens ∈ {77, 128, 256}`) at fixed ~27M (tiny enc + small dec) sat AE architecture on lab2-local, then evaluate reconstruction on 2024/07 test to determine if token count is a real capacity bottleneck for the sat AE.

**Architecture:** Three independent training runs share every hyperparameter except `num_latent_tokens` and a proportionally-scaled `kl_weight` (kept at `1e-6 × 77/N` so per-token KL pressure is constant across cells). All runs write to `/mnt/ssd_2/yghu/Experiments/` to avoid ssd_1 capacity pressure. Two 4090s (GPU 0, 2) run two cells in parallel; the third cell starts on whichever frees first.

**Tech Stack:** PyTorch + Accelerate + FlowTok fork (`Code/FlowTok`), OmegaConf YAML configs, conda env `flowtok`. Spec: `docs/specs/2026-05-14-sat-ae-token-count-sweep-design.md`.

---

## File Structure

**Modify:**
- `modeling/modules/blocks.py:227-241` (TiTokEncoder) and `:307-321` (TiTokDecoder) — add `"tiny"` size entry (width=384, depth=6, heads=6).
- `utils/train_utils.py:2041-2056` (`save_checkpoint`) — add rolling-overwrite + best-val variant.
- `utils/train_utils.py:1248-1300` (eval block inside `train_one_epoch`) — track best val L2, trigger best-val save.

**Create (configs):**
- `configs/sat10ch_ae_tok77_lab2.yaml`
- `configs/sat10ch_ae_tok128_lab2.yaml`
- `configs/sat10ch_ae_tok256_lab2.yaml`

**Create (scripts):**
- `scripts/build_dataset_i2i_train21_val24w1.py` — builds a combined train+val pkl from existing single-split pkls.
- `scripts/launch_sat_ae_token_sweep_lab2.sh` — orchestrates GPU 0+2 parallel + tok256 deferred.
- `scripts/eval_sat_ae_token_sweep.sh` — runs `test_flowtitok_ae.py` over all final + best_val ckpts.
- `scripts/latent_utilization.py` — per-token variance, KL stats, PCA effective rank.
- `scripts/diagnose_ae_token_sweep.py` — port of `/tmp/diagnose_run1_vs_run3.py` adapted for 3-cell comparison.

**Output / data (not git-tracked):**
- `/mnt/ssd_2/yghu/Experiments/sat10ch_ae_tok{77,128,256}_run1/`
- `Data/71_3m/filelists/dataset_filelist_i2i_train21_val24w1.pkl`

**Final results doc:**
- `docs/specs/results/2026-05-15-token-sweep-results.md`

---

## Task 1: Add `"tiny"` size to TiTokEncoder / TiTokDecoder registries

**Files:**
- Modify: `modeling/modules/blocks.py:227-241` and `:307-321`

- [ ] **Step 1: Edit encoder size registry**

In `TiTokEncoder.__init__` (around line 227), add `"tiny"` to each of the three dicts:

```python
self.width = {
        "tiny": 384,
        "small": 512,
        "base": 768,
        "large": 1024,
    }[self.model_size]
self.num_layers = {
        "tiny": 6,
        "small": 8,
        "base": 12,
        "large": 24,
    }[self.model_size]
self.num_heads = {
        "tiny": 6,
        "small": 8,
        "base": 12,
        "large": 16,
    }[self.model_size]
```

- [ ] **Step 2: Edit decoder size registry**

In `TiTokDecoder.__init__` (around line 307), make the identical three-dict edit (width/num_layers/num_heads) — same `"tiny": 384/6/6` entries.

- [ ] **Step 3: Sanity-check instantiation**

Run in repo root:

```bash
cd /mnt/ssd_1/yghu/Code/FlowTok && conda run -n flowtok python -c "
from omegaconf import OmegaConf
import torch, sys
sys.path.insert(0, '.')
from modeling.tatitok import TATiTok

cfg = OmegaConf.create({
    'model': {'vq_model': {
        'quantize_mode': 'vae', 'token_size': 16,
        'vit_enc_model_size': 'tiny', 'vit_dec_model_size': 'small',
        'vit_enc_patch_size': 16, 'vit_dec_patch_size': 16,
        'num_latent_tokens': 77, 'in_channels': 11, 'out_channels': 11,
        'finetune_decoder': False, 'is_legacy': False,
    }},
    'dataset': {'preprocessing': {'crop_size': 128}},
})
m = TATiTok(cfg)
n = sum(p.numel() for p in m.parameters())
print(f'tiny enc + small dec param count: {n/1e6:.1f}M')
assert 20e6 < n < 40e6, f'expected ~27M, got {n}'
print('OK')
"
```

Expected: prints something like `tiny enc + small dec param count: 26.8M` and `OK`. If the assert fails, adjust width/depth (e.g., width=320 for an even smaller model) and re-run. Record the actual number in the commit message.

- [ ] **Step 4: Commit**

```bash
git add modeling/modules/blocks.py
git commit -m "Add 'tiny' size (width=384, depth=6, heads=6) to TiTok enc/dec registries"
```

---

## Task 2: Add best-val tracking + rolling-overwrite to save_checkpoint

**Files:**
- Modify: `utils/train_utils.py:2041-2056` (save_checkpoint)
- Modify: `utils/train_utils.py:1199-1220` (save_every block in train_one_epoch)
- Modify: `utils/train_utils.py:1248-1300` (eval block in train_one_epoch)

- [ ] **Step 1: Patch `save_checkpoint` to support named slots**

Replace `save_checkpoint` (line 2041) with this version:

```python
def save_checkpoint(model, output_dir, accelerator, global_step, logger, slot=None) -> Path:
    """Save ckpt to a slot directory (overwrites if exists) or step-named dir.

    Args:
        slot: if str, saves to f'checkpoint-{slot}' (overwrites). If None, saves to
              f'checkpoint-{global_step}' (legacy step-named behaviour).
    """
    import shutil
    if slot is None:
        save_path = Path(output_dir) / f"checkpoint-{global_step}"
    else:
        save_path = Path(output_dir) / f"checkpoint-{slot}"
        if save_path.exists() and accelerator.is_main_process:
            shutil.rmtree(save_path)
        accelerator.wait_for_everyone()

    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained_weight(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

    accelerator.save_state(save_path)
    return save_path
```

- [ ] **Step 2: Change the `save_every` call to use `slot="latest"`**

In `train_one_epoch` around line 1200, change:

```python
save_path = save_checkpoint(
    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
```

to:

```python
save_path = save_checkpoint(
    model, config.experiment.output_dir, accelerator, global_step + 1,
    logger=logger, slot="latest")
```

There are 3 call sites in `train_one_epoch`/`train_one_epoch_generator`/`train_one_epoch_t2i_generator` (lines 1200, 1468, 1612). Only patch the **first one** (line 1200 — used by AE training). Leave the other two alone.

- [ ] **Step 3: Add best-val tracking state at the top of `train_one_epoch`**

Just after `model.train()` (around line 933), add:

```python
# Best-val tracking — persists across epochs via an attribute on the function.
if not hasattr(train_one_epoch, "_best_val_l2"):
    train_one_epoch._best_val_l2 = float("inf")
    train_one_epoch._best_val_step = -1
```

- [ ] **Step 4: Save best_val ckpt when val L2 improves**

Inside the eval block (around line 1296, where `eval_scores` is logged), AFTER the `accelerator.log(eval_log, step=...)` line, add:

```python
            # Track best val L2 (use EMA scores if EMA, else non-EMA).
            val_l2 = float(eval_scores.get("reconstruction_loss",
                                            eval_scores.get("l2_loss",
                                                eval_scores.get("recon_loss", float("inf")))))
            if val_l2 < train_one_epoch._best_val_l2:
                train_one_epoch._best_val_l2 = val_l2
                train_one_epoch._best_val_step = global_step + 1
                logger.info(
                    f"[BEST-VAL] New best val L2 = {val_l2:.6f} at step {global_step + 1}. Saving."
                )
                save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1,
                    logger=logger, slot="best_val")
                accelerator.wait_for_everyone()
```

The fallback chain on `eval_scores.get(...)` exists because the exact key name depends on which loss path is active — confirm the actual key by running Task 3's smoke test first and patching the key if needed.

- [ ] **Step 5: Add a final-step save after the loop**

In `train_one_epoch` after the for-loop ends but before `return global_step`, add:

```python
    # Final ckpt at end of training run.
    if global_step >= config.training.max_train_steps:
        save_checkpoint(
            model, config.experiment.output_dir, accelerator, global_step,
            logger=logger, slot="final")
```

Locate the return point — there's likely a `return global_step` at the end of the function (line ~1310-ish). Place this right before that return.

- [ ] **Step 6: Sanity-check the patches compile**

```bash
cd /mnt/ssd_1/yghu/Code/FlowTok && conda run -n flowtok python -c "
import sys; sys.path.insert(0, '.')
from utils.train_utils import save_checkpoint, train_one_epoch
import inspect
sig = inspect.signature(save_checkpoint)
assert 'slot' in sig.parameters, 'slot kwarg missing'
print('save_checkpoint accepts slot kwarg: OK')
"
```

- [ ] **Step 7: Commit**

```bash
git add utils/train_utils.py
git commit -m "save_checkpoint: add 'slot' kwarg + rolling overwrite; best-val tracking in train loop"
```

---

## Task 3: Build combined train21+val24w1 pkl filelist

**Files:**
- Create: `scripts/build_dataset_i2i_train21_val24w1.py`

The default AE dataloader (line 657-734 in `train_utils.py`) reads the **same** pkl for both train (`filelist_split="train"`) and val (`filelist_split="val"`). Existing pkls are split-specific (one has only "train", another has only "val"). We need a combined pkl.

- [ ] **Step 1: Write the merge script**

```python
# scripts/build_dataset_i2i_train21_val24w1.py
"""Combine existing train and val i2i pkls into one with both 'train' and 'val' keys."""
import argparse
import pickle
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_pkl", required=True,
                   help="Existing pkl with 'train' key (or assumed train-only)")
    p.add_argument("--val_pkl", required=True,
                   help="Existing pkl with 'val' key (or assumed val-only)")
    p.add_argument("--out_pkl", required=True)
    args = p.parse_args()

    with open(args.train_pkl, "rb") as f:
        train_obj = pickle.load(f)
    with open(args.val_pkl, "rb") as f:
        val_obj = pickle.load(f)

    # Normalize: each obj is dict-of-lists or just a list.
    def extract(obj, key):
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            # single-split dict with different key name — assume first value is the list
            return next(iter(obj.values()))
        return obj  # assume list

    train_files = extract(train_obj, "train")
    val_files = extract(val_obj, "val")

    combined = {"train": train_files, "val": val_files}
    Path(args.out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(combined, f)

    print(f"Wrote {args.out_pkl}: train={len(train_files)} files, val={len(val_files)} files")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Locate existing train and val pkls**

```bash
ls -la /mnt/ssd_1/yghu/Data/71_3m/filelists/ | grep -E "i2i.*train.*ct005|i2i.*val.*202406"
```

Expected: a train pkl (e.g., `dataset_filelist_i2i_train_202105_202110_ct005.pkl`) and a val pkl (e.g., `dataset_filelist_i2i_val_202406w1.pkl`). If the val pkl is i2i and only has frames from 2024/06 first week, use it. If it's the v2v val (clip-based), we'd need an i2i version — note this gap in the plan output and ask the user before proceeding (see Risk in spec §8).

- [ ] **Step 3: Run the merge**

```bash
cd /mnt/ssd_1/yghu/Code/FlowTok && conda run -n flowtok python scripts/build_dataset_i2i_train21_val24w1.py \
  --train_pkl /mnt/ssd_1/yghu/Data/71_3m/filelists/dataset_filelist_i2i_train_202105_202110_ct005.pkl \
  --val_pkl   /mnt/ssd_1/yghu/Data/71_3m/filelists/dataset_filelist_i2i_val_202406w1.pkl \
  --out_pkl   /mnt/ssd_1/yghu/Data/71_3m/filelists/dataset_filelist_i2i_train21_val24w1.pkl
```

Expected: prints `train=~26000 files, val=~600 files` and writes the pkl.

- [ ] **Step 4: Verify pkl loads with the dataset class**

```bash
cd /mnt/ssd_1/yghu/Code/FlowTok && conda run -n flowtok python -c "
import sys; sys.path.insert(0, '.')
from data.dataset import SatelliteRadarNpyDataset
PKL = '/mnt/ssd_1/yghu/Data/71_3m/filelists/dataset_filelist_i2i_train21_val24w1.pkl'
for split in ['train', 'val']:
    ds = SatelliteRadarNpyDataset(
        base_dir=None, mode='satellite', use_lightning=True,
        filelist_path=PKL, filelist_split=split,
    )
    print(f'{split}: len={len(ds)}')
    sample = ds[0]
    print(f'  sample keys: {list(sample.keys())}, image shape: {sample[\"image\"].shape}')
"
```

Expected: both splits load, image shape `[11, H, W]`.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_dataset_i2i_train21_val24w1.py
git commit -m "Add script to build combined i2i train21+val24w1 pkl"
```

---

## Task 4: Create the three sweep configs

**Files:**
- Create: `configs/sat10ch_ae_tok77_lab2.yaml`
- Create: `configs/sat10ch_ae_tok128_lab2.yaml`
- Create: `configs/sat10ch_ae_tok256_lab2.yaml`

- [ ] **Step 1: Write `sat10ch_ae_tok77_lab2.yaml`**

```yaml
experiment:
  project: sat10ch_ae_token_sweep
  name: sat10ch_ae_tok77_run1
  output_dir: /mnt/ssd_2/yghu/Experiments/sat10ch_ae_tok77_run1
  max_train_examples: 250000
  save_every: 5000        # overwrites slot="latest"
  eval_every: 2000        # val + best_val update
  generate_every: 5000
  log_every: 200
  log_grad_norm_every: 1000
  resume: true
  logging_dir: /mnt/ssd_2/yghu/Experiments/sat10ch_ae_tok77_run1/logs

model:
  vq_model:
    quantize_mode: vae
    token_size: 16
    vit_enc_model_size: tiny
    vit_dec_model_size: small
    vit_enc_patch_size: 16
    vit_dec_patch_size: 16
    num_latent_tokens: 77
    in_channels: 11
    out_channels: 11
    finetune_decoder: false
    is_legacy: false

losses:
  discriminator_start: 9999999      # disabled (matches run1 effective behaviour)
  quantizer_weight: 1.0
  discriminator_factor: 1.0
  discriminator_weight: 0.1
  perceptual_loss: lpips-convnext_s-1.0-0.1
  perceptual_weight: 1.1
  reconstruction_loss: l2
  reconstruction_weight: 1.0
  lecam_regularization_weight: 0.001
  kl_weight: 1.0e-06                # baseline; tok128 uses 6.02e-7, tok256 uses 3.01e-7
  logvar_init: 0.0

dataset:
  params:
    filelist_path: /mnt/ssd_1/yghu/Data/71_3m/filelists/dataset_filelist_i2i_train21_val24w1.pkl
    filelist_split: train
    mode: satellite
    use_lightning: true
    num_workers: 4
  preprocessing:
    resize_shorter_edge: 128
    crop_size: 128
    random_crop: true
    random_flip: true
    res_ratio_filtering: true

optimizer:
  name: adamw
  params:
    learning_rate: 0.0001
    discriminator_learning_rate: 0.0001
    beta1: 0.9
    beta2: 0.999
    weight_decay: 0.0001

lr_scheduler:
  scheduler: cosine
  params:
    learning_rate: ${optimizer.params.learning_rate}
    warmup_steps: 6000
    end_lr: 1.0e-05

finetune:
  enabled: false
  text_guidance_from_filename: true

training:
  gradient_accumulation_steps: 1
  per_gpu_batch_size: 32
  mixed_precision: 'no'
  enable_tf32: true
  enable_wandb: false
  use_ema: true
  seed: 42
  max_train_steps: 60000
  num_generated_images: 1
  max_grad_norm: 1.0
  nan_check: false
  nan_skip: true
```

- [ ] **Step 2: Write `sat10ch_ae_tok128_lab2.yaml`**

Same as tok77 with these diffs:

```yaml
experiment:
  name: sat10ch_ae_tok128_run1
  output_dir: /mnt/ssd_2/yghu/Experiments/sat10ch_ae_tok128_run1
  logging_dir: /mnt/ssd_2/yghu/Experiments/sat10ch_ae_tok128_run1/logs

model:
  vq_model:
    num_latent_tokens: 128

losses:
  kl_weight: 6.02e-07              # = 1e-6 × 77/128
```

(All other fields identical to tok77.)

- [ ] **Step 3: Write `sat10ch_ae_tok256_lab2.yaml`**

Same as tok77 with these diffs:

```yaml
experiment:
  name: sat10ch_ae_tok256_run1
  output_dir: /mnt/ssd_2/yghu/Experiments/sat10ch_ae_tok256_run1
  logging_dir: /mnt/ssd_2/yghu/Experiments/sat10ch_ae_tok256_run1/logs

model:
  vq_model:
    num_latent_tokens: 256

losses:
  kl_weight: 3.01e-07              # = 1e-6 × 77/256
```

- [ ] **Step 4: Verify configs load and instantiate**

```bash
cd /mnt/ssd_1/yghu/Code/FlowTok && for N in 77 128 256; do
  conda run -n flowtok python -c "
from omegaconf import OmegaConf
import sys; sys.path.insert(0, '.')
from modeling.tatitok import TATiTok
cfg = OmegaConf.load('configs/sat10ch_ae_tok${N}_lab2.yaml')
m = TATiTok(cfg)
n = sum(p.numel() for p in m.parameters())
print(f'tok${N}: params={n/1e6:.1f}M, kl_weight={cfg.losses.kl_weight}')
"
done
```

Expected: prints param counts (~27M each), kl_weight values matching `77/N` scaling. tok128 ~27M + ε (more latent_token_pos_emb params); tok256 ~27M + 2ε.

- [ ] **Step 5: Commit**

```bash
git add configs/sat10ch_ae_tok77_lab2.yaml configs/sat10ch_ae_tok128_lab2.yaml configs/sat10ch_ae_tok256_lab2.yaml
git commit -m "Configs: sat AE token-count sweep (77/128/256) at tiny+small for lab2-local"
```

---

## Task 5: Smoke-test one cell (100 steps) before committing to the full sweep

This is the GO/NO-GO checkpoint. Catch OOM / config errors / data-pipeline issues here, not 4 hours into a real run.

- [ ] **Step 1: Make a smoke-test copy of tok128 config**

```bash
cp /mnt/ssd_1/yghu/Code/FlowTok/configs/sat10ch_ae_tok128_lab2.yaml \
   /tmp/sat10ch_ae_tok128_smoke.yaml
```

Patch the smoke copy in-place to:
- `max_train_steps: 100`
- `save_every: 50`
- `eval_every: 50`
- `output_dir: /tmp/sat10ch_ae_tok128_smoke_out`
- `logging_dir: /tmp/sat10ch_ae_tok128_smoke_out/logs`

```bash
cd /mnt/ssd_1/yghu/Code/FlowTok && conda run -n flowtok python -c "
from omegaconf import OmegaConf
c = OmegaConf.load('/tmp/sat10ch_ae_tok128_smoke.yaml')
c.training.max_train_steps = 100
c.experiment.save_every = 50
c.experiment.eval_every = 50
c.experiment.output_dir = '/tmp/sat10ch_ae_tok128_smoke_out'
c.experiment.logging_dir = '/tmp/sat10ch_ae_tok128_smoke_out/logs'
OmegaConf.save(c, '/tmp/sat10ch_ae_tok128_smoke.yaml')
print('patched smoke config')
"
```

- [ ] **Step 2: Run 100 steps on GPU 0**

```bash
cd /mnt/ssd_1/yghu/Code/FlowTok && \
  CUDA_VISIBLE_DEVICES=0 conda run -n flowtok --live-stream accelerate launch --num_processes 1 \
    scripts/train_flowtitok_ae.py --config /tmp/sat10ch_ae_tok128_smoke.yaml 2>&1 | tee /tmp/smoke_tok128.log
```

Expected:
- No OOM
- 100 training steps complete
- val eval fires at step 50 and 100
- best_val ckpt saved at least once (look for `[BEST-VAL] New best val L2 = ...`)
- `checkpoint-latest/` and `checkpoint-best_val/` and `checkpoint-final/` dirs exist after run

- [ ] **Step 3: Verify ckpt slots**

```bash
ls -la /tmp/sat10ch_ae_tok128_smoke_out/ | grep checkpoint
```

Expected: exactly `checkpoint-latest`, `checkpoint-best_val`, `checkpoint-final` (and NO step-numbered dirs leaking through).

- [ ] **Step 4: Check VRAM headroom**

While the smoke is running (in another terminal):

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

If GPU 0 used >22 GB of 24 GB → reduce `per_gpu_batch_size` to 16 in all 3 configs. If <18 GB → could try bumping to 48 or 64 but **only if the increase preserves the gradient-noise characteristics** (sat AE training is sensitive). Default: leave at 32.

- [ ] **Step 5: Verify the eval_scores key**

Grep the smoke log for the eval block output:

```bash
grep -A 15 "EMA EVALUATION\|Non-EMA EVALUATION" /tmp/smoke_tok128.log | head -30
```

Confirm one of `reconstruction_loss`, `l2_loss`, or `recon_loss` appears in the printed eval_scores dict. If a *different* key holds the L2 loss, edit the fallback chain in Task 2 Step 4 accordingly and re-run smoke test before proceeding.

- [ ] **Step 6: Clean up smoke output**

```bash
rm -rf /tmp/sat10ch_ae_tok128_smoke_out /tmp/sat10ch_ae_tok128_smoke.yaml /tmp/smoke_tok128.log
```

- [ ] **Step 7: No commit** (smoke test produces no source changes; just a GO/NO-GO).

---

## Task 6: Write the lab2 launcher script

**Files:**
- Create: `scripts/launch_sat_ae_token_sweep_lab2.sh`

Strategy: tok77 starts on GPU 0, tok128 starts on GPU 2, both detached via `nohup`. A poll loop waits for the first to finish, then starts tok256 on whichever GPU freed.

- [ ] **Step 1: Write the launcher**

```bash
#!/bin/bash
# scripts/launch_sat_ae_token_sweep_lab2.sh
# Orchestrate 3-cell sat AE token sweep on lab2 with 2 available GPUs.
# Runs tok77 on GPU 0 + tok128 on GPU 2 in parallel; defers tok256 until one frees.

set -euo pipefail
FLOWTOK_ROOT="/mnt/ssd_1/yghu/Code/FlowTok"
EXP_ROOT="/mnt/ssd_2/yghu/Experiments"
mkdir -p "${EXP_ROOT}"

source "/home/yghu/miniconda3/etc/profile.d/conda.sh"
conda activate flowtok
cd "${FLOWTOK_ROOT}"

launch_cell () {
  local CELL_TAG="$1"; local GPU_IDX="$2"; local CONFIG="$3"
  local OUT_DIR="${EXP_ROOT}/sat10ch_ae_${CELL_TAG}_run1"
  mkdir -p "${OUT_DIR}"
  echo "[$(date '+%F %T')] Launch ${CELL_TAG} on GPU ${GPU_IDX} → ${OUT_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU_IDX}" nohup \
    accelerate launch --num_processes 1 \
      scripts/train_flowtitok_ae.py --config "${CONFIG}" \
      > "${OUT_DIR}/training.log" 2>&1 &
  echo $! > "${OUT_DIR}/training.pid"
  echo "[$(date '+%F %T')] ${CELL_TAG} PID=$(cat ${OUT_DIR}/training.pid)"
}

wait_for_one () {
  local PID1="$1"; local PID2="$2"
  while kill -0 "${PID1}" 2>/dev/null && kill -0 "${PID2}" 2>/dev/null; do
    sleep 60
  done
  # Return whichever PID is alive (the survivor)
  if kill -0 "${PID1}" 2>/dev/null; then echo "${PID1}"; else echo "${PID2}"; fi
}

# Launch tok77 on GPU 0
launch_cell "tok77" 0 "${FLOWTOK_ROOT}/configs/sat10ch_ae_tok77_lab2.yaml"
PID_77=$(cat "${EXP_ROOT}/sat10ch_ae_tok77_run1/training.pid")
GPU_77=0

# Launch tok128 on GPU 2
launch_cell "tok128" 2 "${FLOWTOK_ROOT}/configs/sat10ch_ae_tok128_lab2.yaml"
PID_128=$(cat "${EXP_ROOT}/sat10ch_ae_tok128_run1/training.pid")
GPU_128=2

# Wait for one to finish
SURVIVOR=$(wait_for_one "${PID_77}" "${PID_128}")
if [ "${SURVIVOR}" = "${PID_77}" ]; then
  FREE_GPU=${GPU_128}
  echo "[$(date '+%F %T')] tok128 finished first; freeing GPU ${FREE_GPU}"
else
  FREE_GPU=${GPU_77}
  echo "[$(date '+%F %T')] tok77 finished first; freeing GPU ${FREE_GPU}"
fi

# Launch tok256 on the freed GPU
launch_cell "tok256" "${FREE_GPU}" "${FLOWTOK_ROOT}/configs/sat10ch_ae_tok256_lab2.yaml"
PID_256=$(cat "${EXP_ROOT}/sat10ch_ae_tok256_run1/training.pid")

# Wait for both remaining jobs
echo "[$(date '+%F %T')] Waiting for both remaining jobs to finish (${SURVIVOR}, ${PID_256})"
wait "${SURVIVOR}" 2>/dev/null || true
wait "${PID_256}"  2>/dev/null || true

echo "[$(date '+%F %T')] All three cells finished."
for TAG in tok77 tok128 tok256; do
  ls -la "${EXP_ROOT}/sat10ch_ae_${TAG}_run1/" | grep checkpoint || echo "  (no ckpts for ${TAG})"
done
```

- [ ] **Step 2: Make executable and lint**

```bash
chmod +x /mnt/ssd_1/yghu/Code/FlowTok/scripts/launch_sat_ae_token_sweep_lab2.sh
bash -n /mnt/ssd_1/yghu/Code/FlowTok/scripts/launch_sat_ae_token_sweep_lab2.sh
```

Expected: no syntax errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/launch_sat_ae_token_sweep_lab2.sh
git commit -m "Add token sweep launcher for lab2 (GPU 0+2 parallel + GPU-handoff for tok256)"
```

---

## Task 7: Write latent-utilization analysis script

**Files:**
- Create: `scripts/latent_utilization.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
# scripts/latent_utilization.py
"""Measure how 'used' the latent tokens are by an AE.

Given a TA-TiTok AE ckpt + config + test pkl, encode 256 random samples and report:
- Per-token activation variance (across batch) and count of near-dead tokens
- Per-token mean KL divergence from prior
- PCA effective rank of the flattened [B, N*D] latents

Outputs JSON + markdown to --out_dir.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.dataset import SatelliteRadarNpyDataset
from modeling.tatitok import TATiTok
from modeling.quantizer.quantizer import DiagonalGaussianDistribution


def load_ae(config_path: str, ckpt_path: str, device: str = "cuda:0") -> TATiTok:
    cfg = OmegaConf.load(config_path)
    model = TATiTok(cfg).to(device).eval()
    sd_path = Path(ckpt_path) / "unwrapped_model" / "pytorch_model.bin"
    if not sd_path.exists():
        # Fallback: maybe ckpt_path is a .bin directly
        sd_path = Path(ckpt_path)
    sd = torch.load(sd_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    return model


@torch.no_grad()
def gather_latents(model, dataset, n_samples: int = 256, batch_size: int = 16, device: str = "cuda:0"):
    indices = np.random.RandomState(42).choice(len(dataset), size=n_samples, replace=False)
    means, logvars, latents = [], [], []
    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start:start + batch_size]
        imgs = torch.stack([dataset[i]["image"] for i in batch_idx]).to(device).float()
        # crop to model's expected size
        crop = model.encoder.image_size
        if imgs.shape[-1] != crop:
            imgs = imgs[..., :crop, :crop]
        params = model.encode(imgs)  # returns DiagonalGaussianDistribution.parameters
        if isinstance(params, tuple):
            params = params[0]
        if not isinstance(params, torch.Tensor):
            params = params.parameters
        post = DiagonalGaussianDistribution(params)
        means.append(post.mean.cpu())     # [B, D, N]
        logvars.append(post.logvar.cpu())
        latents.append(post.sample().cpu())
    means = torch.cat(means, 0).numpy()       # [N_samp, D, N_tok]
    logvars = torch.cat(logvars, 0).numpy()
    latents = torch.cat(latents, 0).numpy()
    return means, logvars, latents


def per_token_stats(means, logvars, latents):
    # means/logvars/latents shape: [B, D, N]
    B, D, N = means.shape
    # per-token activation variance across batch, averaged over D
    var_per_token = latents.var(axis=0).mean(axis=0)  # [N]
    mean_var = var_per_token.mean()
    near_dead = int((var_per_token < 0.01 * mean_var).sum())
    # per-token KL: KL of N(mean, var) || N(0, 1), summed over D, averaged over batch
    var = np.exp(logvars)
    kl = 0.5 * (means ** 2 + var - 1.0 - logvars)  # [B, D, N]
    kl_per_token = kl.sum(axis=1).mean(axis=0)  # [N]
    # PCA effective rank: entropy of normalized singular value squared distribution
    flat = latents.reshape(B, -1)
    s = np.linalg.svd(flat - flat.mean(0, keepdims=True), compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    return {
        "num_tokens": int(N),
        "token_dim": int(D),
        "var_per_token_mean": float(mean_var),
        "var_per_token_min": float(var_per_token.min()),
        "var_per_token_max": float(var_per_token.max()),
        "near_dead_count": near_dead,
        "near_dead_frac": near_dead / N,
        "kl_per_token_mean": float(kl_per_token.mean()),
        "kl_per_token_min": float(kl_per_token.min()),
        "kl_per_token_max": float(kl_per_token.max()),
        "pca_effective_rank": eff_rank,
        "pca_max_possible": float(min(B, N * D)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True, help="path to checkpoint-<slot>/ dir")
    p.add_argument("--filelist", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    ds = SatelliteRadarNpyDataset(
        base_dir=None, mode="satellite", use_lightning=True,
        filelist_path=args.filelist, filelist_split=args.split,
    )
    model = load_ae(args.config, args.ckpt)
    means, logvars, latents = gather_latents(model, ds, n_samples=args.n_samples)
    stats = per_token_stats(means, logvars, latents)
    stats["ckpt"] = str(args.ckpt)
    stats["config"] = str(args.config)

    with open(Path(args.out_dir) / "latent_utilization.json", "w") as f:
        json.dump(stats, f, indent=2)

    md_lines = [
        f"# Latent utilization for {Path(args.ckpt).name}",
        "",
        f"- num_tokens: {stats['num_tokens']}",
        f"- token_dim: {stats['token_dim']}",
        f"- mean activation variance: {stats['var_per_token_mean']:.4g}",
        f"- near-dead tokens (variance < 1% of mean): {stats['near_dead_count']} / {stats['num_tokens']} ({stats['near_dead_frac']:.1%})",
        f"- KL per token: mean {stats['kl_per_token_mean']:.4g}, min {stats['kl_per_token_min']:.4g}, max {stats['kl_per_token_max']:.4g}",
        f"- PCA effective rank: {stats['pca_effective_rank']:.1f} (max possible: {stats['pca_max_possible']:.0f})",
    ]
    with open(Path(args.out_dir) / "latent_utilization.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Wrote {args.out_dir}/latent_utilization.json + .md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Quick smoke**

After Task 5's smoke ckpt exists (if cleaned up, re-do briefly), test:

```bash
cd /mnt/ssd_1/yghu/Code/FlowTok && conda run -n flowtok python scripts/latent_utilization.py \
  --config /tmp/sat10ch_ae_tok128_smoke.yaml \
  --ckpt /tmp/sat10ch_ae_tok128_smoke_out/checkpoint-final \
  --filelist /mnt/ssd_1/yghu/Data/71_3m/filelists/dataset_filelist_i2i_test_202407_nofilter.pkl \
  --split test --n_samples 32 \
  --out_dir /tmp/latent_util_test
```

Expected: writes `latent_utilization.json` and `.md` without error. Values will be junk (100-step model) but the script should run cleanly.

- [ ] **Step 3: Commit**

```bash
git add scripts/latent_utilization.py
git commit -m "Add latent_utilization.py — per-token variance, KL stats, PCA effective rank"
```

---

## Task 8: Adapt the diagnose script for 3-cell comparison

**Files:**
- Create: `scripts/diagnose_ae_token_sweep.py` (adapted from `/tmp/diagnose_run1_vs_run3.py`)

- [ ] **Step 1: Copy and adapt**

Read `/tmp/diagnose_run1_vs_run3.py` (~250 lines). Save adapted version with these changes:
- Accept `--cells` as a JSON-list of `{name, config, ckpt}` dicts (3 entries).
- Compute the same low-freq MSE / high-freq MSE / 8x8 patch MSE / Sobel edge IoU / radial power spectrum, but now for 3 columns instead of 2.
- Output: per-sample comparison figures (8 samples, evenly spaced from test set) showing all 3 cells side-by-side with ground truth.
- Markdown summary table with one row per metric × per channel × per cell.

The structure mirrors the existing diagnose script — the heavy lift is generalizing from 2 to N cells. Use a loop over `cells` and aggregate into pandas-style dict-of-dicts.

If reading `/tmp/diagnose_run1_vs_run3.py` returns "file too large" or the file has been cleaned up: re-implement from scratch using the spec §6.2 list as the metric definition. ~200 lines.

- [ ] **Step 2: Test invocation**

```bash
cd /mnt/ssd_1/yghu/Code/FlowTok && conda run -n flowtok python scripts/diagnose_ae_token_sweep.py --help
```

Expected: usage printed without error.

- [ ] **Step 3: Commit**

```bash
git add scripts/diagnose_ae_token_sweep.py
git commit -m "Add token-sweep diagnostic script: freq + spatial + edge for 3 cells"
```

---

## Task 9: Write the evaluation orchestrator

**Files:**
- Create: `scripts/eval_sat_ae_token_sweep.sh`

- [ ] **Step 1: Write the orchestrator**

```bash
#!/bin/bash
# scripts/eval_sat_ae_token_sweep.sh
# After training: run test_flowtitok_ae.py + latent_utilization.py for each cell × {best_val, final}.

set -euo pipefail
FLOWTOK_ROOT="/mnt/ssd_1/yghu/Code/FlowTok"
EXP_ROOT="/mnt/ssd_2/yghu/Experiments"
TEST_PKL="/mnt/ssd_1/yghu/Data/71_3m/filelists/dataset_filelist_i2i_test_202407_nofilter.pkl"

source "/home/yghu/miniconda3/etc/profile.d/conda.sh"
conda activate flowtok
cd "${FLOWTOK_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

for TAG in tok77 tok128 tok256; do
  CELL_DIR="${EXP_ROOT}/sat10ch_ae_${TAG}_run1"
  CONFIG="${FLOWTOK_ROOT}/configs/sat10ch_ae_${TAG}_lab2.yaml"
  for SLOT in best_val final; do
    CKPT="${CELL_DIR}/checkpoint-${SLOT}"
    if [ ! -d "${CKPT}" ]; then
      echo "[$(date '+%F %T')] SKIP ${TAG}/${SLOT}: no ckpt"
      continue
    fi
    OUT="${CELL_DIR}/eval_${SLOT}"
    mkdir -p "${OUT}"
    echo "[$(date '+%F %T')] === ${TAG} / ${SLOT} ==="

    # Reconstruction metrics
    python -u scripts/test_flowtitok_ae.py \
      --config "${CONFIG}" \
      --checkpoint "${CKPT}/unwrapped_model/pytorch_model.bin" \
      --out_dir "${OUT}" \
      --max_batches_metrics -1 \
      --max_batches_images 2 \
      --split test \
      --filelist_path "${TEST_PKL}" \
      --lpips_net alex

    # Latent utilization
    python -u scripts/latent_utilization.py \
      --config "${CONFIG}" \
      --ckpt "${CKPT}" \
      --filelist "${TEST_PKL}" \
      --split test \
      --n_samples 256 \
      --out_dir "${OUT}"
  done
done

# 3-cell diagnostic comparison (uses best_val ckpts)
echo "[$(date '+%F %T')] === 3-cell diagnostic ==="
DIAG_OUT="${EXP_ROOT}/sat_ae_token_sweep_diagnostic"
mkdir -p "${DIAG_OUT}"
python -u scripts/diagnose_ae_token_sweep.py \
  --cells '[
    {"name":"tok77","config":"'${FLOWTOK_ROOT}'/configs/sat10ch_ae_tok77_lab2.yaml","ckpt":"'${EXP_ROOT}'/sat10ch_ae_tok77_run1/checkpoint-best_val"},
    {"name":"tok128","config":"'${FLOWTOK_ROOT}'/configs/sat10ch_ae_tok128_lab2.yaml","ckpt":"'${EXP_ROOT}'/sat10ch_ae_tok128_run1/checkpoint-best_val"},
    {"name":"tok256","config":"'${FLOWTOK_ROOT}'/configs/sat10ch_ae_tok256_lab2.yaml","ckpt":"'${EXP_ROOT}'/sat10ch_ae_tok256_run1/checkpoint-best_val"}
  ]' \
  --filelist "${TEST_PKL}" \
  --split test \
  --n_samples 8 \
  --out_dir "${DIAG_OUT}"

echo "[$(date '+%F %T')] All eval done."
ls -la "${EXP_ROOT}"/sat10ch_ae_*_run1/eval_*/metrics.json "${DIAG_OUT}"/ 2>&1
```

- [ ] **Step 2: Make executable + lint**

```bash
chmod +x /mnt/ssd_1/yghu/Code/FlowTok/scripts/eval_sat_ae_token_sweep.sh
bash -n /mnt/ssd_1/yghu/Code/FlowTok/scripts/eval_sat_ae_token_sweep.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_sat_ae_token_sweep.sh
git commit -m "Add eval orchestrator: recon metrics + latent util + 3-cell diagnostic"
```

---

## Task 10: Kick off the training sweep (long-running, not a code change)

- [ ] **Step 1: Re-verify GPU 0 and GPU 2 are free**

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv
```

If anything new appeared on GPU 0 or GPU 2, stop and consult the user before launching.

- [ ] **Step 2: Verify ssd_2 has ≥10 GB free**

```bash
df -h /mnt/ssd_2
```

Need ≥10 GB headroom (4.5 GB ckpts + sample images + tensorboard logs).

- [ ] **Step 3: Launch in detached `nohup` and tail**

```bash
cd /mnt/ssd_1/yghu/Code/FlowTok && \
  nohup bash scripts/launch_sat_ae_token_sweep_lab2.sh > /tmp/token_sweep_orchestrator.log 2>&1 &
echo "Orchestrator PID: $!"
```

Then check progress periodically:

```bash
tail -5 /tmp/token_sweep_orchestrator.log
tail -3 /mnt/ssd_2/yghu/Experiments/sat10ch_ae_tok77_run1/training.log
tail -3 /mnt/ssd_2/yghu/Experiments/sat10ch_ae_tok128_run1/training.log
```

Expected wallclock: 24-36 hours total.

- [ ] **Step 4: After all three finish, verify ckpts exist**

```bash
for TAG in tok77 tok128 tok256; do
  echo "=== ${TAG} ==="
  ls -la /mnt/ssd_2/yghu/Experiments/sat10ch_ae_${TAG}_run1/ | grep -E "checkpoint|training.log"
done
```

Expected: each cell has `checkpoint-latest`, `checkpoint-best_val`, `checkpoint-final` (and `training.log`).

- [ ] **Step 5: No commit** (training output is not in git).

---

## Task 11: Run the evaluation sweep

- [ ] **Step 1: Run eval on GPU 0 in the foreground (~30 min)**

```bash
cd /mnt/ssd_1/yghu/Code/FlowTok && \
  CUDA_VISIBLE_DEVICES=0 bash scripts/eval_sat_ae_token_sweep.sh 2>&1 | tee /tmp/token_sweep_eval.log
```

Expected:
- 6 × `test_flowtitok_ae.py` runs (3 cells × 2 slots): MSE/PSNR/SSIM/LPIPS/rFID/FSS per cell
- 6 × `latent_utilization.py` runs
- 1 × `diagnose_ae_token_sweep.py` run

- [ ] **Step 2: Aggregate the result jsons**

```bash
cd /mnt/ssd_1/yghu/Code/FlowTok && conda run -n flowtok python -c "
import json
from pathlib import Path
ROOT = Path('/mnt/ssd_2/yghu/Experiments')
print('| Cell | Slot | MSE | LPIPS | rFID | SSIM | EdgeIoU | DeadFrac | EffRank |')
print('|---|---|---|---|---|---|---|---|---|')
for tag in ['tok77', 'tok128', 'tok256']:
    for slot in ['best_val', 'final']:
        cell_dir = ROOT / f'sat10ch_ae_{tag}_run1' / f'eval_{slot}'
        if not (cell_dir / 'metrics.json').exists():
            print(f'| {tag} | {slot} | (missing) |')
            continue
        m = json.load(open(cell_dir / 'metrics.json'))
        lu = json.load(open(cell_dir / 'latent_utilization.json')) if (cell_dir / 'latent_utilization.json').exists() else {}
        # Adjust keys based on what test_flowtitok_ae actually emits
        mse = m.get('mse', m.get('MSE', 'NA'))
        lpips = m.get('lpips', m.get('LPIPS', 'NA'))
        rfid = m.get('rfid', m.get('rFID', 'NA'))
        ssim = m.get('ssim', m.get('SSIM', 'NA'))
        edge = m.get('edge_iou', 'NA')
        dead = lu.get('near_dead_frac', 'NA')
        rank = lu.get('pca_effective_rank', 'NA')
        print(f'| {tag} | {slot} | {mse} | {lpips} | {rfid} | {ssim} | {edge} | {dead} | {rank} |')
" | tee /tmp/token_sweep_summary.md
```

- [ ] **Step 3: Apply spec §7 decision rule**

For each pair (tok128 vs tok77, tok256 vs tok128, tok256 vs tok77):
- Compute relative change on the 3 headline metrics (high-freq MSE, Sobel edge IoU, LPIPS) from the diagnostic JSON output.
- Mark "improved" if ≥15% relative change with consistent sign on ≥2 of 3.
- Record decision in `/tmp/token_sweep_summary.md`.

- [ ] **Step 4: No commit yet** (results go into the doc in Task 12).

---

## Task 12: Write the results document

**Files:**
- Create: `docs/specs/results/2026-05-15-token-sweep-results.md`

- [ ] **Step 1: Author the results doc**

Structure:

```markdown
# Sat AE Token-Count Sweep — Results

**Date:** 2026-05-15 (eval completion)
**Spec:** docs/specs/2026-05-14-sat-ae-token-count-sweep-design.md
**Plan:** docs/plans/2026-05-15-sat-ae-token-count-sweep-plan.md

## TL;DR

[One sentence + decision per spec §7]

## Recon metrics on 2024/07 test (best_val ckpts)

[paste table from Task 11 Step 2]

## Diagnostic (high-freq MSE / Sobel edge IoU / radial power spectrum)

[link to diagnose output figures, paste summary table]

## Latent utilization

[paste latent_utilization summary; if any cell has high near-dead fraction, flag it]

## Decision

[Cite the spec §7 decision-rule row that applies; record action taken: adopt N, move to recipe sweep, etc.]

## Follow-ups

[List any downstream FlowTok pilot or recipe variation that this result motivates]
```

- [ ] **Step 2: Commit**

```bash
git add docs/specs/results/2026-05-15-token-sweep-results.md
git commit -m "Token-sweep results: [TL;DR one-liner from doc]"
```

---

## Self-Review

Run through the checklist below before declaring the plan ready.

### Spec coverage

- §1 Question — Task 12 records the answer. ✓
- §3 Variables / KL normalization — Task 4 (configs encode them). ✓
- §3 Architecture — Task 1 (registry entries), Task 4 (configs use them). ✓
- §4 Compute / orchestration — Task 6 (launcher). ✓
- §5 Ckpt policy — Task 2 (rolling slot save). ✓
- §6.1 Standard recon metrics — Task 9 (test_flowtitok_ae). ✓
- §6.2 Freq/spatial/edge diagnostic — Task 8 (diagnose_ae_token_sweep). ✓
- §6.3 Latent utilization — Task 7 (latent_utilization). ✓
- §7 Decision rule — Task 11 Step 3. ✓
- §8 Risks — combined train+val pkl gap addressed by Task 3; smoke catches OOM (Task 5); per-token KL verified inline (already in spec §8 resolved row).
- §9 Deliverables — Tasks 4, 6, 9, 12 produce them. ✓

### Placeholder scan

No "TBD", "fill in details", or unspecific instructions. The only soft branch is Task 5 Step 5 (key name for L2 loss in eval_scores) — but it's explicitly conditional on smoke output, and Task 2 Step 4 has a fallback chain to handle three likely key names. Acceptable.

### Type / name consistency

- `slot` kwarg on `save_checkpoint` — used consistently in Task 2 Step 1 (definition), Step 2 (call site change), Step 4 (best_val call), Step 5 (final call).
- Cell tags `tok77`, `tok128`, `tok256` — consistent across Tasks 4, 6, 9, 11, 12.
- Output paths `/mnt/ssd_2/yghu/Experiments/sat10ch_ae_<tag>_run1/` — consistent everywhere.
- Config filenames `sat10ch_ae_<tag>_lab2.yaml` — consistent.

### Scope check

12 tasks, one of which is a long-running training (Task 10) and one is documentation (Task 12). Implementation surface is moderate: 2 code-file modifications + 5 new scripts + 3 configs + 1 results doc. Single experiment, single sub-system. Not decomposed further.
