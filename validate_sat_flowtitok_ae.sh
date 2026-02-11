#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/sat_flowtok_ae_bl77_vae.yaml"
CHECKPOINT_PATH="/mnt/ssd_1/yghu/Experiments/sat_flowtok_ae_bl77_vae_run1/checkpoint-650000/unwrapped_model/pytorch_model.bin"
OUTPUT_DIR="/mnt/ssd_1/yghu/Experiments/sat_flowtok_ae_bl77_vae_run1/val_images"
MAX_BATCHES=16

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

python /mnt/ssd_1/yghu/Code/FlowTok/scripts/validate_flowtitok_ae.py \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --out_dir "${OUTPUT_DIR}" \
  --max_batches "${MAX_BATCHES}"
