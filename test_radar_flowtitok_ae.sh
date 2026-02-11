#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/radar_flowtok_ae_bl77_vae.yaml"
CHECKPOINT_PATH="/mnt/ssd_1/yghu/Experiments/radar_flowtok_ae_bl77_vae_run1/checkpoint-450000/ema_model/pytorch_model.bin"
OUTPUT_DIR="/mnt/ssd_1/yghu/Experiments/radar_flowtok_ae_bl77_vae_run1/test_images"
MAX_BATCHES_METRICS=-1
MAX_BATCHES_IMAGES=32

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

python /mnt/ssd_1/yghu/Code/FlowTok/scripts/test_flowtitok_ae.py \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --out_dir "${OUTPUT_DIR}" \
  --max_batches_metrics "${MAX_BATCHES_METRICS}" \
  --max_batches_images "${MAX_BATCHES_IMAGES}" \
  --fss_thresholds "0,5,10,15,20,25,30,35,40,45,50,55,60" \
  --fss_scales "1,2,3,4,5,6,7,8,9,10" \
  --split test
