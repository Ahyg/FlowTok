#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/Sat2Radar-Video-XL.py"
CKPT_PATH="/mnt/ssd_1/yghu/Experiments/sat2radar_video_run/ckpts/50000.ckpt"
OUTPUT_DIR="/mnt/ssd_1/yghu/Experiments/sat2radar_video_run/test_i2i"
SPLIT="test"           # or val/train
MODE="i2i"             # i2i or v2v
MAX_BATCHES_METRICS=-1 # -1 = use full split for metrics
MAX_BATCHES_IMAGES=10  # only save images for first N batches
BATCH_SIZE=8

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

python scripts/test_sat2radar_v2v.py \
  --config "${CONFIG_PATH}" \
  --ckpt "${CKPT_PATH}" \
  --out_dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --mode "${MODE}" \
  --max_batches_metrics "${MAX_BATCHES_METRICS}" \
  --max_batches_images "${MAX_BATCHES_IMAGES}" \
  --batch_size "${BATCH_SIZE}" \
  --gpu "${CUDA_VISIBLE_DEVICES}"

