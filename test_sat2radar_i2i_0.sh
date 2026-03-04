#!/usr/bin/env bash
set -euo pipefail

# Config & checkpoint for Sat2Radar i2i run 0
CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/Sat2Radar-i2i-pretrained-FlowTiTok-XL.py"
CKPT_PATH="/mnt/ssd_1/yghu/Experiments/sat2radar_flowtitok_run/ckpts/50000.ckpt"
OUTPUT_DIR="/mnt/ssd_1/yghu/Experiments/sat2radar_flowtitok_run/test_i2i_50000"

SPLIT="test"              # train/val/test，可按需要改
MODE="v2v"                # 固定为 i2i
MAX_BATCHES_METRICS=-1    # -1 = 用整个 split 做指标
MAX_BATCHES_IMAGES=10     # 只保存前 N 个 batch 的图像/GIF
BATCH_SIZE=8

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

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

