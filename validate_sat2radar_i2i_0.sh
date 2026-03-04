#!/usr/bin/env bash
set -euo pipefail

# Config & checkpoint for Sat2Radar i2i run 0
CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/Sat2Radar-i2i-pretrained-FlowTiTok-XL.py"
CKPT_PATH="/mnt/ssd_1/yghu/Experiments/sat2radar_flowtitok_run/ckpts/30000.ckpt"  # TODO: 按实际改
OUTPUT_DIR="/mnt/ssd_1/yghu/Experiments/sat2radar_flowtitok_run/val"
SPLIT="val"            # train/val/test
MODE="v2v"             # 固定为 i2i
MAX_BATCHES=10
BATCH_SIZE=8

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

python scripts/validate_sat2radar_v2v.py \
  --config "${CONFIG_PATH}" \
  --ckpt "${CKPT_PATH}" \
  --out_dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --mode "${MODE}" \
  --max_batches "${MAX_BATCHES}" \
  --batch_size "${BATCH_SIZE}" \
  --gpu "${CUDA_VISIBLE_DEVICES}"

