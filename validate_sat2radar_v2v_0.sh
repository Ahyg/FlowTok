#!/usr/bin/env bash
set -euo pipefail

# Config & checkpoint for Sat2Radar v2v run 0
CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/Sat2Radar-v2v-pretrained-FlowTiTok-XL.py"
CKPT_PATH="/mnt/ssd_1/yghu/Experiments/sat2radar_flowtitok_run_v2v/ckpts/30000.ckpt"  # TODO: 按实际改
OUTPUT_DIR="/mnt/ssd_1/yghu/Experiments/sat2radar_flowtitok_run_v2v/val"
SPLIT="val"            # train/val/test
MODE="v2v"             # 固定为 v2v
MAX_BATCHES=10
BATCH_SIZE=2

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

python scripts/validate_sat2radar_v2v.py \
  --config "${CONFIG_PATH}" \
  --ckpt "${CKPT_PATH}" \
  --out_dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --mode "${MODE}" \
  --max_batches "${MAX_BATCHES}" \
  --batch_size "${BATCH_SIZE}" \
  --gpu "${CUDA_VISIBLE_DEVICES}"

