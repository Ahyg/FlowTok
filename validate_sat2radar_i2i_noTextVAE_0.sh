#!/usr/bin/env bash
set -euo pipefail

# Pure visualization (no metrics) for Sat2Radar i2i run (noTextVAE, 方案 A)
CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/Sat2Radar-i2i-noTextVAE-FlowTiTok-XL.py"
CKPT_PATH="/mnt/ssd_1/yghu/Experiments/sat2radar_flowtok_run_i2i_noTextVAE/ckpts/100000.ckpt"
OUTPUT_DIR="/mnt/ssd_1/yghu/Experiments/sat2radar_flowtok_run_i2i_noTextVAE/validate_i2i_100000"

SPLIT="val"            # 可改为 train/test
MODE="i2i"             # i2i / v2v
MAX_BATCHES=10         # -1 = 全部 batch
BATCH_SIZE=8

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python scripts/validate_sat2radar_v2v.py \
  --config "${CONFIG_PATH}" \
  --ckpt "${CKPT_PATH}" \
  --out_dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --mode "${MODE}" \
  --max_batches "${MAX_BATCHES}" \
  --batch_size "${BATCH_SIZE}" \
  --gpu "${CUDA_VISIBLE_DEVICES}"

