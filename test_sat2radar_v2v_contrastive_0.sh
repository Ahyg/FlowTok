#!/usr/bin/env bash
set -euo pipefail

# Config & checkpoint for Sat2Radar v2v run (textVAE contrastive)
CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/Sat2Radar-v2v-contrastive-FlowTiTok-XL.py"
CKPT_PATH="/mnt/ssd_1/yghu/Experiments/sat2radar_flowtok_run_v2v_contrastive/ckpts/100000.ckpt"
OUTPUT_DIR="/mnt/ssd_1/yghu/Experiments/sat2radar_flowtok_run_v2v_contrastive/test_v2v_100000"

SPLIT="test"              # train/val/test，可按需要改
MODE="v2v"                # 固定为 v2v
MAX_BATCHES_METRICS=-1    # -1 = 用整个 split 做指标
MAX_BATCHES_IMAGES=10     # 只保存前 N 个 batch 的图像/GIF
BATCH_SIZE=4              # v2v 显存压力更大，默认小一点

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

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

