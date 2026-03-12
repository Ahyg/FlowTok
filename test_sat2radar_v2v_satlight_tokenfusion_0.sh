#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/Sat2Radar-v2v-satlight-tokenfusion-FlowTiTok-XL.py"
CKPT_PATH="/mnt/ssd_1/yghu/Experiments/sat2radar_flowtok_run_v2v_satlight_tokenfusion/ckpts/100000.ckpt"
OUTPUT_DIR="/mnt/ssd_1/yghu/Experiments/sat2radar_flowtok_run_v2v_satlight_tokenfusion/test_v2v_100000"

SPLIT="test"
MODE="v2v"
MAX_BATCHES_METRICS=-1
MAX_BATCHES_IMAGES=10
BATCH_SIZE=4

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

