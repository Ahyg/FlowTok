#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/Sat2Radar-v2v-adapter-FlowTiTok-XL.py"

# 默认使用 2 卡，可通过外部覆盖 CUDA_VISIBLE_DEVICES（例如 "0,1,2,3"）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# 自动根据可见 GPU 数设置进程数，也可外部手动覆盖 NUM_PROCESSES
NUM_PROCESSES=${NUM_PROCESSES:-$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')}

cd /mnt/ssd_1/yghu/Code/FlowTok

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  scripts/train_sat2radar_v2v.py \
  --config="${CONFIG_PATH}"

