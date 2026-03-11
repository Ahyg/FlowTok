#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/Sat2Radar-v2v-newposemb-FlowTiTok-XL.py"

# 默认使用单卡，可通过外部覆盖 CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

cd /mnt/ssd_1/yghu/Code/FlowTok

accelerate launch \
  --num_processes 1 \
  scripts/train_sat2radar_v2v.py \
  --config="${CONFIG_PATH}"

