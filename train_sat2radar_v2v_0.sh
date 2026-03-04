#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/Sat2Radar-v2v-pretrained-FlowTiTok-XL.py"

# 只用 GPU 0 和 1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

cd /mnt/ssd_1/yghu/Code/FlowTok

accelerate launch \
  --num_processes 1 \
  scripts/train_sat2radar_v2v.py \
  --config="${CONFIG_PATH}"