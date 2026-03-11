#!/usr/bin/env bash
set -euo pipefail

# 本地单机训练脚本：Sat2Radar i2i（noTextVAE, 方案 A）
CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/Sat2Radar-i2i-noTextVAE-FlowTiTok-XL.py"

# 默认使用单卡，可通过外部覆盖 CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd /mnt/ssd_1/yghu/Code/FlowTok

accelerate launch \
  --num_processes 1 \
  scripts/train_sat2radar_v2v.py \
  --config="${CONFIG_PATH}"

