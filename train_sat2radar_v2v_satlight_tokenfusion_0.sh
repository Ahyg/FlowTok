#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/Sat2Radar-v2v-satlight-tokenfusion-FlowTiTok-XL.py"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
NUM_PROCESSES=${NUM_PROCESSES:-$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')}

cd /mnt/ssd_1/yghu/Code/FlowTok

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  scripts/train_sat2radar_v2v.py \
  --config="${CONFIG_PATH}"

