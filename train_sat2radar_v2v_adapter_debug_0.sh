#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/Sat2Radar-v2v-adapter-FlowTiTok-XL.py"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_PROCESSES=${NUM_PROCESSES:-$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')}

# Debug controls (can be overridden by env vars)
DEBUG_STEPS=${DEBUG_STEPS:-5}
DEBUG_LOG_EVERY=${DEBUG_LOG_EVERY:-1}
DEBUG_GRAD_EPS_ON=${DEBUG_GRAD_EPS_ON:-1e-12}
DEBUG_GRAD_EPS_OFF=${DEBUG_GRAD_EPS_OFF:-1e-14}

cd /mnt/ssd_1/yghu/Code/FlowTok

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  scripts/train_sat2radar_v2v_debug.py \
  --config="${CONFIG_PATH}" \
  --debug_enabled=true \
  --debug_steps="${DEBUG_STEPS}" \
  --debug_log_every="${DEBUG_LOG_EVERY}" \
  --debug_grad_eps_on="${DEBUG_GRAD_EPS_ON}" \
  --debug_grad_eps_off="${DEBUG_GRAD_EPS_OFF}"
