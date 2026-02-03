#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/sat_flowtok_ae_bl77_vae.yaml"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export WANDB_MODE=disabled

accelerate launch --num_processes 2 /mnt/ssd_1/yghu/Code/FlowTok/scripts/train_flowtitok_ae.py --config="${CONFIG_PATH}"

