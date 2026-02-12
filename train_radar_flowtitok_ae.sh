#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/radar_flowtitok_ae_bl77_vae.yaml"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export WANDB_MODE=disabled

accelerate launch --num_processes 1 /mnt/ssd_1/yghu/Code/FlowTok/scripts/train_flowtitok_ae.py --config="${CONFIG_PATH}"

