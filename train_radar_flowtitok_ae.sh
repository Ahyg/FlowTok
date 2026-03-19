#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/radar_flowtitok_ae_bl77_vae.yaml"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=disabled
# Optional: set local CLIP checkpoint to avoid network access on compute nodes
# export OPENCLIP_LOCAL_CKPT=/path/to/ViT-L-14-336-openai.pt

NUM_PROCESSES=${NUM_PROCESSES:-1}

accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --mixed_precision bf16 \
    /mnt/ssd_1/yghu/Code/FlowTok/scripts/train_flowtitok_ae.py \
    --config="${CONFIG_PATH}"

