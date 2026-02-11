#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for sat2radar FlowTok training (V2V or I2I depending on config).
# Make sure configs/Sat2Radar-Video-XL.py has:
#   - dataset.filelist_path pointing to your v2v filelist (e.g. built with --v2v --clip-length 16)
#   - dataset.num_frames set appropriately (e.g. 16 for fixed-length V2V, or (1,16) for variable T, or 1 for I2I)

CONFIG_PATH="/mnt/ssd_1/yghu/Code/FlowTok/configs/Sat2Radar-Video-XL.py"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

cd /mnt/ssd_1/yghu/Code/FlowTok

python scripts/train_sat2radar_v2v.py \
  --config="${CONFIG_PATH}"

