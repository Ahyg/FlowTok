#!/bin/bash
# FlowTok-S (~30M) uni-direction control: tests whether the XL (698M) failure
# was caused by capacity/data mismatch. Same train data, same 50k steps,
# same batch_size, same flow matching loss; only model size differs.
set -euo pipefail

source "/home/yghu/miniconda3/etc/profile.d/conda.sh"
conda activate flowtok
export PYTHONUNBUFFERED=1

FLOWTOK_ROOT="${FLOWTOK_ROOT:-/mnt/ssd_1/yghu/Code/FlowTok}"
CONFIG="${CONFIG:-${FLOWTOK_ROOT}/configs/Sat2Radar-v2v-uni-sat10ch-direct-FlowTiTok-S.py}"

# GPU 0 is free; FlowTok-XL uni runs on GPU 2 in parallel.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${FLOWTOK_ROOT}"

echo "[INFO] $(date '+%F %T') start FlowTok-S uni training"
echo "[INFO] CONFIG=${CONFIG}  GPU=${CUDA_VISIBLE_DEVICES}"

accelerate launch --num_processes 1 --mixed_precision bf16 \
  scripts/train_sat2radar_v2v_bidir.py \
  --config="${CONFIG}"

echo "[INFO] $(date '+%F %T') done"
