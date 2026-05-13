#!/bin/bash
set -euo pipefail

source "/home/yghu/miniconda3/etc/profile.d/conda.sh"
conda activate flowtok
export PYTHONUNBUFFERED=1

FLOWTOK_ROOT="${FLOWTOK_ROOT:-/mnt/ssd_1/yghu/Code/FlowTok}"
CONFIG="${CONFIG:-${FLOWTOK_ROOT}/configs/Sat2Radar-v2v-bidir-sat10ch-direct-FlowTiTok-XL.py}"

# Single GPU on lab2 (default GPU 2 which was empty when this was authored).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

cd "${FLOWTOK_ROOT}"

# Workdir = config.workdir; output.log is appended by the training script.
echo "[INFO] $(date '+%F %T') start bidir training"
echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

accelerate launch --num_processes 1 --mixed_precision bf16 \
  scripts/train_sat2radar_v2v_bidir.py \
  --config="${CONFIG}"

echo "[INFO] $(date '+%F %T') done"
