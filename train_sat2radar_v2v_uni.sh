#!/bin/bash
# Uni-directional sat→radar FlowTok-XL control run (apples-to-apples vs bidir).
# Same train script (train_sat2radar_v2v_bidir.py) but config.bidir.beta=0, so
# the reverse-direction backward is skipped; effectively a forward-only flow.
set -euo pipefail

source "/home/yghu/miniconda3/etc/profile.d/conda.sh"
conda activate flowtok
export PYTHONUNBUFFERED=1

FLOWTOK_ROOT="${FLOWTOK_ROOT:-/mnt/ssd_1/yghu/Code/FlowTok}"
CONFIG="${CONFIG:-${FLOWTOK_ROOT}/configs/Sat2Radar-v2v-uni-sat10ch-direct-FlowTiTok-XL.py}"

# GPU 2 is free (FlowTok bidir done); use it.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

cd "${FLOWTOK_ROOT}"

echo "[INFO] $(date '+%F %T') start uni-direction training"
echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

accelerate launch --num_processes 1 --mixed_precision bf16 \
  scripts/train_sat2radar_v2v_bidir.py \
  --config="${CONFIG}"

echo "[INFO] $(date '+%F %T') done"
