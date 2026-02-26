#!/usr/bin/env bash
set -euo pipefail

# Test FlowTiTok image tokenizer reconstruction on fused radar npy data
# (e.g. /mnt/ssd_1/yghu/Data/71_3m), using the last channel (radar, 0-60 dBZ),
# normalized as in SatelliteRadarNpyDataset and visualized with HomeyerRainbow.
#
# Assumes you already have:
#   - Radar dataset at: /mnt/ssd_1/yghu/Data/71_3m
#   - dataset_filelist_i2i.pkl under that directory
#   - Tokenizer ckpt at: /mnt/ssd_1/yghu/Data/flowtok_ckpts/FlowTiTok_512.bin
# And that you've activated the conda env:
#   conda activate flowtok
#
# Usage:
#   bash run_flowtitok_radar_recon.sh
#
# Optional overrides:
#   CUDA_VISIBLE_DEVICES=0 bash run_flowtitok_radar_recon.sh
#   DATA_ROOT=/path/to/71_3m CKPT_PATH=/path/to/FlowTiTok_512.bin OUT_DIR=/path/to/out bash run_flowtitok_radar_recon.sh

ROOT_DIR="/mnt/ssd_1/yghu/Code/FlowTok"

DATA_PARENT_DIR="/mnt/ssd_1/yghu/Data"
DATA_ROOT="${DATA_ROOT:-${DATA_PARENT_DIR}/71_3m}"

FILELIST="${FILELIST:-${DATA_ROOT}/dataset_filelist_i2i.pkl}"

CKPT_DIR="${CKPT_DIR:-${DATA_PARENT_DIR}/flowtok_ckpts}"
CKPT_PATH="${CKPT_PATH:-${CKPT_DIR}/FlowTiTok_512.bin}"

CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/FlowTok-XL-Stage3.py}"

OUT_DIR="${OUT_DIR:-/mnt/ssd_1/yghu/Experiments/flowtok_tokenizer_radar71_recon}"
SPLIT="${SPLIT:-val}"

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES_METRICS="${MAX_BATCHES_METRICS:-50}"
MAX_BATCHES_IMAGES="${MAX_BATCHES_IMAGES:-10}"

# Whether to use CLIP-based text guidance.
# 默认开启；如需关闭，在外部设置 USE_TEXT=false 或 0。
USE_TEXT="${USE_TEXT:-true}"

# FSS settings for radar (dBZ), following test_radar_flowtitok_ae.sh defaults.
FSS_THRESHOLDS="${FSS_THRESHOLDS:-0,5,10,15,20,25,30,35,40,45,50,55,60}"
FSS_SCALES="${FSS_SCALES:-1,2,3,4,5,6,7,8,9,10}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[INFO] ROOT_DIR   = ${ROOT_DIR}"
echo "[INFO] DATA_ROOT  = ${DATA_ROOT}"
echo "[INFO] FILELIST   = ${FILELIST}"
echo "[INFO] CKPT_PATH  = ${CKPT_PATH}"
echo "[INFO] CONFIG     = ${CONFIG_PATH}"
echo "[INFO] OUT_DIR    = ${OUT_DIR}"
echo "[INFO] SPLIT      = ${SPLIT}"
echo "[INFO] BATCH_SIZE = ${BATCH_SIZE}"
echo "[INFO] USE_TEXT   = ${USE_TEXT}"
echo "[INFO] FSS_THRESHOLDS = ${FSS_THRESHOLDS}"
echo "[INFO] FSS_SCALES     = ${FSS_SCALES}"

mkdir -p "${OUT_DIR}"

CMD=(python "${ROOT_DIR}/scripts/test_flowtitok_tokenizer_radar.py"
  --config "${CONFIG_PATH}"
  --ckpt "${CKPT_PATH}"
  --data_root "${DATA_ROOT}"
  --filelist "${FILELIST}"
  --split "${SPLIT}"
  --out_dir "${OUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_batches_metrics "${MAX_BATCHES_METRICS}"
  --max_batches_images "${MAX_BATCHES_IMAGES}"
  --fss_thresholds "${FSS_THRESHOLDS}"
  --fss_scales "${FSS_SCALES}"
)

if [ "${USE_TEXT}" = "true" ] || [ "${USE_TEXT}" = "1" ]; then
  CMD+=(--use_text)
fi

"${CMD[@]}"

echo "[DONE] Radar reconstruction metrics and images saved to: ${OUT_DIR}"

