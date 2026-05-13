#!/bin/bash
set -euo pipefail

source "/home/yghu/miniconda3/etc/profile.d/conda.sh"
conda activate flowtok
export PYTHONUNBUFFERED=1

FLOWTOK_ROOT="${FLOWTOK_ROOT:-/mnt/ssd_1/yghu/Code/FlowTok}"
DATA_ROOT="${DATA_ROOT:-/mnt/ssd_1/yghu/Data/71_3m}"
SAVE_DIR="${SAVE_DIR:-${DATA_ROOT}/filelists}"

# Only train uses the ct005 filter (val/test stay nofilter on lab).
SPLIT="${SPLIT:-train}"

case "${SPLIT}" in
  train)
    START_DATE="20210501"
    END_DATE="20211031"
    OUT_NAME="dataset_filelist_i2i_train_202105_202110_ct005.pkl"
    SPLIT_RATIO="1,0,0"
    ;;
  *)
    echo "[ERROR] Unknown SPLIT=${SPLIT}. Only 'train' supported for ct005 locally." >&2
    exit 1
    ;;
esac

mkdir -p "${SAVE_DIR}"
cd "${FLOWTOK_ROOT}"

echo "[INFO] $(date '+%F %T') start split=${SPLIT} range=${START_DATE}-${END_DATE}"
echo "[INFO] split ratio: ${SPLIT_RATIO}"

python3 -u "scripts/build_dataset.py" \
  --data-root "${DATA_ROOT}" \
  --save-dir "${SAVE_DIR}" \
  --dataset-pkl-name "${OUT_NAME}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --split-mode days \
  --split-ratio "${SPLIT_RATIO}" \
  --coverage-threshold 0.05

echo "[INFO] $(date '+%F %T') done split=${SPLIT} output=${SAVE_DIR}/${OUT_NAME}"
