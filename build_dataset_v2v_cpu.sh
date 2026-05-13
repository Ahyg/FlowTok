#!/bin/bash
set -euo pipefail

source "/home/yghu/miniconda3/etc/profile.d/conda.sh"
conda activate flowtok
export PYTHONUNBUFFERED=1

FLOWTOK_ROOT="${FLOWTOK_ROOT:-/mnt/ssd_1/yghu/Code/FlowTok}"
DATA_ROOT="${DATA_ROOT:-/mnt/ssd_1/yghu/Data/71_3m}"
SAVE_DIR="${SAVE_DIR:-${DATA_ROOT}/filelists}"

# Local val/test: no filter. val_small = first week of June 2024 (quick eval).
SPLIT="${SPLIT:-val}"

case "${SPLIT}" in
  val)
    START_DATE="20240601"
    END_DATE="20240630"
    OUT_NAME="dataset_filelist_v2v_val_202406.pkl"
    SPLIT_RATIO="0,1,0"
    ;;
  val_small)
    START_DATE="20240601"
    END_DATE="20240607"
    OUT_NAME="dataset_filelist_v2v_val_202406w1.pkl"
    SPLIT_RATIO="0,1,0"
    ;;
  test)
    START_DATE="20240701"
    END_DATE="20240731"
    OUT_NAME="dataset_filelist_v2v_test_202407.pkl"
    SPLIT_RATIO="0,0,1"
    ;;
  *)
    echo "[ERROR] Unknown SPLIT=${SPLIT}. Use val|val_small|test." >&2
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
  --coverage-threshold 0.0 \
  --v2v \
  --clip-length 16

echo "[INFO] $(date '+%F %T') done split=${SPLIT} output=${SAVE_DIR}/${OUT_NAME}"
