#!/bin/bash
#PBS -P kl02
#PBS -q normal
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=8
#PBS -l mem=32GB
#PBS -l jobfs=20GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N build_i2i_nofilter

set -euo pipefail

source "/scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh"
conda activate flowtok
export PYTHONUNBUFFERED=1

FLOWTOK_ROOT="/scratch/kl02/$USER/Projv2v/FlowTok"
DATA_ROOT="/g/data/kl02/yh0308/Data/71"
SAVE_DIR="/g/data/kl02/yh0308/Data/71/filelists"

# qsub -v SPLIT=train|val|test
SPLIT="${SPLIT:-train}"

JOB_LOG_DIR="/scratch/kl02/$USER/Projv2v/job_logs"
mkdir -p "${JOB_LOG_DIR}"
LOG_PATH="${JOB_LOG_DIR}/${PBS_JOBID}_build_i2i_nofilter_${SPLIT}.log"
exec > "${LOG_PATH}" 2>&1

echo "[INFO] job log: ${LOG_PATH}"
echo "[INFO] PBS_JOBID=${PBS_JOBID}, SPLIT=${SPLIT}"

case "${SPLIT}" in
  train)
    START_DATE="20190601"
    END_DATE="20231231"
    OUT_NAME="dataset_filelist_i2i_train_201906_202312_nofilter.pkl"
    SPLIT_RATIO="1,0,0"
    ;;
  val)
    START_DATE="20240101"
    END_DATE="20240630"
    OUT_NAME="dataset_filelist_i2i_val_202401_202406_nofilter.pkl"
    SPLIT_RATIO="0,1,0"
    ;;
  test)
    START_DATE="20240701"
    END_DATE="20250731"
    OUT_NAME="dataset_filelist_i2i_test_202407_202507_nofilter.pkl"
    SPLIT_RATIO="0,0,1"
    ;;
  *)
    echo "[ERROR] Unknown SPLIT=${SPLIT}. Use train|val|test." >&2
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
  --coverage-threshold 0.0

echo "[INFO] $(date '+%F %T') done split=${SPLIT} output=${SAVE_DIR}/${OUT_NAME}"
