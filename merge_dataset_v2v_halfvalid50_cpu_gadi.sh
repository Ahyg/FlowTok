#!/bin/bash
#PBS -P kl02
#PBS -q normal
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=1
#PBS -l mem=8GB
#PBS -l jobfs=2GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N merge_v2v_half50

set -euo pipefail

source "/scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh"
conda activate flowtok
export PYTHONUNBUFFERED=1

JOB_LOG_DIR="/scratch/kl02/$USER/Projv2v/job_logs"
mkdir -p "${JOB_LOG_DIR}"
LOG_PATH="${JOB_LOG_DIR}/${PBS_JOBID}_merge_v2v_half50.log"
exec > "${LOG_PATH}" 2>&1

echo "[INFO] job log: ${LOG_PATH}"
echo "[INFO] PBS_JOBID=${PBS_JOBID}"
echo "[INFO] $(date '+%F %T') start merge"

python3 -u - <<'PY'
import pickle
from pathlib import Path

base = Path("/g/data/kl02/yh0308/Data/71/filelists")
train_p = base / "dataset_filelist_v2v_train_201906_202312_halfvalid50_ct005.pkl"
val_p = base / "dataset_filelist_v2v_val_202401_202406_halfvalid50_ct005.pkl"
test_p = base / "dataset_filelist_v2v_test_202407_202507_halfvalid50_ct005.pkl"
out_p = base / "dataset_filelist_v2v_timeblock_201906_202507_halfvalid50_ct005.pkl"

for p in (train_p, val_p, test_p):
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")

with train_p.open("rb") as f:
    train_files, _, _ = pickle.load(f)
with val_p.open("rb") as f:
    _, val_files, _ = pickle.load(f)
with test_p.open("rb") as f:
    _, _, test_files = pickle.load(f)

with out_p.open("wb") as f:
    pickle.dump((train_files, val_files, test_files), f)

print(f"[OK] merged file written: {out_p}")
print(f"[OK] counts: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
PY

echo "[INFO] $(date '+%F %T') merge done"

