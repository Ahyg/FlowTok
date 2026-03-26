#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N Sat2Radar_i2i_satlight_validate

set -euo pipefail

export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"
export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"

source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1

FLOWTOK_ROOT="/scratch/kl02/$USER/Projv2v/FlowTok"
CONFIG_PATH="${FLOWTOK_ROOT}/configs/Sat2Radar-i2i-satlight-tokenfusion-FlowTiTok-XL_gadi.py"
# 所有 ckpt 所在目录（每个 step 对应子目录 ${STEP}.ckpt）
CKPT_ROOT="/scratch/kl02/$USER/Projv2v/Experiments/sat2radar_flowtok_run_i2i_satlight_tokenfusion/ckpts"
# 实验根目录（下面会建 validate_i2i_${STEP}/）
RUN_ROOT="/scratch/kl02/$USER/Projv2v/Experiments/sat2radar_flowtok_run_i2i_satlight_tokenfusion"
FILELIST_PATH="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_val_202401_202406_nofilter.pkl"

SPLIT="val"
MODE="i2i"
MAX_BATCHES_METRICS=-1
MAX_BATCHES_IMAGES=10
BATCH_SIZE=4

# 多个训练 step：空格分隔即可；也可在 qsub 前 export STEPS="60000 120000 160000"
STEPS=${STEPS:-"20000 40000 60000 80000 100000 120000 140000 160000 180000 200000"}

cd "${FLOWTOK_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

JOB_LOG_DIR="/scratch/kl02/$USER/Projv2v/job_logs"
mkdir -p "${JOB_LOG_DIR}"
SUMMARY_LOG="${JOB_LOG_DIR}/${PBS_JOBID:-manual}_sat2radar_i2i_satlight_validate_summary.log"
VAL_LOSS_LOG="${JOB_LOG_DIR}/${PBS_JOBID:-manual}_sat2radar_i2i_satlight_validate_val_loss.log"
MAIN_LOG="${JOB_LOG_DIR}/${PBS_JOBID:-manual}_sat2radar_i2i_satlight_validate.log"

{
  echo "[INFO] STEPS=${STEPS}"
  echo "[INFO] CKPT_ROOT=${CKPT_ROOT}"
  echo "[INFO] summary -> ${SUMMARY_LOG}"
  echo "[INFO] val_loss -> ${VAL_LOSS_LOG}"
  : > "${VAL_LOSS_LOG}"
  for STEP in ${STEPS}; do
    CKPT_PATH="${CKPT_ROOT}/${STEP}.ckpt"
    OUTPUT_DIR="${RUN_ROOT}/validate_i2i_${STEP}"
    echo ""
    echo "========== STEP=${STEP} =========="
    echo "[INFO] CKPT_PATH=${CKPT_PATH}"
    echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
    if [[ ! -d "${CKPT_PATH}" ]]; then
      echo "[ERROR] Missing checkpoint directory: ${CKPT_PATH}" >&2
      exit 1
    fi
    mkdir -p "${OUTPUT_DIR}"

    STEP_LOG="${OUTPUT_DIR}/val_step_${STEP}.log"
    rm -f "${STEP_LOG}"

    python -u scripts/validate_sat2radar_v2v.py \
      --config "${CONFIG_PATH}" \
      --filelist_path "${FILELIST_PATH}" \
      --ckpt "${CKPT_PATH}" \
      --out_dir "${OUTPUT_DIR}" \
      --split "${SPLIT}" \
      --mode "${MODE}" \
      --max_batches_metrics "${MAX_BATCHES_METRICS}" \
      --max_batches_images "${MAX_BATCHES_IMAGES}" \
      --batch_size "${BATCH_SIZE}" \
      --gpu "${CUDA_VISIBLE_DEVICES}" \
      2>&1 | tee -a "${SUMMARY_LOG}"

    METRICS_JSON="${OUTPUT_DIR}/validate_metrics.json"
    if [[ -f "${METRICS_JSON}" ]]; then
      python3 -c '
import json, sys
path, step = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
if isinstance(data, list) and data:
    r = data[0]
else:
    r = {}
mse = r.get("mse_mean_radar_0_1", None)
cnt = r.get("mse_frame_count", None)
fss = r.get("fss_accumulated_avg", None)
out_dir = r.get("out_dir", None)
print(f"[VAL] step={step} mse_mean_radar_0_1={mse} n_frames={cnt} fss_accum_avg={fss} out_dir={out_dir}")
' "${METRICS_JSON}" "${STEP}" >> "${VAL_LOSS_LOG}"
    else
      echo "[VAL] step=${STEP} ERROR: missing ${METRICS_JSON}" >> "${VAL_LOSS_LOG}"
    fi
  done
  echo "[OK] All steps done. Per-step metrics: <out_dir>/validate_metrics.json"
} > "${MAIN_LOG}" 2>&1
