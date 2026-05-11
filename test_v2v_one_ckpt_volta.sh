#!/bin/bash
#PBS -P kl02
#PBS -q gpuvolta
#PBS -l walltime=08:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=30GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe

# Generic single-ckpt sat2radar test job (i2i or v2v), parameterized via `qsub -v`:
#   EXP    = experiment dir name under Experiments/
#   STEP   = checkpoint step (e.g. 200000, 80000)
#   CONFIG = config python file (relative to configs/ or absolute)
# Optional:
#   MODE       = i2i | v2v   (default: v2v)
#   FILELIST   = filelist override (default: v2v list for v2v, i2i nofilter list for i2i)
#   OUT_SUFFIX = appended to test output dir (e.g. _aligned) to avoid overwriting old runs

set -euo pipefail

if [[ -z "${EXP:-}" || -z "${STEP:-}" || -z "${CONFIG:-}" ]]; then
  echo "ERROR: must pass EXP, STEP, CONFIG via qsub -v"
  exit 2
fi

export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"
export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"

source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok

FLOWTOK_ROOT="/scratch/kl02/$USER/Projv2v/FlowTok"
EXP_ROOT="/scratch/kl02/$USER/Projv2v/Experiments"

if [[ "$CONFIG" == /* ]]; then
  CONFIG_PATH="$CONFIG"
else
  CONFIG_PATH="${FLOWTOK_ROOT}/configs/${CONFIG}"
fi

MODE="${MODE:-v2v}"
if [[ "$MODE" == "i2i" ]]; then
  DEFAULT_FILELIST="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_test_202407_202507_nofilter.pkl"
else
  DEFAULT_FILELIST="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_test_202407_202507.pkl"
fi
FILELIST_PATH="${FILELIST:-$DEFAULT_FILELIST}"

CKPT_PATH="${EXP_ROOT}/${EXP}/ckpts/${STEP}.ckpt"
OUT_DIR="${EXP_ROOT}/${EXP}/test_${MODE}_${STEP}${OUT_SUFFIX:-}"

if [[ ! -d "$CKPT_PATH" && ! -f "$CKPT_PATH" ]]; then
  echo "SKIP (no ckpt): ${EXP} @ ${STEP}  ($CKPT_PATH)"
  exit 0
fi

cd "${FLOWTOK_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p /scratch/kl02/$USER/Projv2v/job_logs

JOB_LOG="/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_${EXP}_test_${MODE}_${STEP}.log"

echo "========== RUN ${MODE} test ${EXP} @ step ${STEP} =========="
echo "  config   = $CONFIG_PATH"
echo "  ckpt     = $CKPT_PATH"
echo "  filelist = $FILELIST_PATH"
echo "  out      = $OUT_DIR"
echo "  log      = $JOB_LOG"

python -u scripts/test_sat2radar_v2v.py \
  --config "${CONFIG_PATH}" \
  --filelist_path "${FILELIST_PATH}" \
  --ckpt "${CKPT_PATH}" \
  --out_dir "${OUT_DIR}" \
  --split test \
  --mode "${MODE}" \
  --max_batches_metrics -1 \
  --max_batches_images 10 \
  --batch_size 4 \
  --gpu "${CUDA_VISIBLE_DEVICES}" \
  > "${JOB_LOG}" 2>&1

echo "========== DONE ${MODE} ${EXP} @ ${STEP} =========="
