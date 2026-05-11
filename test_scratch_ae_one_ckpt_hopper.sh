#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=10:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=30GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe

# Generic single-ckpt AE test job, parameterized via env vars passed with `qsub -v`:
#   EXP   = experiment dir name under Experiments/ (also output prefix)
#   STEP  = checkpoint step (numeric, e.g. 100000, 198063)
#   CFG   = absolute or relative-to-FlowTok/configs yaml file name
# Optional:
#   KNAME = step label override (default: 50k/100k/150k/200k or raw step)

set -euo pipefail

if [[ -z "${EXP:-}" || -z "${STEP:-}" || -z "${CFG:-}" ]]; then
  echo "ERROR: must pass EXP, STEP, CFG via qsub -v"
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
conda activate 1d-tokenizer

FLOWTOK_ROOT="/scratch/kl02/$USER/Projv2v/FlowTok"
EXP_ROOT="/scratch/kl02/$USER/Projv2v/Experiments"
TEST_FILELIST="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_test_202407_202507_nofilter.pkl"

FSS_THRESHOLDS="0,5,10,15,20,25,30,35,40,45,50,55,60"
FSS_SCALES="1,2,3,4,5,6,7,8,9,10"

# Resolve config to absolute path
if [[ "$CFG" == /* ]]; then
  CFG_PATH="$CFG"
else
  CFG_PATH="${FLOWTOK_ROOT}/configs/${CFG}"
fi

# Default KNAME mapping
if [[ -z "${KNAME:-}" ]]; then
  case "$STEP" in
    50000)  KNAME="50k"  ;;
    100000) KNAME="100k" ;;
    150000) KNAME="150k" ;;
    200000) KNAME="200k" ;;
    *)      KNAME="$STEP" ;;
  esac
fi

CKPT="${EXP_ROOT}/${EXP}/checkpoint-${STEP}/ema_model/pytorch_model.bin"
OUT_DIR="${EXP_ROOT}/${EXP}/test_${KNAME}_ema"
METRICS="${OUT_DIR}/metrics.json"

if [[ ! -f "$CKPT" ]]; then
  echo "SKIP (no ckpt): ${EXP} @ ${KNAME}  ($CKPT)"
  exit 0
fi
if [[ -f "$METRICS" ]]; then
  echo "SKIP (already done): ${EXP} @ ${KNAME}"
  exit 0
fi

cd "${FLOWTOK_ROOT}"
export CUDA_VISIBLE_DEVICES=0

echo "========== RUN ${EXP} @ ${KNAME} =========="
echo "  config = $CFG_PATH"
echo "  ckpt   = $CKPT"
echo "  out    = $OUT_DIR"

python scripts/test_flowtitok_ae.py \
  --config="${CFG_PATH}" \
  --checkpoint="${CKPT}" \
  --filelist_path="${TEST_FILELIST}" \
  --split=test \
  --out_dir="${OUT_DIR}" \
  --max_batches_images=10 \
  --fss_thresholds="${FSS_THRESHOLDS}" \
  --fss_scales="${FSS_SCALES}"

echo "========== DONE ${EXP} @ ${KNAME} =========="
