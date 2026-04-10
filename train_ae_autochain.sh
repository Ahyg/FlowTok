#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=48:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe

# ── Usage ──────────────────────────────────────────────────────────────
# qsub -N AE_radar  -v AE_CONFIG=radar_flowtitok_ae_bl77_vae_gadi.yaml,AE_MAX_STEP=600000 train_ae_autochain.sh
# qsub -N AE_sat    -v AE_CONFIG=sat_flowtitok_ae_bl77_vae_gadi.yaml,AE_MAX_STEP=600000   train_ae_autochain.sh
# qsub -N AE_sat10  -v AE_CONFIG=sat10ch_flowtitok_ae_bl77_vae_gadi.yaml,AE_MAX_STEP=600000 train_ae_autochain.sh
# ───────────────────────────────────────────────────────────────────────

set -euo pipefail

: "${AE_CONFIG:?ERROR: pass -v AE_CONFIG=<yaml>}"
: "${AE_MAX_STEP:=600000}"

# ── Walltime budget for training code (48h = 172800s) ─────────────
export TRAINING_START_TIME=$(date +%s.%N)
export TRAINING_WALLTIME_SEC=172800

export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"
export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"

FLOWTOK_STAGING="${HF_HOME}/flowtok_staging"
export LPIPS_VGG_PTH="${LPIPS_VGG_PTH:-${FLOWTOK_STAGING}/vgg.pth}"
export VGG16_IMAGENET_PTH="${VGG16_IMAGENET_PTH:-${TORCH_HOME}/hub/checkpoints/vgg16-397923af.pth}"
export CONVNEXT_SMALL_IMAGENET_PTH="${CONVNEXT_SMALL_IMAGENET_PTH:-${TORCH_HOME}/hub/checkpoints/convnext_small-0c510722.pth}"

for _req in "$OPENCLIP_LOCAL_CKPT" "$LPIPS_VGG_PTH" "$VGG16_IMAGENET_PTH" "$CONVNEXT_SMALL_IMAGENET_PTH"; do
  if [[ ! -f "$_req" ]]; then
    echo "ERROR: offline job requires: $_req" >&2
    exit 1
  fi
done

source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate 1d-tokenizer

FLOWTOK_ROOT="/scratch/kl02/$USER/Projv2v/FlowTok"
CONFIG_PATH="${FLOWTOK_ROOT}/configs/${AE_CONFIG}"
SCRIPT_PATH="${FLOWTOK_ROOT}/train_ae_autochain.sh"

mkdir -p "/scratch/kl02/$USER/Projv2v/job_logs"

cd "${FLOWTOK_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}

echo "=== $(date) === Starting AE training: ${AE_CONFIG}, max_step=${AE_MAX_STEP}" >&2

# ── Run training (will auto-resume from latest checkpoint) ─────────
accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  scripts/train_flowtitok_ae.py \
  --config="${CONFIG_PATH}" \
  > "/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_train_ae_$(basename ${AE_CONFIG} .yaml).log" 2>&1 \
  || true   # don't fail on walltime kill (SIGTERM → exit != 0)

# ── Check if training is done ──────────────────────────────────────
EXP_DIR=$(python3 -c "
import yaml, sys
with open('${CONFIG_PATH}') as f:
    cfg = yaml.safe_load(f)
print(cfg['experiment']['output_dir'])
")

LATEST_STEP=0
if [[ -d "$EXP_DIR" ]]; then
  for ckpt in "$EXP_DIR"/checkpoint-*; do
    step="${ckpt##*-}"
    if [[ "$step" =~ ^[0-9]+$ ]] && (( step > LATEST_STEP )); then
      LATEST_STEP=$step
    fi
  done
fi

echo "=== $(date) === Latest checkpoint step: ${LATEST_STEP} / ${AE_MAX_STEP}" >&2

if (( LATEST_STEP >= AE_MAX_STEP )); then
  echo "=== Training complete (step ${LATEST_STEP} >= ${AE_MAX_STEP}). No resubmit. ===" >&2
elif (( LATEST_STEP == 0 )); then
  echo "=== No checkpoint produced (step 0). Likely a config/code error. NOT resubmitting. ===" >&2
else
  echo "=== Training incomplete (step ${LATEST_STEP} < ${AE_MAX_STEP}). Resubmitting... ===" >&2
  cd "${FLOWTOK_ROOT}"
  qsub -N "${PBS_JOBNAME}" \
       -v "AE_CONFIG=${AE_CONFIG},AE_MAX_STEP=${AE_MAX_STEP}" \
       "${SCRIPT_PATH}"
fi
