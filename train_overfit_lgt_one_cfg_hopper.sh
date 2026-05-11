#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=04:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=30GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe

# Generic single-config overfit-on-tiny-lgt AE training job.
# Pass via `qsub -v`:
#   CFG = absolute or relative-to-FlowTok/configs yaml file name

set -euo pipefail

if [[ -z "${CFG:-}" ]]; then
  echo "ERROR: must pass CFG via qsub -v"
  exit 2
fi

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
    echo "ERROR: offline job requires this file: $_req" >&2
    exit 1
  fi
done

source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate 1d-tokenizer

FLOWTOK_ROOT="/scratch/kl02/$USER/Projv2v/FlowTok"
if [[ "$CFG" == /* ]]; then
  CFG_PATH="$CFG"
else
  CFG_PATH="${FLOWTOK_ROOT}/configs/${CFG}"
fi

if [[ ! -f "$CFG_PATH" ]]; then
  echo "ERROR: config not found: $CFG_PATH" >&2
  exit 2
fi

mkdir -p "/scratch/kl02/$USER/Projv2v/job_logs"

cd "${FLOWTOK_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}

CFG_BASE="$(basename "$CFG_PATH" .yaml)"

echo "========== RUN overfit-lgt ${CFG_BASE} =========="
accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  scripts/train_flowtitok_ae.py \
  --config="${CFG_PATH}" \
  > "/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_${CFG_BASE}.log" 2>&1
echo "========== DONE ${CFG_BASE} =========="
