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
#PBS -N AE_sat10_r4Bftgan

# run4BftGAN-cond1 sat10ch AE: 50k LATE-GAN finetune (250k->300k) of run4Bft_cond1.
# Mirrors the radar GAN finetune (run4ftgan). Resumes from run4Bft_cond1/checkpoint-250000
# (symlinked into the run4Bftgan_cond1 dir); discriminator_start=250000 engages GAN
# immediately. Generator LR floor 1e-5; discriminator 1e-4. discriminator_weight=0.001
# (cross-modal sweep winner). TRAIN filelist = ARCHIVED cond1 pkl (the OLD dataset radar
# used; the live filelists/ path was moved to _archive_filelists_20260605 in the reorg).

set -euo pipefail

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
    echo "ERROR: offline job requires this file (stage on login node): $_req" >&2
    exit 1
  fi
done

source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate 1d-tokenizer

FLOWTOK_ROOT="/scratch/kl02/$USER/Projv2v/FlowTok"
CONFIG_PATH="${FLOWTOK_ROOT}/configs/sat10ch_flowtitok_ae_bl128_vae_scratch_run4Bftgan_cond1_gadi.yaml"

mkdir -p "/scratch/kl02/$USER/Projv2v/job_logs"

cd "${FLOWTOK_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  scripts/train_flowtitok_ae.py \
  --config="${CONFIG_PATH}" \
  > "/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_train_sat10ch_ae_bl128_run4Bftgan_cond1.log" 2>&1
