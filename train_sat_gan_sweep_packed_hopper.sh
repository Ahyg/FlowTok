#!/bin/bash
#PBS -P kl02
# PACKED on a single H200 (gpuhopper). gpuvolta is DEAD for this env: torch 2.11.0+cu130
# and CUDA 13 dropped Volta (sm_70) support -> every conv2d throws
# "FIND was unable to find an engine to execute this computation" on V100.
# So instead of wasting a whole H200 on one tiny/small cell, we co-locate ALL 5
# sweep cells on one H200: each peaks ~23GB at batch 8, 5x23 = ~115GB < 141GB.
#PBS -q gpuhopper
#PBS -l walltime=48:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=128GB
#PBS -l jobfs=120GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N sat_gan_packed

# Sat10ch AE late-GAN sweep, PACKED: all 5 cells share one H200. Each cell trains
# the sat Variant-B winner recipe (patch8 + kl/10 + perc1.1 + per_channel @128 tokens,
# batch 8, 11ch) for 100k steps with a late GAN (disc_start 60000) at the cell's
# discriminator_weight. Configs: configs/s10_gan_{nogan,w0001,w001,w01,w1}_gadi.yaml.

set -uo pipefail   # NOT -e: one cell crashing must not kill the others

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
LOGDIR="/scratch/kl02/$USER/Projv2v/job_logs"
mkdir -p "$LOGDIR"
cd "${FLOWTOK_ROOT}"
export CUDA_VISIBLE_DEVICES=0          # all cells co-locate on the single H200

CELLS=(s10_gan_nogan s10_gan_w0001 s10_gan_w001 s10_gan_w01 s10_gan_w1)
PORT=29610
declare -A PID2CELL

for cell in "${CELLS[@]}"; do
  cfg="${FLOWTOK_ROOT}/configs/${cell}_gadi.yaml"
  if [[ ! -f "$cfg" ]]; then echo "ERROR: config not found: $cfg (git pull on Gadi?)" >&2; exit 1; fi
  log="${LOGDIR}/${PBS_JOBID}_${cell}.log"
  echo "LAUNCH ${cell} (port ${PORT}) -> ${log}" >&2
  accelerate launch \
    --num_processes 1 \
    --main_process_port "${PORT}" \
    scripts/train_flowtitok_ae.py \
    --config="${cfg}" \
    > "${log}" 2>&1 &
  PID2CELL[$!]="$cell"
  PORT=$((PORT+1))
  sleep 20          # stagger startup so 5 model-build/CUDA-init bursts don't collide
done

echo "All ${#CELLS[@]} cells launched on one H200; waiting..." >&2
rc=0
for pid in "${!PID2CELL[@]}"; do
  if wait "$pid"; then
    echo "DONE  ${PID2CELL[$pid]} (pid $pid)" >&2
  else
    echo "FAIL  ${PID2CELL[$pid]} (pid $pid) exit=$?" >&2
    rc=1
  fi
done
echo "sat_gan_packed ALL FINISHED rc=${rc}" >&2
exit $rc
