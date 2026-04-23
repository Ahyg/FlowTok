#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=30GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N v7_det_vs_sto

export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"
export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"

source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok

FLOWTOK_ROOT="/scratch/kl02/$USER/Projv2v/FlowTok"
CONFIG_PATH="${FLOWTOK_ROOT}/configs/tiny_tokenconcat_test.py"
CKPT_PATH="/scratch/kl02/yh0308/Projv2v/Experiments/tiny_tokenconcat_recon_v7/ckpts/5000.ckpt"
OUT_DIR="/scratch/kl02/yh0308/Projv2v/Experiments/tiny_tokenconcat_recon_v7/diagnostic_det_vs_sto"

mkdir -p "/scratch/kl02/$USER/Projv2v/job_logs"

cd "${FLOWTOK_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python scripts/diagnose_v7_det_vs_sto.py \
  --config="${CONFIG_PATH}" \
  --ckpt="${CKPT_PATH}" \
  --out_dir="${OUT_DIR}" \
  --batch_size=4 \
  --n_stochastic=4 \
  --split=train \
  --seed=42 \
  > /scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_v7_det_vs_sto.log 2>&1
