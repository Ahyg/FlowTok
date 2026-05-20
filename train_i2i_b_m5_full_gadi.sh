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
#PBS -N i2i_b_m5_full
set -uo pipefail
export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"
export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
CFG=$FT/configs/Sat2Radar-i2i-b-m5-2021summer_gadi.py
WD=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_i2i_b_diffusion_tokconcat_2021summer
TARGET=60000
mkdir -p /scratch/kl02/$USER/Projv2v/job_logs
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_i2i_b_m5_full.log
cd $FT
LATEST=$(ls -t "$WD"/ckpts/*.ckpt 2>/dev/null | head -1)
STEP=$(echo "$LATEST" | grep -oP '[0-9]+(?=\.ckpt)'); STEP=${STEP:-0}
echo "[$(date '+%F %T')] m5 full: start step=$STEP target=$TARGET"
if [ "$STEP" -ge "$TARGET" ]; then echo "already done at $STEP"; exit 0; fi
accelerate launch --num_processes 1 scripts/train_sat2radar_v2v.py --config="$CFG" > "$JOBLOG" 2>&1
NEW=$(ls -t "$WD"/ckpts/*.ckpt 2>/dev/null | head -1 | grep -oP '[0-9]+(?=\.ckpt)'); NEW=${NEW:-0}
echo "[$(date '+%F %T')] m5 full: after run step=$NEW"
if [ "$NEW" -lt "$TARGET" ]; then
  cd $FT && qsub train_i2i_b_m5_full_gadi.sh
else
  cd $FT && qsub holdout_test_i2i_b_m5_gadi.sh && echo "[$(date '+%F %T')] m5 hit target, queued holdout test"
fi
