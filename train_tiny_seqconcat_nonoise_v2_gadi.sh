#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=02:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=30GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N tiny_sc_nn_v2

export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"
export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"

source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok

FLOWTOK_ROOT="/scratch/kl02/$USER/Projv2v/FlowTok"
CONFIG_PATH="${FLOWTOK_ROOT}/configs/tiny_seqconcat_nonoise_v2_test.py"

mkdir -p "/scratch/kl02/$USER/Projv2v/job_logs"

cd "${FLOWTOK_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  scripts/train_sat2radar_v2v.py \
  --config="${CONFIG_PATH}" \
  > /scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_tiny_seqconcat_nonoise_v2.log 2>&1
