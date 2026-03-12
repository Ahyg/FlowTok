#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=04:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N Sat2Radar_v2v_adapter_debug

# HuggingFace / OpenCLIP 本地缓存目录，避免计算节点访问外网
export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"
export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"

echo "HF_HOME=$HF_HOME" >&2
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE" >&2
echo "OPENCLIP_LOCAL_CKPT=$OPENCLIP_LOCAL_CKPT" >&2

source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok

FLOWTOK_ROOT="/scratch/kl02/$USER/Projv2v/FlowTok"
CONFIG_PATH="${FLOWTOK_ROOT}/configs/Sat2Radar-v2v-adapter-FlowTiTok-XL_gadi.py"

cd "${FLOWTOK_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}

# Debug 控制参数（可通过 qsub 前 export 覆盖）
DEBUG_STEPS=${DEBUG_STEPS:-5}
DEBUG_LOG_EVERY=${DEBUG_LOG_EVERY:-1}
DEBUG_GRAD_EPS_ON=${DEBUG_GRAD_EPS_ON:-1e-12}
DEBUG_GRAD_EPS_OFF=${DEBUG_GRAD_EPS_OFF:-1e-14}

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  scripts/train_sat2radar_v2v_debug.py \
  --config="${CONFIG_PATH}" \
  --debug_enabled=true \
  --debug_steps="${DEBUG_STEPS}" \
  --debug_log_every="${DEBUG_LOG_EVERY}" \
  --debug_grad_eps_on="${DEBUG_GRAD_EPS_ON}" \
  --debug_grad_eps_off="${DEBUG_GRAD_EPS_OFF}" \
  > /scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_sat2radar_v2v_adapter_debug.log 2>&1

