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
#PBS -N Sat2Radar_v2v_contrastive_train

# HuggingFace / OpenCLIP 本地缓存目录，避免计算节点访问外网
export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"
export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
# 显式指定 open_clip 本地权重路径，供 train_sat2radar_v2v.py 使用
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"

# 打印关键信息到 stderr，方便在 log 里确认环境变量是否生效
echo "HF_HOME=$HF_HOME" >&2
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE" >&2
echo "OPENCLIP_LOCAL_CKPT=$OPENCLIP_LOCAL_CKPT" >&2

source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh

# TODO: 按实际环境修改 conda 环境名
conda activate flowtok

# TODO: 按实际路径修改 FlowTok 根目录与 config 路径
FLOWTOK_ROOT="/scratch/kl02/$USER/Projv2v/FlowTok"
CONFIG_PATH="${FLOWTOK_ROOT}/configs/Sat2Radar-v2v-contrastive-FlowTiTok-XL_gadi.py"

cd "${FLOWTOK_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

accelerate launch \
  --num_processes 1 \
  scripts/train_sat2radar_v2v.py \
  --config="${CONFIG_PATH}" \
  > /scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_sat2radar_v2v_contrastive_train.log 2>&1

