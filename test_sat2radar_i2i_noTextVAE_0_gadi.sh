#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N Sat2Radar_i2i_noTextVAE_test

# HuggingFace / OpenCLIP 本地缓存目录，避免计算节点访问外网
export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"
export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
# 显式指定 open_clip 本地权重路径，供 train_sat2radar_v2v.py 使用
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"

echo "HF_HOME=$HF_HOME" >&2
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE" >&2

source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh

# TODO: 按实际环境修改 conda 环境名
conda activate flowtok

# FlowTok 根目录与 config / ckpt / 输出路径（Gadi）
FLOWTOK_ROOT="/scratch/kl02/$USER/Projv2v/FlowTok"
CONFIG_PATH="${FLOWTOK_ROOT}/configs/Sat2Radar-i2i-noTextVAE-FlowTiTok-XL_gadi.py"
CKPT_PATH="/scratch/kl02/$USER/Projv2v/Experiments/sat2radar_flowtok_run_i2i_noTextVAE/ckpts/300000.ckpt"
OUTPUT_DIR="/scratch/kl02/$USER/Projv2v/Experiments/sat2radar_flowtok_run_i2i_noTextVAE/test_i2i_300000"

SPLIT="test"              # train/val/test，可按需要改
MODE="i2i"                # i2i：单帧卫星 -> 单帧雷达
MAX_BATCHES_METRICS=-1    # -1 = 用整个 split 做指标
MAX_BATCHES_IMAGES=10     # 只保存前 N 个 batch 的图像/PNG
BATCH_SIZE=8              # i2i 显存压力更小，可以适当放大

cd "${FLOWTOK_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python scripts/test_sat2radar_v2v.py \
  --config "${CONFIG_PATH}" \
  --ckpt "${CKPT_PATH}" \
  --out_dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --mode "${MODE}" \
  --max_batches_metrics "${MAX_BATCHES_METRICS}" \
  --max_batches_images "${MAX_BATCHES_IMAGES}" \
  --batch_size "${BATCH_SIZE}" \
  --gpu "${CUDA_VISIBLE_DEVICES}" \
  > /scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_sat2radar_i2i_noTextVAE_test.log 2>&1

