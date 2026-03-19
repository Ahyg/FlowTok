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
#PBS -N Sat2Radar_v2v_contrastive_validate

# HuggingFace / OpenCLIP 本地缓存目录，避免计算节点访问外网
export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"
export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1

echo "HF_HOME=$HF_HOME" >&2
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE" >&2

source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh

# TODO: 按实际环境修改 conda 环境名
conda activate flowtok

# TODO: 按实际路径修改 FlowTok 根目录与 config / ckpt / 输出路径
FLOWTOK_ROOT="/scratch/kl02/$USER/Projv2v/FlowTok"
CONFIG_PATH="${FLOWTOK_ROOT}/configs/Sat2Radar-v2v-contrastive-FlowTiTok-XL_gadi.py"
CKPT_PATH="/scratch/kl02/$USER/Projv2v/Experiments/sat2radar_flowtok_run_v2v_contrastive/ckpts/100000.ckpt"
OUTPUT_DIR="/scratch/kl02/$USER/Projv2v/Experiments/sat2radar_flowtok_run_v2v_contrastive/validate_v2v_100000"
FILELIST_PATH="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_val_202401_202406.pkl"

SPLIT="val"            # 可改为 train/test
MODE="v2v"             # i2i / v2v
MAX_BATCHES=10         # -1 = 全部 batch
BATCH_SIZE=4

cd "${FLOWTOK_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python scripts/validate_sat2radar_v2v.py \
  --config "${CONFIG_PATH}" \
  --filelist_path "${FILELIST_PATH}" \
  --ckpt "${CKPT_PATH}" \
  --out_dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --mode "${MODE}" \
  --max_batches "${MAX_BATCHES}" \
  --batch_size "${BATCH_SIZE}" \
  --gpu "${CUDA_VISIBLE_DEVICES}" \
  > /scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_sat2radar_v2v_contrastive_validate.log 2>&1

