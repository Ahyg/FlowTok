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
#PBS -N Sat2Radar_i2i_satlight_validate

export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"
export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"

source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1

FLOWTOK_ROOT="/scratch/kl02/$USER/Projv2v/FlowTok"
CONFIG_PATH="${FLOWTOK_ROOT}/configs/Sat2Radar-i2i-satlight-tokenfusion-FlowTiTok-XL_gadi.py"
CKPT_PATH="/scratch/kl02/$USER/Projv2v/Experiments/sat2radar_flowtok_run_i2i_satlight_tokenfusion/ckpts/100000.ckpt"
OUTPUT_DIR="/scratch/kl02/$USER/Projv2v/Experiments/sat2radar_flowtok_run_i2i_satlight_tokenfusion/validate_i2i_100000"
FILELIST_PATH="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_val_202401_202406_nofilter.pkl"

SPLIT="val"
MODE="i2i"
MAX_BATCHES=10
BATCH_SIZE=4

cd "${FLOWTOK_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "/scratch/kl02/$USER/Projv2v/job_logs"

python -u scripts/validate_sat2radar_v2v.py \
  --config "${CONFIG_PATH}" \
  --filelist_path "${FILELIST_PATH}" \
  --ckpt "${CKPT_PATH}" \
  --out_dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --mode "${MODE}" \
  --max_batches "${MAX_BATCHES}" \
  --batch_size "${BATCH_SIZE}" \
  --gpu "${CUDA_VISIBLE_DEVICES}" \
  > /scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_sat2radar_i2i_satlight_validate.log 2>&1
