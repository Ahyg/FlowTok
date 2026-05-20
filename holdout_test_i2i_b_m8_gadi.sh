#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=6:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N i2i_b_m8_test
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
CFG=$FT/configs/Sat2Radar-i2i-b-m8-2021summer_gadi.py
WD=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_i2i_b_flow_tokconcat_xpred_2021summer
CKPT=$WD/ckpts/60000.ckpt
OUT=$WD/test_holdout_60000
mkdir -p /scratch/kl02/$USER/Projv2v/job_logs "$OUT"
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_i2i_b_m8_test.log
cd $FT
if [ ! -e "$CKPT" ]; then echo "missing $CKPT, abort"; exit 1; fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python -u scripts/test_sat2radar_v2v.py \
  --config "$CFG" \
  --ckpt "$CKPT" \
  --out_dir "$OUT" \
  --split test \
  --mode i2i \
  --max_batches_metrics -1 \
  --max_batches_images 10 \
  --batch_size 8 \
  --metrics_json "$OUT/metrics.json" \
  --gpu "$CUDA_VISIBLE_DEVICES" \
  > "$JOBLOG" 2>&1
echo "[$(date '+%F %T')] m8 holdout test done: $OUT/metrics.json"
