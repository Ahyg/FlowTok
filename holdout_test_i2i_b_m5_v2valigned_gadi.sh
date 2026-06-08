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
#PBS -N i2i_b_m5_v2vali
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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
FT=/scratch/kl02/$USER/Projv2v/FlowTok
CFG=$FT/configs/Sat2Radar-i2i-b-m5-2021summer_gadi.py
WD=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_i2i_b_diffusion_tokconcat_2021summer
CKPT=$WD/ckpts/60000.ckpt
# v2v-aligned test set: 3856 frames (= 241 v2v test clips x 16). Original metrics
# under test_holdout_60000 stay untouched.
OUT=$WD/test_holdout_60000_v2valigned
FILELIST=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_v2valigned_2021summer_merged.pkl
mkdir -p /scratch/kl02/$USER/Projv2v/job_logs "$OUT"
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_i2i_b_m5_v2vali.log
cd $FT
if [ ! -e "$CKPT" ]; then echo "missing $CKPT, abort"; exit 1; fi
if [ ! -e "$FILELIST" ]; then echo "missing $FILELIST, abort"; exit 1; fi
python -u scripts/test_sat2radar_v2v.py \
  --config "$CFG" \
  --ckpt "$CKPT" \
  --filelist_path "$FILELIST" \
  --out_dir "$OUT" \
  --split test \
  --mode i2i \
  --max_batches_metrics -1 \
  --max_batches_images 10 \
  --batch_size 8 \
  --kid_subsets 50 \
  --kid_subset_size 100 \
  --metrics_json "$OUT/metrics.json" \
  --gpu "$CUDA_VISIBLE_DEVICES" \
  > "$JOBLOG" 2>&1
echo "[$(date '+%F %T')] m5 v2v-aligned holdout done: $OUT/metrics.json"
