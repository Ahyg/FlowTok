#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=03:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m a
#PBS -N x1snap_pixf
# Diagnostic: does reassembling the DiT's DIRECT x1_hat tokens (x1_snap_t=0.99) recover structure,
# vs the default telescoping-average sampler (x1_snap_t=-1, which gives uniform)? Same ckpt, same
# clips, side-by-side. Eval-only, no retrain. STEP env selects the tiny ckpt (default 4000).
set -uo pipefail
export HF_HOME="/scratch/kl02/$USER/hf_cache"; export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"; export XDG_CACHE_HOME="$HF_HOME"; export HF_HUB_OFFLINE=1; export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
CFG=$FT/configs/Sat2Radar-v2v-cmp-pixfact-tiny_gadi.py
TD=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_cmp_pixfact_tiny
STEP="${STEP:-4000}"
CKPT=$TD/ckpts/${STEP}.ckpt
PKL_CT=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_test_202407_202507_cond3nan1_clip16_p005_seed42.pkl
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_x1snap_pixfact.log
cd $FT
if [ ! -e "$CKPT" ]; then echo "missing $CKPT"; exit 1; fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

run () {  # tag  snap_t
  local OUT=$TD/x1snap_diag_${STEP}/$1
  mkdir -p "$OUT"
  echo "[$(date '+%F %T')] pixfact $1 (x1_snap_t=$2) step=$STEP -> $OUT" | tee -a "$JOBLOG"
  python3 -u scripts/test_sat2radar_v2v.py \
    --config "$CFG" --ckpt "$CKPT" --out_dir "$OUT" \
    --split test --mode v2v --filelist_path "$PKL_CT" \
    --max_batches_metrics 8 --max_batches_images 4 \
    --batch_size 8 --x1_snap_t "$2" \
    --metrics_json "$OUT/metrics.json" --gpu "$CUDA_VISIBLE_DEVICES" >> "$JOBLOG" 2>&1
}

run "snap099"  0.99     # reassemble the DiT's direct x1_hat tokens
run "default"  -1.0     # telescoping-average sampler (baseline, expect uniform)
echo "[$(date '+%F %T')] x1snap diag done: compare $TD/x1snap_diag_${STEP}/{snap099,default}/*.png"
