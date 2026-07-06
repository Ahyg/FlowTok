#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=02:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m a
#PBS -N x1snap_ovf
# CLEAN diagnostic: snap099 vs default on the 32 OVERFIT clips (not held-out).
# If snap shows structure on the memorized clips -> sampler was the culprit (rescuable).
# If snap is ALSO uniform on memorized clips -> the DiT can't generate from sat cross-attn (conditioning).
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
PKL=$TD/../sat2radar_flowtok_v2v_cmp_xattn_tiny/dataset_filelist.pkl   # 32 overfit clips
# pick latest ckpt unless STEP given
if [ -z "${STEP:-}" ]; then STEP=$(ls -d "$TD"/ckpts/*.ckpt 2>/dev/null | grep -oP '[0-9]+(?=\.ckpt)' | sort -n | tail -1); fi
CKPT=$TD/ckpts/${STEP}.ckpt
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_x1snap_overfit.log
cd $FT
if [ ! -e "$CKPT" ]; then echo "missing $CKPT"; exit 1; fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "[$(date '+%F %T')] overfit-clip x1snap diag, step=$STEP ckpt=$CKPT" | tee -a "$JOBLOG"

run () {  # tag  snap_t
  local OUT=$TD/x1snap_OVERFIT_${STEP}/$1
  mkdir -p "$OUT"
  python3 -u scripts/test_sat2radar_v2v.py \
    --config "$CFG" --ckpt "$CKPT" --out_dir "$OUT" \
    --split test --mode v2v --filelist_path "$PKL" \
    --max_batches_metrics -1 --max_batches_images 6 \
    --batch_size 8 --x1_snap_t "$2" \
    --metrics_json "$OUT/metrics.json" --gpu "$CUDA_VISIBLE_DEVICES" >> "$JOBLOG" 2>&1
  echo "[$(date '+%F %T')] $1 done" | tee -a "$JOBLOG"
}
run "snap099" 0.99
run "default" -1.0
echo "[$(date '+%F %T')] OVERFIT x1snap diag done -> $TD/x1snap_OVERFIT_${STEP}/{snap099,default}"
