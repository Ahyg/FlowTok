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
#PBS -N v2v_m8align_full
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
CFG=$FT/configs/Sat2Radar-v2v-m8align-tokconcat-xpred-FlowTiTok-XL_gadi.py
WD=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_m8align_tokconcat_xpred_full
TARGET=200000
WALL_SEC=$((47*3600))
mkdir -p /scratch/kl02/$USER/Projv2v/job_logs "$WD/ckpts"
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_v2v_m8align_full.log
cd $FT

latest_step() {
  ls -d "$WD"/ckpts/*.ckpt 2>/dev/null | grep -oP '[0-9]+(?=\.ckpt)' | sort -n | tail -1
}
STEP=$(latest_step); STEP=${STEP:-0}
echo "[$(date '+%F %T')] v2v m8align full: start step=$STEP target=$TARGET (resume auto from ckpt_root)"
if [ "$STEP" -ge "$TARGET" ]; then echo "already done at $STEP"; exit 0; fi

timeout -s TERM --kill-after=180 "$WALL_SEC" \
  accelerate launch --num_processes 1 scripts/train_sat2radar_v2v.py --config="$CFG" > "$JOBLOG" 2>&1
RC=$?
NEW=$(latest_step); NEW=${NEW:-0}
echo "[$(date '+%F %T')] v2v m8align full: after run rc=$RC step=$NEW (124=timeout->resubmit)"

if [ "$NEW" -ge "$TARGET" ]; then
  echo "[$(date '+%F %T')] v2v m8align full: TARGET reached at $NEW. Done."
  rm -f "$WD/RESUBMIT_STALLED"
elif [ "$RC" = "124" ] || [ "$NEW" -gt "$STEP" ]; then
  echo "[$(date '+%F %T')] v2v m8align full: progress $STEP->$NEW (rc=$RC), re-qsub"
  rm -f "$WD/RESUBMIT_STALLED"
  cd $FT && qsub train_v2v_m8align_tokconcat_xpred_full_gadi.sh
else
  echo "[$(date '+%F %T')] v2v m8align full: NO progress (step stuck at $NEW, rc=$RC) -> NOT resubmitting"
  echo "$(date '+%F %T') stuck_step=$NEW rc=$RC jobid=$PBS_JOBID see=$JOBLOG" > "$WD/RESUBMIT_STALLED"
fi
