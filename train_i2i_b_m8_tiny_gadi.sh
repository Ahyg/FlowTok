#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=06:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N i2i_b_m8_tiny
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
CFG=$FT/configs/Sat2Radar-i2i-b-m8-tiny_gadi.py
TD=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_i2i_b_flow_tokconcat_xpred_2021summer_tiny
mkdir -p /scratch/kl02/$USER/Projv2v/job_logs "$TD"
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_i2i_b_m8_tiny.log
cd $FT
echo "[$(date '+%F %T')] m8 tiny: start"
accelerate launch --num_processes 1 scripts/train_sat2radar_v2v.py --config="$CFG" > "$JOBLOG" 2>&1 || true
# Flow + token_concat (x1 target = radar tokens): M4 reached 0.052 at 5k -> use
# lenient gate diff_loss<0.3 AND step>=4900 AND not NaN.
LASTLOSS=$(grep -aoE "'diff_loss': '[0-9.eE+-]+'" "$JOBLOG" | tail -1 | grep -oE '[0-9.eE+-]+' | tail -1)
LASTSTEP=$(grep -aoE "'step': '[0-9]+'" "$JOBLOG" | tail -1 | grep -oE '[0-9]+')
echo "[$(date '+%F %T')] m8 tiny: final step=$LASTSTEP diff_loss=$LASTLOSS"
PASS=0
if [ -n "$LASTLOSS" ] && [ -n "$LASTSTEP" ] && [ "$LASTSTEP" -ge 4900 ] \
   && awk -v v="$LASTLOSS" 'BEGIN{exit !(v+0<0.3 && v+0==v+0)}'; then
  PASS=1
fi
if [ "$PASS" = "1" ]; then
  echo "GATE PASS -> submitting m8 full"
  rm -f "$TD/TINY_FAIL"
  cd $FT && qsub train_i2i_b_m8_full_gadi.sh
else
  echo "GATE FAIL (step=$LASTSTEP loss=$LASTLOSS) -> m8 full NOT submitted"
  echo "$(date '+%F %T') step=$LASTSTEP loss=$LASTLOSS jobid=$PBS_JOBID" > "$TD/TINY_FAIL"
fi
