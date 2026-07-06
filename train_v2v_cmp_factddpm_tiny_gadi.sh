#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=12:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N v2v_factddpm_tiny
set -uo pipefail
export HF_HOME="/scratch/kl02/$USER/hf_cache"; export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"; export XDG_CACHE_HOME="$HF_HOME"; export HF_HUB_OFFLINE=1; export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
CFG=$FT/configs/Sat2Radar-v2v-cmp-factddpm-tiny_gadi.py
TD=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_cmp_factddpm_tiny
mkdir -p /scratch/kl02/$USER/Projv2v/job_logs "$TD"
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_v2v_factddpm_tiny.log
cd $FT
echo "[$(date '+%F %T')] Ablation2 factddpm tiny: fact-v2v w/ flow->diu-DDPM (cross-attn, pred_x0/linear/ddim/T1000), 32-clip overfit 12k"
accelerate launch --num_processes 1 scripts/train_sat2radar_v2v.py --config="$CFG" > "$JOBLOG" 2>&1 || true
LASTLOSS=$(grep -aoE "'diff_loss': '[0-9.eE+-]+'" "$JOBLOG" | tail -1 | grep -oE '[0-9.eE+-]+' | tail -1)
FIRSTLOSS=$(grep -aoE "'diff_loss': '[0-9.eE+-]+'" "$JOBLOG" | head -1 | grep -oE '[0-9.eE+-]+' | tail -1)
LASTSTEP=$(grep -aoE "'step': '[0-9]+'" "$JOBLOG" | tail -1 | grep -oE '[0-9]+')
echo "[$(date '+%F %T')] factddpm tiny: first_loss=$FIRSTLOSS final step=$LASTSTEP diff_loss=$LASTLOSS"
# NO auto-qsub of full. Radar ~94% dry -> a degenerate all-dry predictor also gets low token-MSE, so
# diff_loss alone does NOT certify the gate. Inspect decoded eval samples ($TD/samples/samples_eval) --
# do the 32 clips' rain STRUCTURE come back? -- then manually qsub train_v2v_cmp_factddpm_600k_full_gadi.sh
{
  echo "step=$LASTSTEP first_loss=$FIRSTLOSS final_loss=$LASTLOSS jobid=$PBS_JOBID date=$(date '+%F %T')"
  echo "ACTION: inspect $TD/samples/samples_eval then manually qsub the full run if structure is reproduced."
} > "$TD/GATE_RESULT.txt"
echo "[$(date '+%F %T')] wrote $TD/GATE_RESULT.txt ; full NOT auto-submitted (manual gate review)."
