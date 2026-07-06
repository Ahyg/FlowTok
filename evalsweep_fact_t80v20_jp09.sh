#!/bin/bash
#PBS -P jp09
#PBS -q gpuhopper
#PBS -l walltime=20:00:00
#PBS -l storage=gdata/kl02+scratch/kl02+scratch/jp09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m a
#PBS -N fct_t80_eval
# Per-ckpt VALIDATION full-metric eval, fact-v2v (80/20 split). Reads "val" slot (888 clips)
# of the 8020 pkl. flow-20 sampler (production/benchmark protocol). Locked single-runner,
# skip-existing, drain-loop: evaluates every present un-evaluated ckpt, then exits.
set -uo pipefail
export HF_HOME="/scratch/kl02/$USER/hf_cache"; export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"; export XDG_CACHE_HOME="$HF_HOME"; export HF_HUB_OFFLINE=1; export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
CFG=$FT/configs/Sat2Radar-v2v-fact-t80v20-800k_gadi.py
WD=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_fact_t80v20_800k
VALPKL=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_train80val20_from_cond3nan1_seed42.pkl
OUTD=$WD/val_metrics
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_fct_t80_eval.log
mkdir -p "$OUTD" "$(dirname "$JOBLOG")"
exec > "$JOBLOG" 2>&1
cd $FT
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

LOCK="$OUTD/.evalsweep.lock"
if ! mkdir "$LOCK" 2>/dev/null; then echo "[$(date '+%F %T')] another eval-sweep holds lock -> exit"; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

evaled_any=0
while true; do
  NEWFOUND=0
  STEPS=$(ls -d "$WD"/ckpts/*.ckpt 2>/dev/null | grep -oP '[0-9]+(?=\.ckpt)' | sort -n)
  for N in $STEPS; do
    OJ=$OUTD/step${N}_val_metrics.json
    [ -e "$OJ" ] && continue
    CKPT=$WD/ckpts/${N}.ckpt
    echo "[$(date '+%F %T')] eval VAL step=$N -> $OJ"
    python3 -u scripts/test_sat2radar_v2v.py \
      --config "$CFG" --ckpt "$CKPT" --out_dir "$OUTD/vis_step${N}" \
      --split val --mode v2v --filelist_path "$VALPKL" \
      --max_batches_metrics -1 --max_batches_images 4 \
      --batch_size 8 --metrics_json "$OJ" --gpu "$CUDA_VISIBLE_DEVICES"
    RC=$?
    if [ "$RC" != "0" ] || [ ! -e "$OJ" ]; then echo "  !! FAILED step$N rc=$RC (retry next sweep)"; else NEWFOUND=1; evaled_any=1; fi
  done
  [ "$NEWFOUND" = "0" ] && break
done

echo "[$(date '+%F %T')] fact val-sweep drained (evaled_any=$evaled_any); summarizing"
python3 -u /scratch/kl02/yh0308/Projv2v/Experiments/summarize_val_t80v20.py --dir "$OUTD" --label fact-v2v || echo "summary failed (non-fatal)"
echo "[$(date '+%F %T')] DONE fct_t80_eval"
