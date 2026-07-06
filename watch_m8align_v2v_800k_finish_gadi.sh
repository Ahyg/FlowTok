#!/bin/bash
#PBS -P kl02
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l jobfs=1GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m a
#PBS -N watch_m8a800k
# Hands-off finisher for FlowTok m8align-v2v @ 800k (the sole unfinished 9-model cell).
# Polls for 800000.ckpt (written by training job 172728490 chain, kl02). When present:
#   1. matrix_cell_ft (kl02): canonical holdout for ct+nofilt -> dumps fp32 dBZ pred arrays + metrics.json
#   2. crossmode (kl02, afterok on #1): adds FID family {fid,sfid,kid} to the v2v cell (GT auto-built inline)
# Training's plain auto-holdout is suppressed (.holdout_queued pre-touched) so matrix_cell_ft is the SOLE
# producer of the 800k cell, matching the other 8 models. ALL on -P kl02 (ui54 allocation is exhausted).
set -uo pipefail
FT=/scratch/kl02/$USER/Projv2v/FlowTok
WD=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_cmp_m8align_B_bl128_cond3nan1
CKPT=$WD/ckpts/800000.ckpt
FIRED=$WD/.crossmode_800k_fired
LOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_watch_m8a800k.log
mkdir -p /scratch/kl02/$USER/Projv2v/job_logs
exec >> "$LOG" 2>&1
cd "$FT"
echo "[$(date '+%F %T')] watcher start; waiting for $CKPT"

if [ -e "$FIRED" ]; then echo "already fired downstream ($FIRED exists); exit"; exit 0; fi

END=$(( $(date +%s) + 46*3600 ))   # margin under the 48h walltime
while [ ! -e "$CKPT" ]; do
  if [ "$(date +%s)" -ge "$END" ]; then
    echo "[$(date '+%F %T')] walltime guard hit, 800k not ready -> self-resubmit watcher"
    qsub watch_m8align_v2v_800k_finish_gadi.sh && echo "re-qsubbed watcher"
    exit 0
  fi
  sleep 600
done

echo "[$(date '+%F %T')] 800000.ckpt detected; settling 60s in case write is in progress"
sleep 60
echo "[$(date '+%F %T')] firing downstream on kl02"
touch "$FIRED"

# 1) canonical inference-matrix cell: native metrics + fp32 dBZ array dump, BOTH conds (matrix_cell_ft default)
JM=$(qsub -P kl02 -v MODE=v2v,MODEL=m8align,STEP=800000 matrix_cell_ft_gadi.sh)
echo "[$(date '+%F %T')] matrix_cell_ft (ct+nofilt) -> $JM"

# 2) crossmode: add FID family to the v2v cell; afterok on the matrix cell (arrays must exist). GT auto-built.
for C in ct nofilt; do
  PRED=$WD/test_holdout_800000_${C}/arrays/pred_dbz.npy
  JSON=$WD/test_holdout_800000_${C}/metrics.json
  JX=$(qsub -P kl02 -W depend=afterok:$JM \
         -v MODE=v2v,COND=${C},PRED=${PRED},JSON=${JSON} \
         crossmode_job_gadi.sh)
  echo "[$(date '+%F %T')] crossmode $C (afterok:$JM) -> $JX"
done
echo "[$(date '+%F %T')] downstream chained; watcher done"
