#!/bin/bash
#PBS -P ui54
#PBS -q gpuhopper
#PBS -l walltime=4:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m a
#PBS -N xmode
# Post-hoc CROSS-MODE metrics for one holdout (reuses _generation_metrics on dumped dBZ arrays).
# Params via qsub -v : MODE={i2i|v2v} COND={ct|nofilt} PRED=<pred_dbz.npy> JSON=<holdout.json>
#   [GTCACHE=<path>]  [BUILD_GT_ONLY=1]
set -uo pipefail
: "${MODE:?}"; : "${COND:?}"
export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"; export TORCH_HOME="$HF_HOME"; export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1; export WANDB_MODE=disabled
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
GTDIR=/scratch/kl02/$USER/Projv2v/crossmode_gt
mkdir -p "$GTDIR"
GTCACHE="${GTCACHE:-$GTDIR/gt_${MODE}_${COND}.npy}"
cd "$FT"
echo "[$(date '+%F %T')] xmode MODE=$MODE COND=$COND gt=$GTCACHE host=$(hostname)"
if [ "${BUILD_GT_ONLY:-0}" = "1" ]; then
  python3 -u scripts/crossmode_metrics.py --mode "$MODE" --cond "$COND" --gt-cache "$GTCACHE" --build-gt-only
  exit $?
fi
if [ -n "${GTTC_OUT:-}" ]; then
  python3 -u scripts/crossmode_metrics.py --mode "$MODE" --cond "$COND" --gt-cache "$GTCACHE" --gt-tc-baseline "$GTTC_OUT" --device cuda
  exit $?
fi
: "${PRED:?}"; : "${JSON:?}"
[ -e "$PRED" ] || { echo "MISSING pred $PRED"; exit 1; }
[ -e "$JSON" ] || { echo "MISSING json $JSON"; exit 1; }
python3 -u scripts/crossmode_metrics.py \
  --mode "$MODE" --cond "$COND" --gt-cache "$GTCACHE" \
  --pred "$PRED" --json "$JSON" --device cuda
echo "[$(date '+%F %T')] xmode rc=$?"
