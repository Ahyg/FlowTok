#!/bin/bash
#PBS -P ui54
#PBS -q gpuhopper
#PBS -l walltime=6:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m a
#PBS -N mtx_ft
# Inference-matrix cell (FlowTok). Faithful superset of the proven repro-check:
# same CFG/CKPT/filelist/batch_size -> bit-identical metrics; adds fp32 dBZ array dump.
# Compute on -P ui54; ALL outputs under Experiments/<model>/ on kl02 (overwrites holdout).
# Params via qsub -v : MODE={i2i|v2v} MODEL={m8align|xattn|fact} STEP=<int> [COND={ct|nofilt}]
set -uo pipefail
: "${MODE:?need MODE}"; : "${MODEL:?need MODEL}"; : "${STEP:?need STEP}"
export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"; export TORCH_HOME="$HF_HOME"; export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1; export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
CFG=$FT/configs/Sat2Radar-${MODE}-cmp-${MODEL}-B-bl128-cond3nan1_gadi.py
WD=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_${MODE}_cmp_${MODEL}_B_bl128_cond3nan1
CKPT=$WD/ckpts/${STEP}.ckpt
FLDIR=/g/data/kl02/yh0308/Data/71/filelists
PKL_CT=$FLDIR/dataset_filelist_${MODE}_test_202407_202507_cond3nan1_clip16_p005_seed42.pkl
PKL_NF=$FLDIR/dataset_filelist_${MODE}_test_202407_202507_nofilter_nan1_clip16.pkl
BS=16; [ "$MODE" = "v2v" ] && BS=8
CONDS="${COND:-ct nofilt}"
cd "$FT"
[ -e "$CKPT" ] || { echo "MISSING ckpt $CKPT"; exit 1; }
[ -f "$CFG" ]  || { echo "MISSING cfg $CFG"; exit 1; }
echo "[$(date '+%F %T')] cell MODE=$MODE MODEL=$MODEL STEP=$STEP CONDS='$CONDS' BS=$BS host=$(hostname) groups=$(groups)"
rc_all=0
for COND in $CONDS; do
  case "$COND" in ct) PKL=$PKL_CT;; nofilt) PKL=$PKL_NF;; *) echo "bad COND $COND"; exit 2;; esac
  OUT=$WD/test_holdout_${STEP}_${COND}
  ARR=$OUT/arrays
  mkdir -p "$ARR"
  echo "[$(date '+%F %T')] >>> $COND  ckpt=$CKPT  out=$OUT"
  python3 -u scripts/test_sat2radar_v2v.py \
    --config "$CFG" --ckpt "$CKPT" --out_dir "$OUT" \
    --split test --mode "$MODE" --filelist_path "$PKL" \
    --max_batches_metrics -1 --max_batches_images 0 \
    --batch_size $BS --metrics_json "$OUT/metrics.json" \
    --dump_arrays --dump_pred_only --arrays_dir "$ARR" \
    --gpu 0
  rc=$?; echo "[$(date '+%F %T')] <<< $COND rc=$rc  pred=$(ls -la "$ARR"/pred_dbz.npy 2>/dev/null | awk '{print $5}')"
  [ $rc -ne 0 ] && rc_all=$rc
done
echo "[$(date '+%F %T')] DONE rc_all=$rc_all"
exit $rc_all
