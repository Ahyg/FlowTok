#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=06:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=128GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N seedvar_ft
# Seed-variance for FlowTok generative models. Params via qsub -v MODEL=...,SEEDS="0 1 2 3"
# Full nofilt test, ALL metrics (incl FID/FVD/KVD/TC), pred-only fp16 dump. Resumable (skips done seeds).
set -uo pipefail
export HF_HOME="/scratch/kl02/$USER/hf_cache"; export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"; export XDG_CACHE_HOME="$HF_HOME"; export HF_HUB_OFFLINE=1; export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
EXP=/scratch/kl02/yh0308/Projv2v/Experiments
OUTROOT=$EXP/_seed_variance_20260616
cd $FT

: "${MODEL:?set MODEL}"; : "${GRP:?set GRP}"
# qsub -v splits on commas, so pass a single GRP (0..3); compute the 4 seeds here (16 seeds 0..15).
SPG=${SPG:-4}; SEEDS="$(seq $((GRP*SPG)) $((GRP*SPG+SPG-1)))"
echo "MODEL=$MODEL GRP=$GRP SEEDS=[$SEEDS]"
PREDONLY="--dump_pred_only"; OUTSUF=""
if [ "${GT_ANCHOR:-0}" = 1 ]; then SEEDS=0; OUTSUF="_gtanchor"; PREDONLY=""; echo "GT_ANCHOR: dump gt+pred seed0 -> ${MODEL}/seed0_gtanchor"; fi
I2I_PKL=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_test_202407_202507_nofilter_nan1_clip16.pkl
V2V_PKL=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_test_202407_202507_nofilter_nan1_clip16.pkl

case "$MODEL" in
  m8_i2i)    CFG=Sat2Radar-i2i-cmp-m8align-B-bl128-cond3nan1_gadi.py; DIR=sat2radar_flowtok_i2i_cmp_m8align_B_bl128_cond3nan1; STEP=200000; MODE=i2i; PKL=$I2I_PKL; BS=32 ;;
  xattn_i2i) CFG=Sat2Radar-i2i-cmp-xattn-B-bl128-cond3nan1_gadi.py;   DIR=sat2radar_flowtok_i2i_cmp_xattn_B_bl128_cond3nan1;   STEP=200000; MODE=i2i; PKL=$I2I_PKL; BS=32 ;;
  m8_v2v)    CFG=Sat2Radar-v2v-cmp-m8align-B-bl128-cond3nan1_gadi.py; DIR=sat2radar_flowtok_v2v_cmp_m8align_B_bl128_cond3nan1; STEP=300000; MODE=v2v; PKL=$V2V_PKL; BS=16 ;;
  xattn_v2v) CFG=Sat2Radar-v2v-cmp-xattn-B-bl128-cond3nan1_gadi.py;   DIR=sat2radar_flowtok_v2v_cmp_xattn_B_bl128_cond3nan1;   STEP=300000; MODE=v2v; PKL=$V2V_PKL; BS=16 ;;
  fact_v2v)  CFG=Sat2Radar-v2v-cmp-fact-B-bl128-cond3nan1_gadi.py;    DIR=sat2radar_flowtok_v2v_cmp_fact_B_bl128_cond3nan1;    STEP=300000; MODE=v2v; PKL=$V2V_PKL; BS=16 ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
CKPT=$EXP/$DIR/ckpts/${STEP}.ckpt
[ -e "$CKPT" ] || { echo "missing ckpt $CKPT"; exit 1; }

for S in $SEEDS; do
  OUT=$OUTROOT/$MODEL/seed${S}${OUTSUF}
  if [ -f "$OUT/metrics.json" ] && [ -f "$OUT/arrays/pred_dbz.npy" ]; then echo "[skip] $MODEL seed$S done"; continue; fi
  mkdir -p "$OUT/arrays"
  echo "[$(date '+%F %T')] === $MODEL seed=$S (mode=$MODE bs=$BS step=$STEP) ==="
  python3 -u scripts/test_sat2radar_v2v.py \
    --config $FT/configs/$CFG --ckpt "$CKPT" --out_dir "$OUT" \
    --split test --mode $MODE --filelist_path "$PKL" \
    --batch_size $BS --max_batches_metrics -1 --max_batches_images 0 \
    --seed $S \
    --dump_arrays --arrays_dir "$OUT/arrays" $PREDONLY --defer_fss \
    --metrics_json "$OUT/metrics.json" --gpu 0
  echo "[$(date '+%F %T')] done $MODEL seed=$S rc=$?"
done
echo "[$(date '+%F %T')] GROUP DONE $MODEL seeds=[$SEEDS]"
