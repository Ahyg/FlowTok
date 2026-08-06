#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=12:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=128GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m a
#PBS -N sv600_ftarr
# 16-seed sampling-variance for the 4 remaining FlowTok flow models @600k ns20, WITH
# per-seed pred_dbz.npy array dump so accumulated CRPS is computable (the earlier
# seedvar600k_flowtok_gadi.sh was METRICS-ONLY). seed0 also dumps gt_dbz.npy (drop
# --dump_pred_only) -> a self-aligned GT for this exact model/loader order.
# Resumable: skips any seed that already has arrays/pred_dbz.npy.
#   qsub -P kl02 -v MODEL=xattn_v2v,GRP=0,SPG=8 seedvar600k_flowtok_arrays_gadi.sh  # GRP 0..1
#   qsub -P ui54 -v MODEL=pixfact,GRP=0,SPG=4 seedvar600k_flowtok_arrays_gadi.sh    # GRP 0..3
set -uo pipefail
: "${MODEL:?set MODEL (xattn_i2i|m8_i2i|xattn_v2v|m8_v2v|pixfact)}"; : "${GRP:?set GRP}"
SPG=${SPG:-8}; SEEDS="$(seq $((GRP*SPG)) $((GRP*SPG+SPG-1)))"
export HF_HOME="/scratch/kl02/$USER/hf_cache"; export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"; export XDG_CACHE_HOME="$HF_HOME"; export HF_HUB_OFFLINE=1; export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
EXP=/scratch/kl02/yh0308/Projv2v/Experiments
OUTROOT=$EXP/_seed_variance_600k_20260624
I2I_PKL=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_test_202407_202507_nofilter_nan1_clip16.pkl
V2V_PKL=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_test_202407_202507_nofilter_nan1_clip16.pkl
cd $FT

case "$MODEL" in
  xattn_i2i) CFG=Sat2Radar-i2i-cmp-xattn-B-bl128-cond3nan1_gadi.py;   DIR=sat2radar_flowtok_i2i_cmp_xattn_B_bl128_cond3nan1;   MODE=i2i; PKL=$I2I_PKL; BS=32; OUTNAME=xattn_i2i_600k_ns20 ;;
  m8_i2i)    CFG=Sat2Radar-i2i-cmp-m8align-B-bl128-cond3nan1_gadi.py; DIR=sat2radar_flowtok_i2i_cmp_m8align_B_bl128_cond3nan1; MODE=i2i; PKL=$I2I_PKL; BS=32; OUTNAME=m8align_i2i_600k_ns20 ;;
  xattn_v2v) CFG=Sat2Radar-v2v-cmp-xattn-B-bl128-cond3nan1_gadi.py;   DIR=sat2radar_flowtok_v2v_cmp_xattn_B_bl128_cond3nan1;   MODE=v2v; PKL=$V2V_PKL; BS=16; OUTNAME=xattn_v2v_600k_ns20 ;;
  m8_v2v)    CFG=Sat2Radar-v2v-cmp-m8align-B-bl128-cond3nan1_gadi.py; DIR=sat2radar_flowtok_v2v_cmp_m8align_B_bl128_cond3nan1; MODE=v2v; PKL=$V2V_PKL; BS=16; OUTNAME=m8align_v2v_600k_ns20 ;;
  # pixel-space factorized arm (patch_size 8, pixel_space=True). BS=8, NOT 16 like the token v2v
  # arms above: the production nofilt run at bs=8 peaked at exactly the 90GB it requested, so bs=16
  # would risk OOM. Frame order is independent of BS (loader is shuffle=False), so the ensemble
  # stays frame-aligned with the canonical GT regardless.
  pixfact)   CFG=Sat2Radar-v2v-cmp-pixfact-B-p8-cond3nan1_gadi.py;   DIR=sat2radar_flowtok_v2v_cmp_pixfact_B_p8_cond3nan1;   MODE=v2v; PKL=$V2V_PKL; BS=8;  OUTNAME=pixfact_v2v_600k_ns20 ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
CKPT=$EXP/$DIR/ckpts/600000.ckpt
[ -e "$CKPT" ] || { echo "missing ckpt $CKPT"; exit 1; }
echo "[$(date '+%F %T')] MODEL=$MODEL ($OUTNAME) mode=$MODE bs=$BS ns=20 SEEDS=[$SEEDS] +arrays"

for S in $SEEDS; do
  OUT=$OUTROOT/$OUTNAME/seed${S}
  if [ -f "$OUT/arrays/pred_dbz.npy" ]; then echo "[skip] $OUTNAME seed$S arrays exist"; continue; fi
  mkdir -p "$OUT/arrays"
  DUMP=(--dump_arrays --arrays_dir "$OUT/arrays")
  [ "$S" != "0" ] && DUMP+=(--dump_pred_only)   # seed0 keeps gt_dbz.npy (self-aligned GT)
  echo "[$(date '+%F %T')] === $OUTNAME seed=$S (gt=$([ "$S" = 0 ] && echo yes || echo no)) ==="
  python3 -u scripts/test_sat2radar_v2v.py \
    --config $FT/configs/$CFG --ckpt "$CKPT" --out_dir "$OUT" \
    --split test --mode $MODE --filelist_path "$PKL" \
    --batch_size $BS --max_batches_metrics -1 --max_batches_images 0 \
    --seed $S "${DUMP[@]}" \
    --metrics_json "$OUT/metrics.json" --gpu 0
  echo "[$(date '+%F %T')] done $OUTNAME seed=$S rc=$?"
done
echo "[$(date '+%F %T')] GROUP DONE $MODEL grp=$GRP seeds=[$SEEDS]"
