#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=20:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m a
#PBS -N sv600_factddpm
# 16-seed sampling-variance for factddpm (token+DiT+DDPM ablation) @600k ns500 (its
# reported production setting; generation_algorithm=diffusion), WITH per-seed array
# dump for accumulated CRPS. seed0 also dumps gt_dbz.npy (self-aligned GT).
# ns500 ~9h/pass -> SPG=2 (2 seeds/job, ~18h), 8 jobs. Resumable via skip-on-arrays.
#   qsub -P kl02 -v GRP=0,SPG=2 seedvar600k_factddpm_gadi.sh    # GRP 0..7
set -uo pipefail
: "${GRP:?set GRP 0..7}"; SPG=${SPG:-2}; SEEDS="$(seq $((GRP*SPG)) $((GRP*SPG+SPG-1)))"
export HF_HOME="/scratch/kl02/$USER/hf_cache"; export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"; export XDG_CACHE_HOME="$HF_HOME"; export HF_HUB_OFFLINE=1; export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
EXP=/scratch/kl02/yh0308/Projv2v/Experiments
CFG=$FT/configs/Sat2Radar-v2v-cmp-factddpm-B-bl128-cond3nan1_gadi.py
CKPT=$EXP/sat2radar_flowtok_v2v_cmp_factddpm_B_bl128_cond3nan1/ckpts/600000.ckpt
OUTROOT=$EXP/_seed_variance_600k_20260624/factddpm_600k_ns500
V2V_PKL=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_test_202407_202507_nofilter_nan1_clip16.pkl
NFE=500; BS=8
cd $FT
[ -e "$CKPT" ] || { echo "missing ckpt $CKPT"; exit 1; }
echo "[$(date '+%F %T')] factddpm 600k ns${NFE} SEEDS=[$SEEDS] +arrays"

for S in $SEEDS; do
  OUT=$OUTROOT/seed${S}
  if [ -f "$OUT/arrays/pred_dbz.npy" ]; then echo "[skip] factddpm seed$S arrays exist"; continue; fi
  mkdir -p "$OUT/arrays"
  DUMP=(--dump_arrays --arrays_dir "$OUT/arrays")
  [ "$S" != "0" ] && DUMP+=(--dump_pred_only)
  echo "[$(date '+%F %T')] === factddpm seed=$S (gt=$([ "$S" = 0 ] && echo yes || echo no)) ==="
  python3 -u scripts/test_sat2radar_v2v.py \
    --config "$CFG" --ckpt "$CKPT" --out_dir "$OUT" \
    --split test --mode v2v --filelist_path "$V2V_PKL" \
    --batch_size $BS --max_batches_metrics -1 --max_batches_images 0 \
    --diffusion_sample_steps $NFE --seed $S "${DUMP[@]}" \
    --metrics_json "$OUT/metrics.json" --gpu 0
  echo "[$(date '+%F %T')] done factddpm seed=$S rc=$?"
done
echo "[$(date '+%F %T')] GROUP DONE factddpm grp=$GRP seeds=[$SEEDS]"
