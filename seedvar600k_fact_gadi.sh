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
#PBS -N sv_fact600
# Sampling-variance: fact-v2v @600k, ns20 (config sample_steps=20), ONE SEED per job.
# Full nofilt holdout, inline full metrics (matches production matrix_cell: bs8, max_batches_metrics=-1).
# Production point estimate (seed 42) = avg_fss 0.4787; this runs seeds 0..15 for a 16-sample band.
set -uo pipefail
: "${SEED:?need SEED}"
export HF_HOME="/scratch/kl02/$USER/hf_cache"; export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"; export XDG_CACHE_HOME="$HF_HOME"; export HF_HUB_OFFLINE=1; export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
EXP=/scratch/kl02/yh0308/Projv2v/Experiments
CFG=$FT/configs/Sat2Radar-v2v-cmp-fact-B-bl128-cond3nan1_gadi.py
CKPT=$EXP/sat2radar_flowtok_v2v_cmp_fact_B_bl128_cond3nan1/ckpts/600000.ckpt
PKL=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_test_202407_202507_nofilter_nan1_clip16.pkl
OUT=$EXP/_seed_variance_600k_20260624/fact_v2v_600k_ns20/seed${SEED}
cd $FT
[ -e "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
if [ -f "$OUT/metrics.json" ] && [ -f "$OUT/arrays/pred_dbz.npy" ]; then echo "[skip] fact seed$SEED done"; exit 0; fi
mkdir -p "$OUT/arrays"
echo "[$(date '+%F %T')] fact-v2v 600k ns20 seed=$SEED -> $OUT"
python3 -u scripts/test_sat2radar_v2v.py \
  --config "$CFG" --ckpt "$CKPT" --out_dir "$OUT" \
  --split test --mode v2v --filelist_path "$PKL" \
  --batch_size 8 --max_batches_metrics -1 --max_batches_images 0 \
  --seed $SEED \
  --dump_arrays --dump_pred_only --arrays_dir "$OUT/arrays" \
  --metrics_json "$OUT/metrics.json" --gpu 0
echo "[$(date '+%F %T')] done fact seed=$SEED rc=$?"
