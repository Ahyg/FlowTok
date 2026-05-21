#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=6:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N i2i_m1m8_genmetrics
set -uo pipefail
export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"          # InceptionV3 FID weight mirrored here
export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
FT=/scratch/kl02/$USER/Projv2v/FlowTok
EXP=/scratch/kl02/yh0308/Projv2v/Experiments
cd $FT
mkdir -p /scratch/kl02/$USER/Projv2v/job_logs

# arm -> workdir suffix ; diffusion arms (m3,m5) get NFE=20 override.
NAMES=(m1 m2 m3 m4 m5 m6 m7 m8)
WDIRS=(baseline textvae diffusion xpred diffusion_tokconcat textvae_lowkld flow_tokconcat flow_tokconcat_xpred)
DIFFUSION=(0 0 1 0 1 0 0 0)

for i in "${!NAMES[@]}"; do
  N=${NAMES[$i]}
  WD=$EXP/sat2radar_flowtok_i2i_b_${WDIRS[$i]}_2021summer
  CFG=$FT/configs/Sat2Radar-i2i-b-${N}-2021summer_gadi.py
  CKPT=$WD/ckpts/60000.ckpt
  OUT=$WD/test_holdout_60000_genmetrics
  JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_${N}_genmetrics.log
  if [ ! -e "$CKPT" ]; then echo "[$(date '+%F %T')] $N: missing $CKPT, skip"; continue; fi
  mkdir -p "$OUT"
  NFE_ARG=""
  [ "${DIFFUSION[$i]}" = "1" ] && NFE_ARG="--diffusion_sample_steps 20"
  echo "[$(date '+%F %T')] $N: recompute (FSS+FID/KID) -> $OUT/metrics.json ${NFE_ARG}"
  python -u scripts/test_sat2radar_v2v.py \
    --config "$CFG" \
    --ckpt "$CKPT" \
    --out_dir "$OUT" \
    --split test \
    --mode i2i \
    --max_batches_metrics -1 \
    --max_batches_images 0 \
    --batch_size 8 \
    --kid_subsets 50 \
    --kid_subset_size 100 \
    --metrics_json "$OUT/metrics.json" \
    --gpu "$CUDA_VISIBLE_DEVICES" \
    $NFE_ARG \
    > "$JOBLOG" 2>&1 || echo "[$(date '+%F %T')] $N: FAILED (see $JOBLOG)"
  echo "[$(date '+%F %T')] $N: done"
done
echo "[$(date '+%F %T')] all M1-M8 gen-metric recompute complete"
