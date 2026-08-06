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
#PBS -m abe
#PBS -N ht_factddpm_nsweep
set -uo pipefail
export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"
export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
CFG=$FT/configs/Sat2Radar-v2v-cmp-factddpm-B-bl128-cond3nan1_gadi.py
WD=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_cmp_factddpm_B_bl128_cond3nan1
STEP="${STEP:-600000}"
CKPT=$WD/ckpts/${STEP}.ckpt
# nofilt (full, unfiltered) v2v test set — same pkl the existing nofilt holdout used.
TEST_PKL_NF=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_test_202407_202507_nofilter_nan1_clip16.pkl
mkdir -p /scratch/kl02/$USER/Projv2v/job_logs
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_htest_v2v_factddpm_nsweep.log
cd $FT
if [ ! -e "$CKPT" ]; then echo "missing $CKPT, abort"; exit 1; fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# factddpm = DDPM/diffusion arm (generation_algorithm=diffusion). NFE = DDIM steps,
# set via --diffusion_sample_steps (overrides config.diffusion.sample_steps=500).
# Compare vs fact (flow) which runs at NFE=20:
#   * ns20  = equal sampling-budget comparison against flow-ns20
#   * ns500 = give DDPM its full budget (reproduces existing _nofilt result + adds arrays)
# --dump_arrays/--dump_pred_only saves fp32 pred arrays so crossmode_metrics.py can add
# FID/KID later (the existing base holdout never dumped arrays for this arm).
run_nfe () {
  local NFE="$1"
  local OUT="$WD/test_holdout_${STEP}_nofilt_ns${NFE}"
  mkdir -p "$OUT"
  echo "[$(date '+%F %T')] factddpm nofilt ns${NFE} -> $OUT" >> "$JOBLOG"
  python3 -u scripts/test_sat2radar_v2v.py \
    --config "$CFG" --ckpt "$CKPT" --out_dir "$OUT" \
    --split test --mode v2v --filelist_path "$TEST_PKL_NF" \
    --max_batches_metrics -1 --max_batches_images 6 \
    --batch_size 8 --metrics_json "$OUT/metrics.json" \
    --diffusion_sample_steps "$NFE" \
    --dump_arrays --dump_pred_only --arrays_dir "$OUT/arrays" \
    --gpu "$CUDA_VISIBLE_DEVICES" >> "$JOBLOG" 2>&1
  echo "[$(date '+%F %T')] DONE ns${NFE} rc=$?" >> "$JOBLOG"
}

# ns20 first (cheap, the genuinely-missing datapoint), then ns500 (slow, ~9h over 43.5k frames).
run_nfe 20
run_nfe 500
echo "[$(date '+%F %T')] factddpm nofilt NFE sweep {20,500} done." >> "$JOBLOG"
