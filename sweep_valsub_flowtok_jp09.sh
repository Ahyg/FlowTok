#!/bin/bash
#PBS -P jp09
#PBS -q gpuhopper
#PBS -l walltime=14:00:00
#PBS -l storage=gdata/kl02+scratch/kl02+scratch/jp09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N sweep_ft_fill
# Validation-subset ckpt sweep — FlowTok models (i2i: m8align,xattn ; v2v: m8align,xattn,fact).
# Evaluate EVERY saved ckpt on the small val-subset (full metric suite + a few vis) to find best ckpt.
# Resumable: a ckpt whose step<N>_metrics.json already exists is skipped. Per-ckpt errors are logged & skipped.
# jp09 fill run: only ft_v2v_m8align 700k/750k/800k remain un-evaluated; everything else is skipped.
set -uo pipefail
export HF_HOME="/scratch/kl02/$USER/hf_cache"; export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"; export XDG_CACHE_HOME="$HF_HOME"; export HF_HUB_OFFLINE=1; export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
EXP=/scratch/kl02/yh0308/Projv2v/Experiments
ROOT=$EXP/_ckpt_sweep_valsub_20260618
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_sweep_ft.log
mkdir -p "$(dirname "$JOBLOG")"
exec > "$JOBLOG" 2>&1
cd $FT

# key|mode|config|workdir|batch_size
MODELS=(
  "ft_i2i_m8align|i2i|Sat2Radar-i2i-cmp-m8align-B-bl128-cond3nan1_gadi.py|sat2radar_flowtok_i2i_cmp_m8align_B_bl128_cond3nan1|16"
  "ft_i2i_xattn|i2i|Sat2Radar-i2i-cmp-xattn-B-bl128-cond3nan1_gadi.py|sat2radar_flowtok_i2i_cmp_xattn_B_bl128_cond3nan1|16"
  "ft_v2v_m8align|v2v|Sat2Radar-v2v-cmp-m8align-B-bl128-cond3nan1_gadi.py|sat2radar_flowtok_v2v_cmp_m8align_B_bl128_cond3nan1|8"
  "ft_v2v_xattn|v2v|Sat2Radar-v2v-cmp-xattn-B-bl128-cond3nan1_gadi.py|sat2radar_flowtok_v2v_cmp_xattn_B_bl128_cond3nan1|8"
  "ft_v2v_fact|v2v|Sat2Radar-v2v-cmp-fact-B-bl128-cond3nan1_gadi.py|sat2radar_flowtok_v2v_cmp_fact_B_bl128_cond3nan1|8"
)

for spec in "${MODELS[@]}"; do
  IFS='|' read -r KEY MODE CFGN WDN BS <<< "$spec"
  CFG=$FT/configs/$CFGN
  WD=$EXP/$WDN
  PKL=$ROOT/valsub_${MODE}.pkl
  OUTD=$ROOT/$KEY; mkdir -p "$OUTD"
  echo "[$(date '+%F %T')] ===== $KEY ($MODE, bs=$BS) ====="
  STEPS=$(ls -d "$WD"/ckpts/*.ckpt 2>/dev/null | grep -oP '[0-9]+(?=\.ckpt)' | sort -n)
  for N in $STEPS; do
    OJ=$OUTD/step${N}_metrics.json
    if [ -e "$OJ" ]; then echo "  skip step$N (done)"; continue; fi
    CKPT=$WD/ckpts/${N}.ckpt
    VIS=$OUTD/vis_step${N}; mkdir -p "$VIS"
    echo "  [$(date '+%T')] eval step$N -> $OJ"
    python3 -u scripts/test_sat2radar_v2v.py \
      --config "$CFG" --ckpt "$CKPT" --out_dir "$VIS" \
      --split test --mode $MODE --filelist_path "$PKL" \
      --max_batches_metrics -1 --max_batches_images 1 \
      --batch_size $BS --metrics_json "$OJ" --gpu 0
    RC=$?
    if [ "$RC" != "0" ] || [ ! -e "$OJ" ]; then echo "  !! FAILED step$N rc=$RC (skip, will retry on resume)"; rm -rf "$VIS"; fi
  done
done

echo "[$(date '+%F %T')] FlowTok sweep loop done; running summary"
python3 -u "$ROOT/summarize_bestckpt.py" || echo "summary failed (non-fatal)"
echo "[$(date '+%F %T')] DONE sweep_ft"
