#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=05:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=128GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N factv2v_calib
# Decisive test: can deployable light-rain calibration close fact-v2v's aFSS gap vs diu-i2i (0.422)?
# Dump FULL fact-v2v@300k nofilt (pred+gt, NFE20), then rescore RAW vs quantile-map-to-train-clim (w sweep).
set -uo pipefail
export HF_HOME="/scratch/kl02/$USER/hf_cache"; export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"; export XDG_CACHE_HOME="$HF_HOME"; export HF_HUB_OFFLINE=1; export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
EX=/scratch/kl02/yh0308/Projv2v/Experiments
ROOT=$EX/_eval_factv2v_calib_20260617
OUT=$ROOT/fact_v2v_full
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_factv2v_calib.log
mkdir -p "$OUT/arrays"; exec > "$JOBLOG" 2>&1
cd $FT

CKPT=$EX/sat2radar_flowtok_v2v_cmp_fact_B_bl128_cond3nan1/ckpts/300000.ckpt
PKL=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_test_202407_202507_nofilter_nan1_clip16.pkl

if [ ! -f "$OUT/arrays/pred_dbz.npy" ]; then
  echo "[$(date '+%F %T')] dump full fact-v2v@300k nofilt (pred+gt, NFE20)"
  python3 -u scripts/test_sat2radar_v2v.py \
    --config $FT/configs/Sat2Radar-v2v-cmp-fact-B-bl128-cond3nan1_gadi.py --ckpt "$CKPT" \
    --out_dir "$OUT" --split test --mode v2v --filelist_path "$PKL" \
    --batch_size 16 --max_batches_metrics -1 --max_batches_images 0 \
    --flow_sample_steps 20 --skip_gen_metrics \
    --dump_arrays --arrays_dir "$OUT/arrays" \
    --metrics_json "$OUT/metrics.json" --gpu 0
fi
echo "[$(date '+%F %T')] calibration rescore (stride 8; clean-QM=train clim, partial weights)"
python3 -u "$EX/_eval_pmm_nfe_20260616/pmm_rescore.py" \
  --eval_root "$ROOT" --train_clim "$EX/_eval_pmm_nfe_20260616/radar_clim_train.npy" --stride 8
echo "[$(date '+%F %T')] DONE — compare clean_w* aFSS to diu-i2i target 0.422"
