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
#PBS -m abe
#PBS -N rc_ft_m8a_i2i
# REPRO CHECK: re-run m8align-i2i@600k on ct subset under project ui54 with array dump.
# Metrics -> kl02 temp (compare to existing holdout, do NOT overwrite). Arrays -> /scratch/ui54.
set -uo pipefail
export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"; export TORCH_HOME="$HF_HOME"; export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1; export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
CFG=$FT/configs/Sat2Radar-i2i-cmp-m8align-B-bl128-cond3nan1_gadi.py
WD=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_i2i_cmp_m8align_B_bl128_cond3nan1
STEP=600000
CKPT=$WD/ckpts/${STEP}.ckpt
PKL=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_test_202407_202507_cond3nan1_clip16_p005_seed42.pkl
RC=/scratch/kl02/$USER/Projv2v/repro_check
ARR=$RC/arrays_ft_m8align_i2i_600k_ct
mkdir -p "$RC" "$ARR"
JOBLOG=$RC/${PBS_JOBID}_rc_ft_m8a_i2i.log
cd $FT
echo "[$(date '+%F %T')] ui54 probe: whoami=$(whoami) groups=$(groups) host=$(hostname)" > "$JOBLOG"
[ -e "$CKPT" ] || { echo "missing $CKPT" | tee -a "$JOBLOG"; exit 1; }
python3 -u scripts/test_sat2radar_v2v.py \
  --config "$CFG" --ckpt "$CKPT" --out_dir "$RC/ft_m8align_i2i_600k_ct_out" \
  --split test --mode i2i --filelist_path "$PKL" \
  --max_batches_metrics -1 --max_batches_images 0 \
  --batch_size 16 --metrics_json "$RC/ft_m8align_i2i_600k_ct.json" \
  --dump_arrays --dump_pred_only --arrays_dir "$ARR" \
  --gpu 0 >> "$JOBLOG" 2>&1
echo "[$(date '+%F %T')] python rc=$?" >> "$JOBLOG"
echo "=== arrays dir listing ===" >> "$JOBLOG"
ls -la "$ARR" >> "$JOBLOG" 2>&1 || echo "(arrays dir not accessible)" >> "$JOBLOG"
