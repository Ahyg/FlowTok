#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=10:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N htest_v2v_m8align
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
CFG=$FT/configs/Sat2Radar-v2v-cmp-m8align-B-2021summer_gadi.py
WD=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_cmp_m8align_B
CKPT=$WD/ckpts/60000.ckpt
OUT=$WD/test_holdout_60000_posfix
TEST_PKL=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_test_202407_202507.pkl
mkdir -p /scratch/kl02/$USER/Projv2v/job_logs "$OUT"
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_htest_v2v_m8align.log
cd $FT
if [ ! -e "$CKPT" ]; then echo "missing $CKPT, abort"; exit 1; fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python3 -u scripts/test_sat2radar_v2v.py   --config "$CFG"   --ckpt "$CKPT"   --out_dir "$OUT"   --split test   --mode v2v   --filelist_path "$TEST_PKL"   --max_batches_metrics -1   --max_batches_images 6   --batch_size 8   --skip_gen_metrics   --metrics_json "$OUT/metrics.json"   --gpu "$CUDA_VISIBLE_DEVICES"   > "$JOBLOG" 2>&1
echo "[$(date '+%F %T')] m8align v2v holdout test done: $OUT/metrics.json"
