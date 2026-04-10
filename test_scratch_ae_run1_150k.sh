#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=06:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N test_ae_r1_150k

set -euo pipefail

export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"
export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"

source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate 1d-tokenizer

FLOWTOK_ROOT="/scratch/kl02/$USER/Projv2v/FlowTok"
EXP_ROOT="/scratch/kl02/$USER/Projv2v/Experiments"
VAL_FILELIST="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_val_202401_202406_nofilter.pkl"
TEST_FILELIST="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_test_202407_202507_nofilter.pkl"

FSS_THRESHOLDS="0,5,10,15,20,25,30,35,40,45,50,55,60"
FSS_SCALES="1,2,3,4,5,6,7,8,9,10"

cd "${FLOWTOK_ROOT}"
export CUDA_VISIBLE_DEVICES=0

# ──────────────────────────────────────────────────────────
# Radar scratch AE run1 @ 150K (EMA)
# ──────────────────────────────────────────────────────────
RADAR_CFG="${FLOWTOK_ROOT}/configs/radar_flowtitok_ae_bl77_vae_scratch_gadi.yaml"
RADAR_CKPT="${EXP_ROOT}/radar_flowtitok_ae_bl77_vae_scratch_run1_gadi/checkpoint-150000/ema_model/pytorch_model.bin"

echo "========== Radar scratch run1 @ 150K — VAL =========="
python scripts/test_flowtitok_ae.py \
  --config="${RADAR_CFG}" \
  --checkpoint="${RADAR_CKPT}" \
  --filelist_path="${VAL_FILELIST}" \
  --split=val \
  --out_dir="${EXP_ROOT}/radar_flowtitok_ae_bl77_vae_scratch_run1_gadi/val_150k_ema" \
  --max_batches_images=10 \
  --fss_thresholds="${FSS_THRESHOLDS}" \
  --fss_scales="${FSS_SCALES}"

echo "========== Radar scratch run1 @ 150K — TEST =========="
python scripts/test_flowtitok_ae.py \
  --config="${RADAR_CFG}" \
  --checkpoint="${RADAR_CKPT}" \
  --filelist_path="${TEST_FILELIST}" \
  --split=test \
  --out_dir="${EXP_ROOT}/radar_flowtitok_ae_bl77_vae_scratch_run1_gadi/test_150k_ema" \
  --max_batches_images=10 \
  --fss_thresholds="${FSS_THRESHOLDS}" \
  --fss_scales="${FSS_SCALES}"

# ──────────────────────────────────────────────────────────
# Sat (3ch) scratch AE run1 @ 150K (EMA)
# ──────────────────────────────────────────────────────────
SAT_CFG="${FLOWTOK_ROOT}/configs/sat_flowtitok_ae_bl77_vae_scratch_gadi.yaml"
SAT_CKPT="${EXP_ROOT}/sat_flowtitok_ae_bl77_vae_scratch_run1_gadi/checkpoint-150000/ema_model/pytorch_model.bin"

echo "========== Sat scratch run1 @ 150K — VAL =========="
python scripts/test_flowtitok_ae.py \
  --config="${SAT_CFG}" \
  --checkpoint="${SAT_CKPT}" \
  --filelist_path="${VAL_FILELIST}" \
  --split=val \
  --out_dir="${EXP_ROOT}/sat_flowtitok_ae_bl77_vae_scratch_run1_gadi/val_150k_ema" \
  --max_batches_images=10 \
  --fss_thresholds="${FSS_THRESHOLDS}" \
  --fss_scales="${FSS_SCALES}"

echo "========== Sat scratch run1 @ 150K — TEST =========="
python scripts/test_flowtitok_ae.py \
  --config="${SAT_CFG}" \
  --checkpoint="${SAT_CKPT}" \
  --filelist_path="${TEST_FILELIST}" \
  --split=test \
  --out_dir="${EXP_ROOT}/sat_flowtitok_ae_bl77_vae_scratch_run1_gadi/test_150k_ema" \
  --max_batches_images=10 \
  --fss_thresholds="${FSS_THRESHOLDS}" \
  --fss_scales="${FSS_SCALES}"

# ──────────────────────────────────────────────────────────
# Sat10ch scratch AE run1 @ 150K (EMA)
# ──────────────────────────────────────────────────────────
SAT10_CFG="${FLOWTOK_ROOT}/configs/sat10ch_flowtitok_ae_bl77_vae_scratch_gadi.yaml"
SAT10_CKPT="${EXP_ROOT}/sat10ch_flowtitok_ae_bl77_vae_scratch_run1_gadi/checkpoint-150000/ema_model/pytorch_model.bin"

echo "========== Sat10ch scratch run1 @ 150K — VAL =========="
python scripts/test_flowtitok_ae.py \
  --config="${SAT10_CFG}" \
  --checkpoint="${SAT10_CKPT}" \
  --filelist_path="${VAL_FILELIST}" \
  --split=val \
  --out_dir="${EXP_ROOT}/sat10ch_flowtitok_ae_bl77_vae_scratch_run1_gadi/val_150k_ema" \
  --max_batches_images=10 \
  --fss_thresholds="${FSS_THRESHOLDS}" \
  --fss_scales="${FSS_SCALES}"

echo "========== Sat10ch scratch run1 @ 150K — TEST =========="
python scripts/test_flowtitok_ae.py \
  --config="${SAT10_CFG}" \
  --checkpoint="${SAT10_CKPT}" \
  --filelist_path="${TEST_FILELIST}" \
  --split=test \
  --out_dir="${EXP_ROOT}/sat10ch_flowtitok_ae_bl77_vae_scratch_run1_gadi/test_150k_ema" \
  --max_batches_images=10 \
  --fss_thresholds="${FSS_THRESHOLDS}" \
  --fss_scales="${FSS_SCALES}"

echo "========== ALL DONE =========="
