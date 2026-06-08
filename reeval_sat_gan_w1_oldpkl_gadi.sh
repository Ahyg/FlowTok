#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=03:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N sat_gan_w1_reeval

# CORRECTIVE re-eval of ONLY the w1 cell of the sat-GAN sweep.
# Why: the small20 test pkl was rewritten 2026-06-06 00:47 (~41% of frames swapped).
# Cells nogan/w0001/w001/w01 ran on the OLD (pre-00:47) test set; w1 started after
# 00:47 so it ran on the NEW set -> not comparable. This re-evaluates w1 on the
# ARCHIVED OLD pkl so all 5 cells share one test set. The w1 NEW-pkl result is moved
# aside to eval_final_newpkl/ (not deleted).

set -uo pipefail

export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"
export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
FLOWTOK_STAGING="${HF_HOME}/flowtok_staging"
export LPIPS_VGG_PTH="${LPIPS_VGG_PTH:-${FLOWTOK_STAGING}/vgg.pth}"
export VGG16_IMAGENET_PTH="${VGG16_IMAGENET_PTH:-${TORCH_HOME}/hub/checkpoints/vgg16-397923af.pth}"
export CONVNEXT_SMALL_IMAGENET_PTH="${CONVNEXT_SMALL_IMAGENET_PTH:-${TORCH_HOME}/hub/checkpoints/convnext_small-0c510722.pth}"

source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate 1d-tokenizer

FLOWTOK_ROOT="/scratch/kl02/$USER/Projv2v/FlowTok"
EXP_ROOT="/scratch/kl02/$USER/Projv2v/Experiments/ae_recipe_sweep/gan_sat"
# ARCHIVED OLD pkl = the test set the other 4 cells used (pre-00:47 rewrite).
OLD_PKL="/g/data/kl02/yh0308/Data/71/_archive_filelists_20260605/dataset_filelist_i2i_test_202407_202507_nofilter_clip16_small20.pkl"
cell="s10_gan_w1"
cd "${FLOWTOK_ROOT}"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cfg="${FLOWTOK_ROOT}/configs/${cell}_gadi.yaml"
ckptdir=$(ls -d "${EXP_ROOT}/${cell}"/checkpoint-[0-9]* 2>/dev/null | sort -t- -k2 -n | tail -1)
ckpt="${ckptdir}/ema_model/pytorch_model.bin"
out="${EXP_ROOT}/${cell}/eval_final"

if [[ ! -f "$OLD_PKL" ]]; then echo "ERROR: archived pkl missing: $OLD_PKL"; exit 1; fi
if [[ ! -f "$ckpt" ]]; then echo "ERROR: no ema ckpt under ${ckptdir:-<none>}"; exit 1; fi

# discard the NEW-pkl w1 result (user: no need to keep it) and any stale _newpkl from a prior run
rm -rf "$out" "${out}_newpkl"
mkdir -p "$out"
echo "discarded NEW-pkl w1 result; will recompute on OLD pkl"

echo "RE-EVAL ${cell} on ARCHIVED OLD pkl -> ${out}"
python -u scripts/test_flowtitok_ae.py \
  --config "$cfg" \
  --checkpoint "$ckpt" \
  --out_dir "$out" \
  --max_batches_metrics -1 \
  --max_batches_images 2 \
  --split test \
  --filelist_path "${OLD_PKL}" \
  --lpips_net alex \
&& echo "DONE ${cell} (old pkl)" || echo "FAIL ${cell}"
