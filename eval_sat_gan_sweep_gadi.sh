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
#PBS -N sat_gan_eval

# Sat10ch AE late-GAN sweep — eval on Gadi. Runs test_flowtitok_ae.py over all 5
# cells' checkpoint-final on the small20 nofilter test set (9280 samples). The
# small20 train index has no val split, so there is no best_val checkpoint —
# `final` is the slot (radar showed final is the apples-to-apples GAN-on slot
# anyway). Submit AFTER the 5 training jobs finish:  qsub eval_sat_gan_sweep_gadi.sh
#
# Emits metrics.json + metrics.md per cell under each cell's eval_final/.

set -uo pipefail   # NOT -e: one failing eval must not abort the rest

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
TEST_PKL="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_test_202407_202507_nofilter_clip16_small20.pkl"
SLOT="${SLOT:-final}"
cd "${FLOWTOK_ROOT}"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CELLS=(s10_gan_nogan s10_gan_w0001 s10_gan_w001 s10_gan_w01 s10_gan_w1)

for cell in "${CELLS[@]}"; do
  cfg="${FLOWTOK_ROOT}/configs/${cell}_gadi.yaml"
  # train_flowtitok_ae.py saves checkpoint-<step> (NOT checkpoint-final), so resolve the
  # highest-numbered checkpoint dir; and eval the EMA weights (use_ema=true) to match the
  # run1/run4 holdout pipeline (which evaluates ema_model, not unwrapped_model).
  ckptdir=$(ls -d "${EXP_ROOT}/${cell}"/checkpoint-[0-9]* 2>/dev/null | sort -t- -k2 -n | tail -1)
  ckpt="${ckptdir}/ema_model/pytorch_model.bin"
  out="${EXP_ROOT}/${cell}/eval_${SLOT}"
  if [[ ! -f "$cfg" ]];  then echo "SKIP ${cell}: no config $cfg";        continue; fi
  if [[ ! -f "$ckpt" ]]; then echo "SKIP ${cell}: no ema ckpt under ${ckptdir:-<none>} (training unfinished?)"; continue; fi
  if [[ -f "${out}/metrics.json" ]]; then echo "SKIP ${cell}: already evaluated"; continue; fi
  mkdir -p "$out"
  echo "EVAL ${cell} -> ${out}"
  python -u scripts/test_flowtitok_ae.py \
    --config "$cfg" \
    --checkpoint "$ckpt" \
    --out_dir "$out" \
    --max_batches_metrics -1 \
    --max_batches_images 2 \
    --split test \
    --filelist_path "${TEST_PKL}" \
    --lpips_net alex \
  && echo "DONE ${cell}" || echo "FAIL ${cell}"
done
echo "eval_sat_gan_sweep_gadi ALL DONE — slot=${SLOT}"
