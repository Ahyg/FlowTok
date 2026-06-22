#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=10:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=30GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N ae350k_eval

# Holdout eval of the 300k->350k GAN-continuation checkpoints (run4ftgan2_cond3nan1) on the
# SAME 46,416-frame nofilter clip16 test set used for the 300k per-threshold runs, so 350k is
# directly comparable to 300k. FULL metric suite (FID + LPIPS + SSIM/PSNR + FSS) PLUS the new
# fss_per_threshold keys. seed 42 deterministic (matches the 300k fss_perthr re-runs).
#   for r in radar350 sat350; do qsub -v RUN=$r eval_ae_350k_holdout_gadi.sh; done

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
EXP="/scratch/kl02/$USER/Projv2v/Experiments"
cd "${FLOWTOK_ROOT}"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

RUN="${RUN:?must pass -v RUN=<radar350|sat350>}"
case "$RUN" in
  radar350) DIR=radar_flowtitok_ae_bl128_vae_scratch_run4ftgan2_cond3nan1_gadi;
            CFG=radar_flowtitok_ae_bl128_vae_scratch_run4ftgan2_cond3nan1_gadi.yaml;;
  radar350hil) DIR=radar_flowtitok_ae_bl128_vae_scratch_run4hilgan_cond3nan1_gadi;
            CFG=radar_flowtitok_ae_bl128_vae_scratch_run4hilgan_cond3nan1_gadi.yaml;;
  sat350)   DIR=sat10ch_flowtitok_ae_bl128_vae_scratch_run4Bftgan2_cond3nan1_gadi;
            CFG=sat10ch_flowtitok_ae_bl128_vae_scratch_run4Bftgan2_cond3nan1_gadi.yaml;;
  *) echo "BAD RUN=$RUN"; exit 2;;
esac

CKPT="${EXP}/${DIR}/checkpoint-350000/ema_model/pytorch_model.bin"
CFGP="${FLOWTOK_ROOT}/configs/${CFG}"
PKL="/g/data/kl02/yh0308/Data/71/_archive_filelists_20260605/dataset_filelist_i2i_test_202407_202507_nofilter_clip16.pkl"
OUT="${EXP}/${DIR}/test_350k_ema_nfclip16"

echo "RUN=$RUN DIR=$DIR"
for f in "$CKPT" "$CFGP" "$PKL"; do [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }; done
mkdir -p "$OUT"

python -u scripts/test_flowtitok_ae.py \
  --config "$CFGP" --checkpoint "$CKPT" --out_dir "$OUT" \
  --filelist_path "$PKL" --split test \
  --max_batches_metrics -1 --max_batches_images 2 \
  --seed 42 --lpips_net alex \
  && echo "DONE ${RUN} -> ${OUT}/metrics.json" || { echo "FAIL ${RUN}"; exit 1; }

echo "===== per-threshold FSS (${RUN}) ====="
python - "${OUT}/metrics.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
thr=d.get("fss_thresholds")
print("avg_fss=",d.get("avg_fss")," avg_fss_excl_lgt=",d.get("avg_fss_excl_lgt"),
      " avg_rfid=",d.get("avg_rfid")," avg_lpips=",d.get("avg_lpips"))
for key in ("fss_per_threshold","fss_per_threshold_ir","fss_per_threshold_lgt"):
    v=d.get(key)
    if v is None: continue
    print(key+":", [f"{t:.0f}:{f:.4f}" for t,f in zip(thr,v)])
PY
