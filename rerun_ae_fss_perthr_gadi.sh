#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=06:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=30GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N ae_fss_perthr

# Re-evaluate the bl128 AE holdout (46,416-frame nofilter clip16 test set) ONLY to
# recover per-threshold FSS, which the original runs computed internally but never
# persisted (metrics.json kept only the scalar mean avg_fss). The edited
# test_flowtitok_ae.py now writes fss_per_threshold / _ir / _lgt (mean over scales,
# aligned index-wise to fss_thresholds = 0,5,...,60).
#
# FID + LPIPS are skipped: not needed for FSS and they dominate runtime (esp. sat's
# 11-channel InceptionV3). All 6 runs share seed 42 + deterministic VAE noise, so the
# 200k/250k/300k comparison is apples-to-apples. As a correctness check, the re-run
# avg_fss must match the stored metrics.json scalar to ~3 decimals.
#
# Submit one job per checkpoint:  for r in radar200k radar250k radar300k sat200k sat250k sat300k; do qsub -v RUN=$r rerun_ae_fss_perthr_gadi.sh; done

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

RUN="${RUN:?must pass -v RUN=<radar200k|radar250k|radar300k|sat200k|sat250k|sat300k>}"
case "$RUN" in
  radar200k) DIR=radar_flowtitok_ae_bl128_vae_scratch_run4_gadi;            STEP=200000;;
  radar250k) DIR=radar_flowtitok_ae_bl128_vae_scratch_run4ft_cond1_gadi;    STEP=250000;;
  radar300k) DIR=radar_flowtitok_ae_bl128_vae_scratch_run4ftgan_cond1_gadi; STEP=300000;;
  sat200k)   DIR=sat10ch_flowtitok_ae_bl128_vae_scratch_run4B_gadi;          STEP=200000;;
  sat250k)   DIR=sat10ch_flowtitok_ae_bl128_vae_scratch_run4Bft_cond1_gadi;  STEP=250000;;
  sat300k)   DIR=sat10ch_flowtitok_ae_bl128_vae_scratch_run4Bftgan_cond1_gadi;STEP=300000;;
  *) echo "BAD RUN=$RUN"; exit 2;;
esac

CFG="${FLOWTOK_ROOT}/configs/${DIR}.yaml"
CKPT="${EXP}/${DIR}/checkpoint-${STEP}/ema_model/pytorch_model.bin"
PKL="/g/data/kl02/yh0308/Data/71/_archive_filelists_20260605/dataset_filelist_i2i_test_202407_202507_nofilter_clip16.pkl"
OUT="${EXP}/${DIR}/fss_perthr_${STEP}"

echo "RUN=$RUN  DIR=$DIR  STEP=$STEP"
for f in "$CFG" "$CKPT" "$PKL"; do [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }; done
mkdir -p "$OUT"

python -u scripts/test_flowtitok_ae.py \
  --config "$CFG" --checkpoint "$CKPT" --out_dir "$OUT" \
  --filelist_path "$PKL" --split test \
  --max_batches_metrics -1 --max_batches_images 0 \
  --skip_fid --skip_lpips --seed 42 \
  && echo "DONE ${RUN} -> ${OUT}/metrics.json" || { echo "FAIL ${RUN}"; exit 1; }

echo "================= per-threshold FSS ================="
python - "${OUT}/metrics.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
thr=d.get("fss_thresholds")
print("avg_fss        =", d.get("avg_fss"))
print("avg_fss_excl_lgt=", d.get("avg_fss_excl_lgt"))
for key in ("fss_per_threshold","fss_per_threshold_ir","fss_per_threshold_lgt"):
    v=d.get(key)
    if v is None: continue
    print(f"\n{key}:")
    for t,f in zip(thr,v):
        print(f"  thr={t:5.0f}  FSS={f:.4f}")
PY
