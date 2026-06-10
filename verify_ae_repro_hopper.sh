#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=02:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=30GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N ae_repro_verify

# Determinism verification for the AE test path: run test_flowtitok_ae.py TWICE
# on the same checkpoint + same N batches, then diff the two metrics.json.
# If the repro patch works, run A and run B must be BIT-IDENTICAL.
# Uses the radar bl128 GAN ckpt-300000 (single-channel = fastest).

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
cd "${FLOWTOK_ROOT}"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

EXP="/scratch/kl02/$USER/Projv2v/Experiments/radar_flowtitok_ae_bl128_vae_scratch_run4ftgan_cond1_gadi"
CFG="${FLOWTOK_ROOT}/configs/radar_flowtitok_ae_bl128_vae_scratch_run4ftgan_cond1_gadi.yaml"
CKPT="${EXP}/checkpoint-300000/ema_model/pytorch_model.bin"
PKL="/g/data/kl02/yh0308/Data/71/_archive_filelists_20260605/dataset_filelist_i2i_test_202407_202507_nofilter_clip16.pkl"
NB=4   # batches for metrics (small = fast; determinism holds regardless of N)

for f in "$CKPT" "$PKL"; do [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }; done

run_once () {
  local tag="$1"
  local out="${EXP}/repro_check_${tag}"
  rm -rf "$out"; mkdir -p "$out"
  echo "===== RUN ${tag} (seed 42, ${NB} batches) -> ${out} ====="
  python -u scripts/test_flowtitok_ae.py \
    --config "$CFG" --checkpoint "$CKPT" --out_dir "$out" \
    --filelist_path "$PKL" --split test \
    --max_batches_metrics "$NB" --max_batches_images 0 \
    --seed 42 --lpips_net alex \
    && echo "DONE ${tag}" || { echo "FAIL ${tag}"; exit 1; }
}

run_once A
run_once B

echo ""
echo "================= DETERMINISM DIFF ================="
python - "${EXP}/repro_check_A/metrics.json" "${EXP}/repro_check_B/metrics.json" <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
scalar=lambda d:{k:v for k,v in d.items() if isinstance(v,(int,float))}
sa,sb=scalar(a),scalar(b)
keys=sorted(set(sa)|set(sb)); bad=[]
for k in keys:
    va,vb=sa.get(k),sb.get(k)
    if va is None or vb is None or va!=vb:
        bad.append((k,va,vb))
print(f"compared {len(keys)} scalar metrics")
if not bad:
    print("RESULT: ✅ BIT-IDENTICAL across both runs — AE test path is reproducible.")
else:
    print("RESULT: ❌ MISMATCH in:")
    for k,va,vb in bad: print(f"   {k}: A={va}  B={vb}  Δ={None if (va is None or vb is None) else vb-va}")
    sys.exit(2)
PY
