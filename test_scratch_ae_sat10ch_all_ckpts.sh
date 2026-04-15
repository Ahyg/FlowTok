#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N test_ae_sat10_all

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
TEST_FILELIST="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_test_202407_202507_nofilter.pkl"

FSS_THRESHOLDS="0,5,10,15,20,25,30,35,40,45,50,55,60"
FSS_SCALES="1,2,3,4,5,6,7,8,9,10"

SAT10_CFG="${FLOWTOK_ROOT}/configs/sat10ch_flowtitok_ae_bl77_vae_scratch_gadi.yaml"

cd "${FLOWTOK_ROOT}"
export CUDA_VISIBLE_DEVICES=0

run_test() {
  local cfg="$1"; local exp="$2"; local step="$3"; local kname
  if [[ "$step" == "50000" ]]; then kname="50k";
  elif [[ "$step" == "100000" ]]; then kname="100k";
  elif [[ "$step" == "150000" ]]; then kname="150k";
  elif [[ "$step" == "200000" ]]; then kname="200k";
  else kname="${step}"; fi
  local ckpt="${EXP_ROOT}/${exp}/checkpoint-${step}/ema_model/pytorch_model.bin"
  local out_dir="${EXP_ROOT}/${exp}/test_${kname}_ema"
  local metrics="${out_dir}/metrics.json"
  if [[ ! -f "$ckpt" ]]; then echo "SKIP (no ckpt): ${exp} @ ${kname}"; return 0; fi
  if [[ -f "$metrics" ]]; then echo "SKIP (already done): ${exp} @ ${kname}"; return 0; fi
  echo "========== RUN ${exp} @ ${kname} =========="
  python scripts/test_flowtitok_ae.py \
    --config="${cfg}" \
    --checkpoint="${ckpt}" \
    --filelist_path="${TEST_FILELIST}" \
    --split=test \
    --out_dir="${out_dir}" \
    --max_batches_images=10 \
    --fss_thresholds="${FSS_THRESHOLDS}" \
    --fss_scales="${FSS_SCALES}"
}

for step in 50000 100000 150000 200000; do
  run_test "${SAT10_CFG}" "sat10ch_flowtitok_ae_bl77_vae_scratch_run1_gadi" "${step}"
done
for step in 50000 100000 150000 200000; do
  run_test "${SAT10_CFG}" "sat10ch_flowtitok_ae_bl77_vae_scratch_run2_gadi" "${step}"
done

echo "========== ALL DONE (sat10ch) =========="
