#!/bin/bash
# scripts/eval_sat_ae_token_sweep.sh
# After training: run test_flowtitok_ae.py + latent_utilization.py for each cell × {best_val, final}.

set -euo pipefail
FLOWTOK_ROOT="/mnt/ssd_1/yghu/Code/FlowTok"
EXP_ROOT="/mnt/ssd_2/yghu/Experiments"
TEST_PKL="/mnt/ssd_1/yghu/Data/71_3m/filelists/dataset_filelist_i2i_test_202407_nofilter.pkl"

source "/home/yghu/miniconda3/etc/profile.d/conda.sh"
conda activate flowtok
cd "${FLOWTOK_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

for TAG in tok77 tok128 tok256; do
  CELL_DIR="${EXP_ROOT}/sat10ch_ae_${TAG}_run1"
  CONFIG="${FLOWTOK_ROOT}/configs/sat10ch_ae_${TAG}_lab2.yaml"
  for SLOT in best_val final; do
    CKPT="${CELL_DIR}/checkpoint-${SLOT}"
    if [ ! -d "${CKPT}" ]; then
      echo "[$(date '+%F %T')] SKIP ${TAG}/${SLOT}: no ckpt"
      continue
    fi
    OUT="${CELL_DIR}/eval_${SLOT}"
    mkdir -p "${OUT}"
    echo "[$(date '+%F %T')] === ${TAG} / ${SLOT} ==="

    # Reconstruction metrics
    python -u scripts/test_flowtitok_ae.py \
      --config "${CONFIG}" \
      --checkpoint "${CKPT}/unwrapped_model/pytorch_model.bin" \
      --out_dir "${OUT}" \
      --max_batches_metrics -1 \
      --max_batches_images 2 \
      --split test \
      --filelist_path "${TEST_PKL}" \
      --lpips_net alex

    # Latent utilization
    python -u scripts/latent_utilization.py \
      --config "${CONFIG}" \
      --ckpt "${CKPT}" \
      --filelist "${TEST_PKL}" \
      --split test \
      --n_samples 256 \
      --out_dir "${OUT}"
  done
done

# 3-cell diagnostic comparison (uses best_val ckpts)
echo "[$(date '+%F %T')] === 3-cell diagnostic ==="
DIAG_OUT="${EXP_ROOT}/sat_ae_token_sweep_diagnostic"
mkdir -p "${DIAG_OUT}"
python -u scripts/diagnose_ae_token_sweep.py \
  --cells '[
    {"name":"tok77","config":"'${FLOWTOK_ROOT}'/configs/sat10ch_ae_tok77_lab2.yaml","ckpt":"'${EXP_ROOT}'/sat10ch_ae_tok77_run1/checkpoint-best_val"},
    {"name":"tok128","config":"'${FLOWTOK_ROOT}'/configs/sat10ch_ae_tok128_lab2.yaml","ckpt":"'${EXP_ROOT}'/sat10ch_ae_tok128_run1/checkpoint-best_val"},
    {"name":"tok256","config":"'${FLOWTOK_ROOT}'/configs/sat10ch_ae_tok256_lab2.yaml","ckpt":"'${EXP_ROOT}'/sat10ch_ae_tok256_run1/checkpoint-best_val"}
  ]' \
  --filelist "${TEST_PKL}" \
  --split test \
  --n_samples 8 \
  --out_dir "${DIAG_OUT}"

echo "[$(date '+%F %T')] All eval done."
ls -la "${EXP_ROOT}"/sat10ch_ae_*_run1/eval_*/metrics.json "${DIAG_OUT}"/ 2>&1
