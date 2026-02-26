#!/usr/bin/env bash
set -euo pipefail

# Run FlowTok text-to-image evaluation with custom prompts or MJHQ dataset
#
# Usage:
#   # Use custom prompts (default, 10 prompts)
#   bash eval_mjhq.sh
#
#   # Use MJHQ dataset
#   USE_CUSTOM_PROMPTS=false MJHQ_META_PATH=/path/to/meta_data.json bash eval_mjhq.sh
#
# Optional overrides:
#   CUDA_VISIBLE_DEVICES=0 bash eval_mjhq.sh
#   WORKDIR=/path/to/workdir bash eval_mjhq.sh

ROOT_DIR="/mnt/ssd_1/yghu/Code/FlowTok"

# Paths
DATA_PARENT_DIR="/mnt/ssd_1/yghu/Data"
CKPT_DIR="${CKPT_DIR:-${DATA_PARENT_DIR}/flowtok_ckpts}"
FLOWTOK_CKPT="${FLOWTOK_CKPT:-${CKPT_DIR}/FlowTok-XL.pth}"

CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/FlowTok-XL-Stage3.py}"
WORKDIR="${WORKDIR:-/mnt/ssd_1/yghu/Experiments/flowtok_t2i_demo}"

# Use custom prompts by default
USE_CUSTOM_PROMPTS="${USE_CUSTOM_PROMPTS:-true}"

# MJHQ data path - only needed if USE_CUSTOM_PROMPTS=false
MJHQ_META_PATH="${MJHQ_META_PATH:-}"
if [[ "${USE_CUSTOM_PROMPTS}" != "true" ]]; then
  if [[ -z "${MJHQ_META_PATH}" ]]; then
    # Try common locations
    for alt_path in \
      "${DATA_PARENT_DIR}/MJHQ-30K/meta_data.json" \
      "${WORKDIR}/MJHQ-30K/meta_data.json" \
      "/opt/tiger/ju/data/MJHQ-30K/meta_data.json"; do
      if [[ -f "${alt_path}" ]]; then
        MJHQ_META_PATH="${alt_path}"
        break
      fi
    done
  fi
  
  if [[ -z "${MJHQ_META_PATH}" ]] || [[ ! -f "${MJHQ_META_PATH}" ]]; then
    echo "[ERROR] MJHQ meta_data.json not found"
    echo "Please set MJHQ_META_PATH environment variable or place the file in one of:"
    echo "  - ${DATA_PARENT_DIR}/MJHQ-30K/meta_data.json"
    echo "  - ${WORKDIR}/MJHQ-30K/meta_data.json"
    echo "  - /opt/tiger/ju/data/MJHQ-30K/meta_data.json"
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Check required files
if [[ ! -f "${FLOWTOK_CKPT}" ]]; then
  echo "[ERROR] FlowTok checkpoint not found: ${FLOWTOK_CKPT}"
  echo "Please download it or set FLOWTOK_CKPT environment variable"
  exit 1
fi

echo "[INFO] Using FlowTok checkpoint: ${FLOWTOK_CKPT}"
if [[ "${USE_CUSTOM_PROMPTS}" == "true" ]]; then
  echo "[INFO] Mode: Custom prompts (10 prompts)"
else
  echo "[INFO] Mode: MJHQ dataset"
  echo "[INFO] Using MJHQ data: ${MJHQ_META_PATH}"
fi
echo "[INFO] Work directory: ${WORKDIR}"
echo "[INFO] Config: ${CONFIG_PATH}"

# Create workdir
mkdir -p "${WORKDIR}"

# Run evaluation
if [[ "${USE_CUSTOM_PROMPTS}" == "true" ]]; then
  echo "[RUN] Starting evaluation with custom prompts..."
else
  echo "[RUN] Starting MJHQ evaluation..."
  export MJHQ_META_PATH="${MJHQ_META_PATH}"
fi

export USE_CUSTOM_PROMPTS="${USE_CUSTOM_PROMPTS}"

cd "${ROOT_DIR}"

accelerate launch --num_processes=1 \
  eval_t2i_mjhq.py \
  --config="${CONFIG_PATH}" \
  --workdir="${WORKDIR}"

if [[ "${USE_CUSTOM_PROMPTS}" == "true" ]]; then
  echo "[DONE] Evaluation complete. Results saved to: ${WORKDIR}/custom_eval"
else
  echo "[DONE] Evaluation complete. Results saved to: ${WORKDIR}/mjhq_eval"
  echo "[DONE] FID scores saved to: ${WORKDIR}/mjhq_eval/fid_scores.txt"
fi
