#!/usr/bin/env bash
set -euo pipefail

# One-command end-to-end:
#  - download Imagenette (small natural image dataset)
#  - download FlowTiTok tokenizer ckpt from Hugging Face
#  - run tokenizer reconstruction eval and save metrics/images
#
# Usage:
#   bash test_flowtitok_tokenizer_imagenette.sh
#
# Optional overrides:
#   CUDA_VISIBLE_DEVICES=0 bash test_flowtitok_tokenizer_imagenette.sh
#   CKPT_DIR=/path/to/ckpts OUT_DIR=/path/to/out bash test_flowtitok_tokenizer_imagenette.sh

ROOT_DIR="/mnt/ssd_1/yghu/Code/FlowTok"

DATA_PARENT_DIR="/mnt/ssd_1/yghu/Data"
DATA_ROOT="${DATA_ROOT:-${DATA_PARENT_DIR}/imagenette2-160}"

CKPT_DIR="${CKPT_DIR:-${DATA_PARENT_DIR}/flowtok_ckpts}"
CKPT_PATH="${CKPT_PATH:-${CKPT_DIR}/FlowTiTok_512.bin}"

CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/FlowTok-XL-Stage3.py}"

OUT_DIR="${OUT_DIR:-/mnt/ssd_1/yghu/Experiments/flowtok_tokenizer_imagenette_eval}"
SPLIT="${SPLIT:-val}"

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES_METRICS="${MAX_BATCHES_METRICS:-50}"
MAX_BATCHES_IMAGES="${MAX_BATCHES_IMAGES:-10}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# NOTE: This script now assumes you have already activated a proper Python/conda env
# (e.g. `conda activate flowtok`) with torch, torchvision, and other deps installed.

echo "[1/3] Download Imagenette dataset..."
bash "${ROOT_DIR}/scripts/download_imagenette2_160.sh" "${DATA_PARENT_DIR}"

echo "[2/3] Download FlowTiTok tokenizer checkpoint..."
bash "${ROOT_DIR}/scripts/download_flowtitok_512_ckpt.sh" "${CKPT_DIR}"

echo "[3/3] Run tokenizer evaluation..."
mkdir -p "${OUT_DIR}"

python "${ROOT_DIR}/scripts/test_flowtitok_tokenizer_imagenet.py" \
  --config "${CONFIG_PATH}" \
  --ckpt "${CKPT_PATH}" \
  --data_root "${DATA_ROOT}" \
  --split "${SPLIT}" \
  --out_dir "${OUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_batches_metrics "${MAX_BATCHES_METRICS}" \
  --max_batches_images "${MAX_BATCHES_IMAGES}"

echo "[DONE] Results at: ${OUT_DIR}"
