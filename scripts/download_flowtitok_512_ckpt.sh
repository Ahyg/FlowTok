#!/usr/bin/env bash
set -euo pipefail

# Download FlowTok image tokenizer checkpoint (FlowTiTok_512.bin) from Hugging Face.
# Source: https://huggingface.co/turkeyju/FlowTok/tree/main
#
# Default output:
#   /mnt/ssd_1/yghu/Data/flowtok_ckpts/FlowTiTok_512.bin
#
# Usage:
#   bash scripts/download_flowtitok_512_ckpt.sh
#   bash scripts/download_flowtitok_512_ckpt.sh /custom/ckpt/dir

OUT_DIR="${1:-/mnt/ssd_1/yghu/Data/flowtok_ckpts}"
FILENAME="FlowTiTok_512.bin"
URL="https://huggingface.co/turkeyju/FlowTok/resolve/main/${FILENAME}"

mkdir -p "${OUT_DIR}"
OUT_PATH="${OUT_DIR}/${FILENAME}"

if [[ -f "${OUT_PATH}" ]]; then
  echo "[OK] Checkpoint already exists: ${OUT_PATH}"
  exit 0
fi

echo "[DL] Downloading ${FILENAME} -> ${OUT_PATH}"
if command -v wget >/dev/null 2>&1; then
  wget --continue -O "${OUT_PATH}.tmp" "${URL}"
else
  curl -L -C - -o "${OUT_PATH}.tmp" "${URL}"
fi

mv "${OUT_PATH}.tmp" "${OUT_PATH}"
echo "[DONE] Saved checkpoint: ${OUT_PATH}"

