#!/usr/bin/env bash
set -euo pipefail

# Small natural-image dataset with ImageNet-like folder structure (train/val).
# Source: fast.ai Imagenette (10-class ImageNet subset).
#
# Default output:
#   /mnt/ssd_1/yghu/Data/imagenette2-160/
#
# Usage:
#   bash scripts/download_imagenette2_160.sh
#   bash scripts/download_imagenette2_160.sh /custom/output/dir

OUT_ROOT="${1:-/mnt/ssd_1/yghu/Data}"
NAME="imagenette2-160"
URL="https://s3.amazonaws.com/fast-ai-imageclas/${NAME}.tgz"

mkdir -p "${OUT_ROOT}"
cd "${OUT_ROOT}"

if [[ -d "${OUT_ROOT}/${NAME}/train" && -d "${OUT_ROOT}/${NAME}/val" ]]; then
  echo "[OK] Dataset already exists: ${OUT_ROOT}/${NAME}"
  exit 0
fi

ARCHIVE="${OUT_ROOT}/${NAME}.tgz"

echo "[DL] Downloading ${NAME} -> ${ARCHIVE}"
if command -v wget >/dev/null 2>&1; then
  wget -O "${ARCHIVE}" "${URL}"
else
  curl -L -o "${ARCHIVE}" "${URL}"
fi

echo "[UNPACK] Extracting ${ARCHIVE} -> ${OUT_ROOT}/${NAME}"
tar -xzf "${ARCHIVE}" -C "${OUT_ROOT}"

if [[ "${KEEP_ARCHIVE:-0}" != "1" ]]; then
  rm -f "${ARCHIVE}"
  echo "[CLEAN] Removed archive (set KEEP_ARCHIVE=1 to keep): ${ARCHIVE}"
fi

echo "[DONE] ${NAME} is ready at: ${OUT_ROOT}/${NAME}"
echo "       train dir: ${OUT_ROOT}/${NAME}/train"
echo "       val   dir: ${OUT_ROOT}/${NAME}/val"

