#!/bin/bash
# scripts/eval_gan_sweep.sh
# Run the test_flowtitok_ae.py harness on every late-GAN sweep cell that has a
# checkpoint, on the 2024/07 held-out test set. Sibling of eval_recipe_sweep.sh,
# retargeted to the GAN sweep (Stage 3): configs live at configs/<cell>.yaml (no
# rs_ prefix) and checkpoints under .../ae_recipe_sweep/gan/<cell>/.
#
# The harness emits overall + per-channel + FSS/CSI/POD/FAR/r2/rFID into
# metrics.json + metrics.md per cell — the disc_weight-vs-metrics table, with
# radar_gan_nogan as the GAN-off baseline.
#
# Idempotent: skips cells with no config, no checkpoint (still training), or an
# existing metrics.json. Safe to re-run as cells finish.
#
# CRITICAL (lab2): root LV (/, /tmp) is 100% full. Force scratch onto ssd_2.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_gan_sweep.sh [SLOT]
#     SLOT defaults to best_val (also accepts: final)

set -uo pipefail   # NOT -e: one failing eval must not abort the rest

FLOWTOK_ROOT="/mnt/ssd_1/yghu/Code/FlowTok"
EXP_ROOT="/mnt/ssd_2/yghu/Experiments/ae_recipe_sweep/gan"
TEST_PKL="/mnt/ssd_1/yghu/Data/71_3m/filelists/dataset_filelist_i2i_test_202407_nofilter.pkl"
SLOT="${1:-best_val}"
LOG="${EXP_ROOT}/eval_${SLOT}.log"

# Keep ALL scratch off the full root FS (see memory: lab2-root-fs-full).
export TMPDIR=/mnt/ssd_2/yghu/tmp
export MPLCONFIGDIR=/mnt/ssd_2/yghu/tmp/mpl
export TRITON_CACHE_DIR=/mnt/ssd_2/yghu/tmp/triton
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"

source "/home/yghu/miniconda3/etc/profile.d/conda.sh"
conda activate flowtok
cd "${FLOWTOK_ROOT}"
export PYTHONUNBUFFERED=1

# nogan (GAN-off baseline) first, then ascending disc_weight.
CELLS=(
  radar_gan_nogan
  radar_gan_w0001
  radar_gan_w001
  radar_gan_w01
  radar_gan_w1
)

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

log "eval_gan_sweep start — slot=${SLOT}, ${#CELLS[@]} cells, test=${TEST_PKL##*/}"

n_done=0 n_skip=0 n_fail=0
for cell in "${CELLS[@]}"; do
  cfg="${FLOWTOK_ROOT}/configs/${cell}.yaml"
  ckpt="${EXP_ROOT}/${cell}/checkpoint-${SLOT}/unwrapped_model/pytorch_model.bin"
  out="${EXP_ROOT}/${cell}/eval_${SLOT}"
  if [[ ! -f "$cfg" ]];  then log "SKIP ${cell}: no config";                    n_skip=$((n_skip+1)); continue; fi
  if [[ ! -f "$ckpt" ]]; then log "SKIP ${cell}: no ${SLOT} ckpt (training?)";   n_skip=$((n_skip+1)); continue; fi
  if [[ -f "${out}/metrics.json" ]]; then log "SKIP ${cell}: already evaluated"; n_skip=$((n_skip+1)); continue; fi
  mkdir -p "$out"
  log "EVAL ${cell} -> ${out}"
  if python -u scripts/test_flowtitok_ae.py \
       --config "$cfg" \
       --checkpoint "$ckpt" \
       --out_dir "$out" \
       --max_batches_metrics -1 \
       --max_batches_images 2 \
       --split test \
       --filelist_path "${TEST_PKL}" \
       --lpips_net alex \
       >> "${LOG}" 2>&1; then
    log "DONE ${cell}"; n_done=$((n_done+1))
  else
    log "FAIL ${cell} (see ${LOG})"; n_fail=$((n_fail+1))
  fi
done

log "eval_gan_sweep ALL DONE — slot=${SLOT}: ${n_done} evaluated, ${n_skip} skipped, ${n_fail} failed"
