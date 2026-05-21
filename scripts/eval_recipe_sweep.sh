#!/bin/bash
# scripts/eval_recipe_sweep.sh
# Run the existing test_flowtitok_ae.py harness on EVERY recipe-sweep cell that
# has a checkpoint, on the 2024/07 held-out test set. Reuses the test script as
# is — it emits overall + excl-lightning + per-channel(incl ch10) + FSS/CSI/POD/
# FAR/r2/rFID into metrics.json + metrics.md per cell.
#
# Idempotent: skips cells with no config, no checkpoint (still training), or an
# existing metrics.json. Safe to re-run as more cells finish.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_recipe_sweep.sh [SLOT]
#     SLOT defaults to best_val (also accepts: final)

set -uo pipefail   # NOT -e: one failing eval must not abort the rest

FLOWTOK_ROOT="/mnt/ssd_1/yghu/Code/FlowTok"
EXP_ROOT="/mnt/ssd_2/yghu/Experiments/ae_recipe_sweep"
TEST_PKL="/mnt/ssd_1/yghu/Data/71_3m/filelists/dataset_filelist_i2i_test_202407_nofilter.pkl"
SLOT="${1:-best_val}"
LOG="${EXP_ROOT}/eval_${SLOT}.log"

source "/home/yghu/miniconda3/etc/profile.d/conda.sh"
conda activate flowtok
cd "${FLOWTOK_ROOT}"
export PYTHONUNBUFFERED=1

# All cells across the loss-recipe sweep + patch-8 probe + token sweep +
# lightning experiments. Radar cells are 1ch (no excl-lightning columns);
# sat10ch cells produce the full with/without-lightning breakdown.
CELLS=(
  radar_b0          s10_b0
  radar_s1_kl10     s10_s1_kl10
  radar_s1_pc       s10_s1_pc
  radar_s1_p16      s10_s1_p16
  radar_s1_p06      s10_s1_p06
  radar_s1_kl05     s10_s1_kl05
  s10_b0_b8         radar_b0_p8        s10_b0_p8
  radar_tok64       radar_tok77        radar_tok256
  s10_s1_pc_nolgt   s10_lgt10          s10_lgt20
)

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

log "eval_recipe_sweep start — slot=${SLOT}, ${#CELLS[@]} cells, test=${TEST_PKL##*/} (4391 samples)"

n_done=0 n_skip=0 n_fail=0
for cell in "${CELLS[@]}"; do
  cfg="${FLOWTOK_ROOT}/configs/rs_${cell}.yaml"
  ckpt="${EXP_ROOT}/${cell}/checkpoint-${SLOT}/unwrapped_model/pytorch_model.bin"
  out="${EXP_ROOT}/${cell}/eval_${SLOT}"
  if [[ ! -f "$cfg" ]];  then log "SKIP ${cell}: no config";                       n_skip=$((n_skip+1)); continue; fi
  if [[ ! -f "$ckpt" ]]; then log "SKIP ${cell}: no ${SLOT} ckpt (training?)";      n_skip=$((n_skip+1)); continue; fi
  if [[ -f "${out}/metrics.json" ]]; then log "SKIP ${cell}: already evaluated";    n_skip=$((n_skip+1)); continue; fi
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

log "eval_recipe_sweep ALL DONE — slot=${SLOT}: ${n_done} evaluated, ${n_skip} skipped, ${n_fail} failed"
