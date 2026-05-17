#!/bin/bash
# Staged overnight orchestrator for the joint sat/radar tokenizer experiment.
# Design: docs/specs/2026-05-18-joint-sat-radar-tokenizer-design.md
#
# Run inside tmux (the launcher does this). Single GPU 0. Continue-on-failure
# AFTER the AE phase so the primary tokenizer result survives even if v2v dies.
set -u

ROOT="/mnt/ssd_1/yghu/Code/FlowTok"
EXP="/mnt/ssd_2/yghu/Experiments"
TEST_PKL="/mnt/ssd_1/yghu/Data/71_3m/filelists/dataset_filelist_v2v_test_202407.pkl"
ALIGN_DIR="${EXP}/joint_tok_align"
MLOG="/tmp/joint_tok_orchestrator.log"

mkdir -p "${ALIGN_DIR}"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "${MLOG}"; }

source /home/yghu/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

run(){  # run <name> <cmd...> ; logs, never aborts the script
  local name="$1"; shift
  log "=== START ${name} ==="
  if "$@" >>"${MLOG}" 2>&1; then log "=== OK ${name} ==="; return 0
  else log "=== FAIL ${name} (exit $?) ==="; return 1; fi
}

# ── 0. Wait for GPU 0 (token-sweep eval is using it) ────────────────────────
log "Waiting for GPU 0 to free (token-sweep eval)..."
WAITED=0
while :; do
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
  BUSY=$(pgrep -fc "eval_sat_ae_token_sweep|test_flowtitok_ae|latent_utilization|diagnose_ae_token_sweep" || true)
  if [ "${MEM:-99999}" -lt 2000 ] && [ "${BUSY:-0}" -eq 0 ]; then
    log "GPU 0 free (mem=${MEM} MiB). Starting."; break
  fi
  if [ "${WAITED}" -ge 21600 ]; then log "Waited 6h, proceeding anyway."; break; fi
  sleep 120; WAITED=$((WAITED+120))
done

# ── 1. AE phase (prerequisite) ──────────────────────────────────────────────
run AE_A accelerate launch --num_processes 1 scripts/train_joint_sat_radar_ae.py \
    --config=configs/joint_ae_sep_lab2.yaml
run AE_B accelerate launch --num_processes 1 scripts/train_joint_sat_radar_ae.py \
    --config=configs/joint_ae_joint_lab2.yaml

# ── 2. Tokenizer-phase eval (the primary result) ────────────────────────────
for G in A:joint_ae_sep_run1 B:joint_ae_joint_run1; do
  TAG="${G%%:*}"; RUN="${G##*:}"
  SAT="${EXP}/${RUN}/sat/checkpoint-best_val/pytorch_model.bin"
  RAD="${EXP}/${RUN}/radar/checkpoint-best_val/pytorch_model.bin"
  [ -f "${SAT}" ] || SAT="${EXP}/${RUN}/sat/checkpoint-final/pytorch_model.bin"
  [ -f "${RAD}" ] || RAD="${EXP}/${RUN}/radar/checkpoint-final/pytorch_model.bin"
  if [ -f "${SAT}" ] && [ -f "${RAD}" ]; then
    run EVAL_${TAG} python scripts/eval_joint_tokenizer.py \
      --joint_config configs/joint_ae_sep_lab2.yaml \
      --sat_ckpt "${SAT}" --radar_ckpt "${RAD}" \
      --filelist "${TEST_PKL}" --split test --n_samples 512 \
      --tag "${TAG}" --out_dir "${ALIGN_DIR}"
  else
    log "SKIP EVAL_${TAG}: ckpt missing (${SAT} / ${RAD})"
  fi
done
run ANALYZE_AE python scripts/analyze_joint_tok.py --phase ae \
    --align_dir "${ALIGN_DIR}"
log "Tokenizer-phase result written to docs/specs/results/2026-05-18-joint-tokenizer-results.md"

# ── 3. v2v pilot (secondary; failures must not lose the AE result) ──────────
run V2V_A accelerate launch --num_processes 1 scripts/train_sat2radar_v2v.py \
    --config=configs/Sat2Radar-v2v-jointtok-A-FlowTiTok-S.py
run V2V_B accelerate launch --num_processes 1 scripts/train_sat2radar_v2v.py \
    --config=configs/Sat2Radar-v2v-jointtok-B-FlowTiTok-S.py
run ANALYZE_FINAL python scripts/analyze_joint_tok.py --phase final \
    --align_dir "${ALIGN_DIR}" \
    --v2v_a_workdir "${EXP}/v2v_jointtok_A_run1" \
    --v2v_b_workdir "${EXP}/v2v_jointtok_B_run1"

log "ALL DONE. Results: ${ROOT}/docs/specs/results/2026-05-18-joint-tokenizer-results.md"
