#!/bin/bash
# sim_weight sweep orchestrator — results-doc §5 follow-up.
# Per cell: joint AE (15k) -> alignment eval -> v2v FlowTok-S (8k) ->
# DECODE to dBZ radar (the decisive metric) -> prune heavy ckpts.
# w=0 (=A) and w=0.5 (=B) are reused as anchors, not retrained.
#
# Runs in tmux, single GPU 0, continue-on-failure. §7 of the results doc is
# re-written after every cell so partial progress is always visible.
set -u
ROOT="/mnt/ssd_1/yghu/Code/FlowTok"
EXP="/mnt/ssd_2/yghu/Experiments"
TEST_PKL="/mnt/ssd_1/yghu/Data/71_3m/filelists/dataset_filelist_v2v_test_202407.pkl"
ALIGN_DIR="${EXP}/joint_tok_align"
MLOG="/tmp/jointtok_sweep.log"
mkdir -p "${ALIGN_DIR}"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "${MLOG}"; }
run(){ local n="$1"; shift; log "=== START ${n} ==="
  if "$@" >>"${MLOG}" 2>&1; then log "=== OK ${n} ==="; return 0
  else log "=== FAIL ${n} (exit $?) ==="; return 1; fi; }

source /home/yghu/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

# Wait for GPU 0 (mem < 2 GiB and no other heavy job), 6h cap.
log "Waiting for GPU 0..."
W=0
while :; do
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
  BUSY=$(pgrep -fc "train_joint_sat_radar_ae|train_sat2radar_v2v|test_sat2radar_v2v|eval_joint_tokenizer" || true)
  [ "${MEM:-99999}" -lt 2000 ] && [ "${BUSY:-0}" -eq 0 ] && { log "GPU 0 free (${MEM} MiB)."; break; }
  [ "${W}" -ge 21600 ] && { log "Waited 6h, proceeding."; break; }
  sleep 120; W=$((W+120))
done

# Seed §7 with the anchors before any new cell finishes.
run ANALYZE_0 python scripts/analyze_jointtok_sweep.py

for TW in w002:0.02 w005:0.05 w010:0.1 w025:0.25; do
  T="${TW%%:*}"; W="${TW##*:}"
  AE="${EXP}/joint_ae_sweep_${T}_run1"
  V2V="${EXP}/v2v_jointtok_sweep_${T}_run1"
  AECFG="configs/joint_ae_sweep_${T}_lab2.yaml"
  VCFG="configs/Sat2Radar-v2v-jointtok-sweep-${T}-FlowTiTok-S.py"
  log "########## CELL ${T} (sim_weight=${W}) ##########"

  run AE_${T} accelerate launch --num_processes 1 \
      scripts/train_joint_sat_radar_ae.py --config="${AECFG}"

  # Resolve best_val ckpt (fall back to final so a missing best_val still evals).
  for M in sat radar; do
    bv="${AE}/${M}/checkpoint-best_val/pytorch_model.bin"
    fn="${AE}/${M}/checkpoint-final/pytorch_model.bin"
    [ -f "${bv}" ] || { [ -f "${fn}" ] && { mkdir -p "${AE}/${M}/checkpoint-best_val"; cp -f "${fn}" "${bv}"; log "best_val<-final for ${T}/${M}"; }; }
  done
  SAT="${AE}/sat/checkpoint-best_val/pytorch_model.bin"
  RAD="${AE}/radar/checkpoint-best_val/pytorch_model.bin"

  if [ -f "${SAT}" ] && [ -f "${RAD}" ]; then
    run EVAL_${T} python scripts/eval_joint_tokenizer.py \
        --joint_config "${AECFG}" --sat_ckpt "${SAT}" --radar_ckpt "${RAD}" \
        --filelist "${TEST_PKL}" --split test --n_samples 512 \
        --tag "${T}" --out_dir "${ALIGN_DIR}"

    run V2V_${T} accelerate launch --num_processes 1 \
        scripts/train_sat2radar_v2v.py --config="${VCFG}"

    CK="${V2V}/ckpts/8000.ckpt"
    [ -f "${CK}" ] || CK="${V2V}/ckpts/4000.ckpt"
    if [ -f "${CK}" ]; then
      run DECODE_${T} python -u scripts/test_sat2radar_v2v.py \
          --config "${VCFG}" --ckpt "${CK}" --out_dir "${V2V}/test8000" \
          --mode v2v --split test --filelist_path "${TEST_PKL}" \
          --batch_size 8 --max_batches_metrics 50 --max_batches_images 4 \
          --skip_gen_metrics --gpu 0 --metrics_json "${V2V}/test8000/metrics.json"
    else
      log "SKIP DECODE_${T}: no v2v ckpt at ${V2V}/ckpts/"
    fi
  else
    log "SKIP EVAL/V2V_${T}: AE ckpt missing (${SAT} / ${RAD})"
  fi

  # Prune heavy ckpts now that metrics + align json are captured (disk bounded;
  # test8000/metrics.json + samples + align_${T}.json are kept).
  rm -rf "${AE}/sat/checkpoint-"* "${AE}/radar/checkpoint-"* "${V2V}/ckpts" 2>/dev/null
  log "pruned heavy ckpts for ${T}"
  run ANALYZE_${T} python scripts/analyze_jointtok_sweep.py
done

run ANALYZE_FINAL python scripts/analyze_jointtok_sweep.py
log "ALL DONE. §7 in docs/specs/results/2026-05-18-joint-tokenizer-results.md"
