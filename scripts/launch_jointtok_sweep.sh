#!/bin/bash
# sim_weight sweep — REDUCED set, BIG budget (the 8k pilot was visually
# unconverged: neg R², CSI35=0, low-freq blobs). Per the user's call:
#   cells w ∈ {0.0 separate, 0.05, 0.25} ; AE 25k ; v2v 40k
# Per cell: joint AE -> alignment eval + tokenizer-ceiling recon panel ->
# big-budget v2v -> DECODE to dBZ radar (the decisive metric) -> prune ckpts.
#
# tmux, single GPU 0, continue-on-failure. §7 re-written after every cell.
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

log "Waiting for GPU 0..."
W=0
while :; do
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
  BUSY=$(pgrep -fc "train_joint_sat_radar_ae|train_sat2radar_v2v|test_sat2radar_v2v|eval_joint_tokenizer" || true)
  [ "${MEM:-99999}" -lt 2000 ] && [ "${BUSY:-0}" -eq 0 ] && { log "GPU 0 free (${MEM} MiB)."; break; }
  [ "${W}" -ge 21600 ] && { log "Waited 6h, proceeding."; break; }
  sleep 120; W=$((W+120))
done

run ANALYZE_0 python scripts/analyze_jointtok_sweep.py

for TW in w000:0.0 w005:0.05 w025:0.25; do
  T="${TW%%:*}"; WV="${TW##*:}"
  AE="${EXP}/joint_ae_sweep_${T}_run1"
  V2V="${EXP}/v2v_jointtok_sweep_${T}_run1"
  AECFG="configs/joint_ae_sweep_${T}_lab2.yaml"
  VCFG="configs/Sat2Radar-v2v-jointtok-sweep-${T}-FlowTiTok-S.py"
  log "########## CELL ${T} (sim_weight=${WV}) ##########"

  run AE_${T} accelerate launch --num_processes 1 \
      scripts/train_joint_sat_radar_ae.py --config="${AECFG}"

  for M in sat radar; do
    bv="${AE}/${M}/checkpoint-best_val/pytorch_model.bin"
    fn="${AE}/${M}/checkpoint-final/pytorch_model.bin"
    [ -f "${bv}" ] || { [ -f "${fn}" ] && { mkdir -p "${AE}/${M}/checkpoint-best_val"; cp -f "${fn}" "${bv}"; log "best_val<-final ${T}/${M}"; }; }
  done
  SAT="${AE}/sat/checkpoint-best_val/pytorch_model.bin"
  RAD="${AE}/radar/checkpoint-best_val/pytorch_model.bin"

  if [ -f "${SAT}" ] && [ -f "${RAD}" ]; then
    # alignment metrics + tokenizer-ceiling recon panel (pure encode->decode).
    run EVAL_${T} python scripts/eval_joint_tokenizer.py \
        --joint_config "${AECFG}" --sat_ckpt "${SAT}" --radar_ckpt "${RAD}" \
        --filelist "${TEST_PKL}" --split test --n_samples 512 \
        --n_recon_images 6 --tag "${T}" --out_dir "${ALIGN_DIR}"
    [ -f "${ALIGN_DIR}/recon_${T}.png" ] && cp -f "${ALIGN_DIR}/recon_${T}.png" "${AE}/recon_test.png" && log "recon panel -> ${AE}/recon_test.png"

    run V2V_${T} accelerate launch --num_processes 1 \
        scripts/train_sat2radar_v2v.py --config="${VCFG}"

    CK="${V2V}/ckpts/40000.ckpt"
    [ -f "${CK}" ] || CK=$(ls -1t "${V2V}"/ckpts/*.ckpt 2>/dev/null | head -1)
    if [ -n "${CK:-}" ] && [ -f "${CK}" ]; then
      run DECODE_${T} python -u scripts/test_sat2radar_v2v.py \
          --config "${VCFG}" --ckpt "${CK}" --out_dir "${V2V}/test_final" \
          --mode v2v --split test --filelist_path "${TEST_PKL}" \
          --batch_size 8 --max_batches_metrics 50 --max_batches_images 4 \
          --skip_gen_metrics --gpu 0 --metrics_json "${V2V}/test_final/metrics.json"
    else
      log "SKIP DECODE_${T}: no v2v ckpt in ${V2V}/ckpts/"
    fi
  else
    log "SKIP EVAL/V2V_${T}: AE ckpt missing (${SAT} / ${RAD})"
  fi

  # Prune heavy ckpts; keep test_final/ (metrics+panels), recon_test.png,
  # align_${T}.json, log0.txt.
  rm -rf "${AE}/sat/checkpoint-"* "${AE}/radar/checkpoint-"* "${V2V}/ckpts" 2>/dev/null
  log "pruned heavy ckpts for ${T}"
  run ANALYZE_${T} python scripts/analyze_jointtok_sweep.py
done

run ANALYZE_FINAL python scripts/analyze_jointtok_sweep.py
log "ALL DONE. §7 in docs/specs/results/2026-05-18-joint-tokenizer-results.md"
