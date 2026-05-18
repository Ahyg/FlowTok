#!/bin/bash
# InfoNCE sim_weight sweep — BIG budget, MULTI-GPU parallel (design §3.3;
# index-wise cosine was refuted by alignment-by-collapse). Per-index symmetric
# InfoNCE, learnable CLIP log-temp, 1k-step sim-loss warmup. Cells:
#   w ∈ {0.0 separate, 0.1, 0.5, 1.0} ; AE 25k ; v2v 40k
# Each cell is fully independent (one-knob ablation) → run cells in PARALLEL
# across every idle GPU on this shared box. A cell is pinned to one GPU for
# its whole AE→eval→v2v→decode→prune→analyze pipeline; when it finishes the
# GPU is released for the next queued cell. Foreign/busy GPUs are skipped and
# picked up later if they free. §7 rewrites are flock-serialized (shared doc).
# InfoNCE alignment artifacts tagged inf_* so cosine §6.2 ones are preserved.
#
# tmux, continue-on-failure, recoverable ckpts on failure.
set -u
ROOT="/mnt/ssd_1/yghu/Code/FlowTok"
EXP="/mnt/ssd_2/yghu/Experiments"
TEST_PKL="/mnt/ssd_1/yghu/Data/71_3m/filelists/dataset_filelist_v2v_test_202407.pkl"
ALIGN_DIR="${EXP}/joint_tok_align"
MLOG="/tmp/jointtok_infonce_sweep.log"
ALCK="/tmp/jointtok_infonce_analyze.lock"
GPU_MEM_FREE_MIB=2000          # a GPU is "idle" if used mem < this
CELLS=(w000:0.0 w010:0.1 w050:0.5 w100:1.0)
GPUS=(0 1 2 3)
mkdir -p "${ALIGN_DIR}"; : >"${MLOG}"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "${MLOG}"; }

source /home/yghu/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
cd "${ROOT}"
export PYTHONUNBUFFERED=1

# Regenerate the InfoNCE sweep configs fresh (self-contained / reproducible).
if python scripts/gen_sweep_configs.py >>"${MLOG}" 2>&1; then log "GEN_CONFIGS OK"
else log "GEN_CONFIGS FAILED — abort"; exit 1; fi

# §7 rewrite touches the shared results doc — serialize across parallel cells.
analyze(){ flock "${ALCK}" python scripts/analyze_jointtok_sweep.py \
             >>"${MLOG}" 2>&1; }
analyze   # seed §7 (all pending) once up front

# A GPU is usable iff currently idle (no foreign job) — checked at claim time.
gpu_idle(){ local g="$1"
  local m; m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                 -i "$g" 2>/dev/null | tr -d ' ')
  [ -n "${m:-}" ] && [ "${m}" -lt "${GPU_MEM_FREE_MIB}" ]; }

# ── full per-cell pipeline (runs in a subshell, CUDA_VISIBLE_DEVICES pinned) ──
run_cell(){
  local T="$1" WV="$2" G="$3"
  local AE="${EXP}/joint_ae_infonce_${T}_run1"
  local V2V="${EXP}/v2v_jointtok_infonce_${T}_run1"
  local AECFG="configs/joint_ae_infonce_${T}_lab2.yaml"
  local VCFG="configs/Sat2Radar-v2v-jointtok-infonce-${T}-FlowTiTok-S.py"
  local CLOG="${ALIGN_DIR}/cell_${T}.log"; : >"${CLOG}"
  # cell-scoped runner: verbose tool output → cell log; 1-line status → MLOG
  cr(){ local n="$1"; shift; log "[${T}|gpu${G}] START ${n}"
    if "$@" >>"${CLOG}" 2>&1; then log "[${T}|gpu${G}] OK ${n}"; return 0
    else log "[${T}|gpu${G}] FAIL ${n} (exit $?)"; return 1; fi; }

  log "########## CELL ${T} (sim_weight=${WV}) on GPU ${G} ##########"
  cr AE_${T} accelerate launch --num_processes 1 \
      scripts/train_joint_sat_radar_ae.py --config="${AECFG}"

  local M bv fn
  for M in sat radar; do
    bv="${AE}/${M}/checkpoint-best_val/pytorch_model.bin"
    fn="${AE}/${M}/checkpoint-final/pytorch_model.bin"
    [ -f "${bv}" ] || { [ -f "${fn}" ] && { mkdir -p "${AE}/${M}/checkpoint-best_val"; cp -f "${fn}" "${bv}"; log "[${T}] best_val<-final ${M}"; }; }
  done
  local SAT="${AE}/sat/checkpoint-best_val/pytorch_model.bin"
  local RAD="${AE}/radar/checkpoint-best_val/pytorch_model.bin"

  if [ -f "${SAT}" ] && [ -f "${RAD}" ]; then
    cr EVAL_${T} python scripts/eval_joint_tokenizer.py \
        --joint_config "${AECFG}" --sat_ckpt "${SAT}" --radar_ckpt "${RAD}" \
        --filelist "${TEST_PKL}" --split test --n_samples 512 \
        --n_recon_images 6 --tag "inf_${T}" --out_dir "${ALIGN_DIR}"
    [ -f "${ALIGN_DIR}/recon_inf_${T}.png" ] && cp -f "${ALIGN_DIR}/recon_inf_${T}.png" "${AE}/recon_test.png" && log "[${T}] recon panel saved"

    cr V2V_${T} accelerate launch --num_processes 1 \
        scripts/train_sat2radar_v2v.py --config="${VCFG}"

    # a v2v ckpt is a *directory* (…/ckpts/40000.ckpt/) → test with -e, ls -1dt
    local CK="${V2V}/ckpts/40000.ckpt"
    [ -e "${CK}" ] || CK=$(ls -1dt "${V2V}"/ckpts/*.ckpt 2>/dev/null | head -1)
    if [ -n "${CK:-}" ] && [ -e "${CK}" ]; then
      cr DECODE_${T} python -u scripts/test_sat2radar_v2v.py \
          --config "${VCFG}" --ckpt "${CK}" --out_dir "${V2V}/test_final" \
          --mode v2v --split test --filelist_path "${TEST_PKL}" \
          --batch_size 8 --max_batches_metrics 50 --max_batches_images 4 \
          --skip_gen_metrics --gpu 0 --metrics_json "${V2V}/test_final/metrics.json"
    else
      log "[${T}] SKIP DECODE: no v2v ckpt in ${V2V}/ckpts/"
    fi
  else
    log "[${T}] SKIP EVAL/V2V: AE ckpt missing"
  fi

  # Prune heavy ckpts ONLY once DECODE produced metrics.json — else keep them
  # so a failed/contended cell stays recoverable.
  if [ -f "${V2V}/test_final/metrics.json" ]; then
    rm -rf "${AE}/sat/checkpoint-"* "${AE}/radar/checkpoint-"* "${V2V}/ckpts" 2>/dev/null
    log "[${T}] pruned heavy ckpts (decode metrics secured)"
  else
    log "[${T}] KEEP ckpts: no decode metrics — left recoverable"
  fi
  analyze
  log "########## CELL ${T} DONE ##########"
}

# ── scheduler: dispatch cells onto idle GPUs, reap, repeat ───────────────────
declare -A GPU_PID             # gpu -> pid of the cell currently on it
qi=0
log "scheduler start: ${#CELLS[@]} cells, GPUs ${GPUS[*]} (idle<${GPU_MEM_FREE_MIB}MiB)"
while :; do
  # dispatch as many queued cells as there are idle, unclaimed GPUs
  while [ "${qi}" -lt "${#CELLS[@]}" ]; do
    g=""
    for cand in "${GPUS[@]}"; do
      [ -z "${GPU_PID[$cand]:-}" ] && gpu_idle "${cand}" && { g="${cand}"; break; }
    done
    [ -z "${g}" ] && break
    T="${CELLS[$qi]%%:*}"; WV="${CELLS[$qi]##*:}"
    ( export CUDA_VISIBLE_DEVICES="${g}"; run_cell "${T}" "${WV}" "${g}" ) &
    GPU_PID[$g]=$!
    log "DISPATCH ${T} (w=${WV}) -> GPU ${g} pid ${GPU_PID[$g]}"
    qi=$((qi+1)); sleep 8
  done
  # reap finished cells, freeing their GPU
  for g in "${!GPU_PID[@]}"; do
    p="${GPU_PID[$g]}"
    if [ -n "${p}" ] && ! kill -0 "${p}" 2>/dev/null; then
      wait "${p}" 2>/dev/null || true
      log "REAP GPU ${g} (pid ${p} exited)"
      unset 'GPU_PID[$g]'
    fi
  done
  # done when queue exhausted and no cells in flight
  [ "${qi}" -ge "${#CELLS[@]}" ] && [ "${#GPU_PID[@]}" -eq 0 ] && break
  sleep 30
done

analyze
log "ALL DONE. §7 in docs/specs/results/2026-05-18-joint-tokenizer-results.md"
