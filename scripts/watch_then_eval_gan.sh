#!/bin/bash
# scripts/watch_then_eval_gan.sh
# Detached watcher: wait for radar_gan_w0001 to finish (its checkpoint-final/
# appears), then auto-run eval_gan_sweep.sh over all 5 cells on the freed GPU.
# Survives session detach. If w0001's process dies WITHOUT producing
# checkpoint-final, write an ALERT and exit WITHOUT evaluating — a crash must
# not masquerade as a completed sweep.
#
# Launch: setsid bash scripts/watch_then_eval_gan.sh >.../watch_eval.log 2>&1 &

set -uo pipefail

EXP_ROOT="/mnt/ssd_2/yghu/Experiments/ae_recipe_sweep/gan"
REPO="/mnt/ssd_1/yghu/Code/FlowTok"
CELL=radar_gan_w0001
FINAL="${EXP_ROOT}/${CELL}/checkpoint-final/metadata.json"
PIDFILE="${EXP_ROOT}/${CELL}/training.pid"
ALERT="${EXP_ROOT}/WATCH_ALERT.txt"
EVAL_GPU="${EVAL_GPU:-1}"      # w0001 runs on GPU1; it frees on completion
GRACE_AFTER_DEATH=300         # s to wait for checkpoint-final after pid dies

log() { echo "[$(date '+%F %T')] $*"; }

log "watcher start — waiting for ${CELL} checkpoint-final at ${FINAL}"
death_at=0
while true; do
  if [[ -f "$FINAL" ]]; then
    log "${CELL} finished (checkpoint-final present)."
    break
  fi
  pid=$(cat "$PIDFILE" 2>/dev/null || echo "")
  if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
    # process gone; allow a grace window for the final ckpt to flush
    now=$(date +%s)
    if (( death_at == 0 )); then death_at=$now; log "${CELL} pid ${pid} gone; grace ${GRACE_AFTER_DEATH}s for checkpoint-final."; fi
    if (( now - death_at > GRACE_AFTER_DEATH )); then
      { echo "[$(date '+%F %T')] ALERT: ${CELL} pid ${pid} died with NO checkpoint-final after ${GRACE_AFTER_DEATH}s grace."
        echo "Eval NOT started. Inspect ${EXP_ROOT}/${CELL}/training.log (likely re-crashed)."; } | tee "$ALERT"
      exit 1
    fi
  else
    death_at=0   # pid alive again (or pidfile updated) — reset
  fi
  sleep 120
done

log "Launching eval_gan_sweep.sh (best_val) on GPU ${EVAL_GPU}."
CUDA_VISIBLE_DEVICES="$EVAL_GPU" bash "${REPO}/scripts/eval_gan_sweep.sh" best_val
rc=$?
log "eval_gan_sweep finished rc=${rc}."
exit "$rc"
