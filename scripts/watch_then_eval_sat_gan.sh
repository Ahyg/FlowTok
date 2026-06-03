#!/bin/bash
# scripts/watch_then_eval_sat_gan.sh
# Detached watcher (sat analog of watch_then_eval_gan.sh): wait until ALL five
# sat10ch GAN cells have a checkpoint-final, then auto-run eval_sat_gan_sweep.sh
# over both slots on a free GPU. Survives session detach. If the orchestrator
# dies leaving cells unfinished past the cap, write an ALERT and exit without a
# partial-success claim.
#
# Launch: setsid bash scripts/watch_then_eval_sat_gan.sh >.../watch_eval.log 2>&1 &

set -uo pipefail

EXP_ROOT="/mnt/ssd_2/yghu/Experiments/ae_recipe_sweep/gan_sat"
REPO="/mnt/ssd_1/yghu/Code/FlowTok"
CELLS=(s10_gan_nogan s10_gan_w0001 s10_gan_w001 s10_gan_w01 s10_gan_w1)
ALERT="${EXP_ROOT}/WATCH_ALERT.txt"
EVAL_GPU="${EVAL_GPU:-1}"
CAP_HOURS="${CAP_HOURS:-30}"          # give up waiting after this many hours

log() { echo "[$(date '+%F %T')] $*"; }

deadline=$(( $(date +%s) + CAP_HOURS*3600 ))
log "watcher start — waiting for all ${#CELLS[@]} sat GAN cells' checkpoint-final"
while true; do
  done=0
  for c in "${CELLS[@]}"; do
    [[ -f "${EXP_ROOT}/${c}/checkpoint-final/metadata.json" ]] && done=$((done+1))
  done
  if (( done >= ${#CELLS[@]} )); then
    log "all ${done}/${#CELLS[@]} cells finished."
    break
  fi
  if (( $(date +%s) > deadline )); then
    { echo "[$(date '+%F %T')] ALERT: only ${done}/${#CELLS[@]} sat GAN cells finished after ${CAP_HOURS}h."
      echo "Eval run anyway on whatever finished; inspect ${EXP_ROOT}/orchestrator.log for stalls."; } | tee "$ALERT"
    break
  fi
  sleep 180
done

log "Launching eval_sat_gan_sweep.sh (best_val then final) on GPU ${EVAL_GPU}."
CUDA_VISIBLE_DEVICES="$EVAL_GPU" bash "${REPO}/scripts/eval_sat_gan_sweep.sh" best_val
CUDA_VISIBLE_DEVICES="$EVAL_GPU" bash "${REPO}/scripts/eval_sat_gan_sweep.sh" final
rc=$?
log "eval_sat_gan_sweep (both slots) finished rc=${rc}."
exit "$rc"
