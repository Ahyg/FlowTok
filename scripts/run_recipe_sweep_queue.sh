#!/bin/bash
# AE loss-recipe sweep — Stage 0 + Stage 1 queue orchestrator (lab2).
# Spec: docs/specs/2026-05-20-ae-loss-recipe-sweep-design.md
#
# Runs 12 GAN-free 40k screening cells. Prioritized queue: baselines +
# run3-attribution cells (kl10, pc) first so the headline science lands first.
# Opportunistic multi-GPU: dispatches the next cell to ANY truly-empty GPU
# (mem.used < 1500 MiB) so other users' reserved GPUs (1,3) and the actively
# training GPU (2) are never touched; grabs them only if they actually free.
#
# Designed to run detached inside tmux session "ae_sweep". Survives detach.

set -u  # NOT -e: one failing cell must not kill the queue.

REPO="/mnt/ssd_1/yghu/Code/FlowTok"
EXP_ROOT="/mnt/ssd_2/yghu/Experiments/ae_recipe_sweep"
ORCH_LOG="${EXP_ROOT}/orchestrator.log"
STATUS="${EXP_ROOT}/STATUS.txt"
MEM_FREE_THRESH=1500          # MiB — below this a GPU counts as truly empty
GPUS=(0 1 2 3)

# Priority order: baselines first, then run3-attribution (kl10, pc),
# then perceptual ceiling, then perceptual floor, then kl-low.
QUEUE=(
  radar_b0  s10_b0
  radar_s1_kl10  s10_s1_kl10
  radar_s1_pc    s10_s1_pc
  radar_s1_p16   s10_s1_p16
  radar_s1_p06   s10_s1_p06
  radar_s1_kl05  s10_s1_kl05
)

mkdir -p "${EXP_ROOT}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${ORCH_LOG}"; }

declare -A RUN_GPU RUN_PID GPU_BUSY
idx=0
total=${#QUEUE[@]}

log "Orchestrator start. ${total} cells queued. conda env=flowtok. tmux=ae_sweep."
log "Queue order: ${QUEUE[*]}"

write_status() {
  {
    echo "AE recipe sweep — $(date '+%F %T')"
    echo "queued: $idx / $total dispatched"
    echo "RUNNING:"
    for c in "${!RUN_PID[@]}"; do
      st=$(grep -oE 'Step: [0-9]+' "${EXP_ROOT}/$c/training.log" 2>/dev/null | tail -1 || echo 'Step: ?')
      echo "  $c  gpu=${RUN_GPU[$c]}  pid=${RUN_PID[$c]}  $st"
    done
    echo "PENDING: ${QUEUE[*]:$idx}"
  } > "${STATUS}"
}

while (( idx < total )) || (( ${#RUN_PID[@]} > 0 )); do
  # ---- reap finished jobs ----
  for c in "${!RUN_PID[@]}"; do
    if ! kill -0 "${RUN_PID[$c]}" 2>/dev/null; then
      fin=$(grep -c 'Finishing training' "${EXP_ROOT}/$c/training.log" 2>/dev/null || echo 0)
      log "FINISHED ${c} (gpu ${RUN_GPU[$c]}, pid ${RUN_PID[$c]}, finish-marker=${fin})"
      unset 'GPU_BUSY[${RUN_GPU[$c]}]'
      unset 'RUN_GPU[$c]' 'RUN_PID[$c]'
    fi
  done

  # ---- skip already-completed cells (resume support) ----
  while (( idx < total )); do
    cell="${QUEUE[$idx]}"
    if [[ -f "${EXP_ROOT}/${cell}/checkpoint-final/metadata.json" ]]; then
      log "SKIP ${cell} — checkpoint-final exists (resume; $((idx+1))/${total})"
      idx=$((idx+1))
    else
      break
    fi
  done

  # ---- dispatch to any free GPU ----
  if (( idx < total )); then
    for g in "${GPUS[@]}"; do
      (( idx < total )) || break
      [[ -n "${GPU_BUSY[$g]:-}" ]] && continue
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ' || echo 999999)
      [[ "$used" =~ ^[0-9]+$ ]] || used=999999
      if (( used < MEM_FREE_THRESH )); then
        cell="${QUEUE[$idx]}"
        cfg="${REPO}/configs/rs_${cell}.yaml"
        out="${EXP_ROOT}/${cell}"
        mkdir -p "$out"
        log "LAUNCH ${cell} on GPU ${g} (mem ${used} MiB; $((idx+1))/${total})"
        CUDA_VISIBLE_DEVICES="$g" WANDB_MODE=disabled PYTHONUNBUFFERED=1 setsid bash -c \
          "cd '${REPO}' && conda run --no-capture-output -n flowtok accelerate launch --num_processes 1 scripts/train_flowtitok_ae.py --config '${cfg}'" \
          > "${out}/training.log" 2>&1 &
        pid=$!
        echo "$pid" > "${out}/training.pid"
        RUN_GPU[$cell]=$g; RUN_PID[$cell]=$pid; GPU_BUSY[$g]=$cell
        idx=$((idx+1))
        sleep 40   # let VRAM register before considering the next GPU
      fi
    done
  fi

  write_status
  sleep 60
done

log "ALL DONE. ${total}/${total} cells completed."
write_status
