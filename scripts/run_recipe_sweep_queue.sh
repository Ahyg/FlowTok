#!/bin/bash
# AE loss-recipe sweep — Stage 0 + Stage 1 queue orchestrator (lab2).
# Spec: docs/specs/2026-05-20-ae-loss-recipe-sweep-design.md
#
# Runs 12 GAN-free 40k screening cells. Prioritized queue: baselines +
# run3-attribution cells (kl10, pc) first so the headline science lands first.
# Opportunistic multi-GPU: dispatches the next cell to ANY GPU with at least
# MEM_FREE_REQUIRED MiB free, co-tenanting alongside other users' jobs when
# capacity permits. Sized for the heaviest sat10ch cell (~12 GiB) plus 3 GiB
# safety margin against neighbour growth and PyTorch fragmentation.
#
# Designed to run detached inside tmux session "ae_sweep". Survives detach.

set -u  # NOT -e: one failing cell must not kill the queue.

REPO="/mnt/ssd_1/yghu/Code/FlowTok"
EXP_ROOT="/mnt/ssd_2/yghu/Experiments/ae_recipe_sweep"
ORCH_LOG="${EXP_ROOT}/orchestrator.log"
STATUS="${EXP_ROOT}/STATUS.txt"
MEM_FREE_REQUIRED=15000       # MiB — GPU needs at least this much FREE memory
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
  s10_b0_b8
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
      gpu_idx="${RUN_GPU[$c]}"
      unset "GPU_BUSY[$gpu_idx]"
      unset "RUN_GPU[$c]"
      unset "RUN_PID[$c]"
    fi
  done

  # ---- skip already-completed cells (resume support) ----
  # Also skip cells with a live training.pid we did not launch (e.g. a manual
  # run started outside the orchestrator), so we never duplicate a cell on a
  # second GPU that frees mid-run.
  while (( idx < total )); do
    cell="${QUEUE[$idx]}"
    pid_file="${EXP_ROOT}/${cell}/training.pid"
    if [[ -f "${EXP_ROOT}/${cell}/checkpoint-final/metadata.json" ]]; then
      log "SKIP ${cell} — checkpoint-final exists (resume; $((idx+1))/${total})"
      idx=$((idx+1))
    elif [[ -f "$pid_file" ]] \
         && kill -0 "$(cat "$pid_file" 2>/dev/null)" 2>/dev/null \
         && [[ -z "${RUN_PID[$cell]:-}" ]]; then
      log "SKIP ${cell} — externally running (pid=$(cat "$pid_file"); $((idx+1))/${total})"
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
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ' || echo 0)
      [[ "$free" =~ ^[0-9]+$ ]] || free=0
      if (( free >= MEM_FREE_REQUIRED )); then
        cell="${QUEUE[$idx]}"
        cfg="${REPO}/configs/rs_${cell}.yaml"
        out="${EXP_ROOT}/${cell}"
        mkdir -p "$out"
        log "LAUNCH ${cell} on GPU ${g} (free ${free} MiB; $((idx+1))/${total})"
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
