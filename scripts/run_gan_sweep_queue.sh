#!/bin/bash
# Radar AE late-GAN sweep — queue orchestrator (lab2).
# Deferred Stage 3 of docs/specs/2026-05-20-ae-loss-recipe-sweep-design.md, run
# on user request. Base = the radar GAN-free WINNER (patch8 + kl÷10 + perc0.6 @
# 128 tokens, 2026-05-22 results). Sweeps discriminator_weight {0.001,0.01,0.1,1}
# + a paired GAN-OFF anchor, 100k steps each. Non-swept GAN params follow
# TA-TiTok (tatitok_bl64_vae): disc_start 0.31×horizon=30k, disc_lr 1e-4,
# lecam 1e-3, disc_factor 1.0.
#
# CRITICAL (lab2-specific): root LV (/, hence /tmp) is 100% FULL. All scratch
# (TMPDIR, matplotlib, triton kernel cache) is forced onto ssd_2, and every
# config writes output_dir under ssd_2. Do NOT let anything write to /tmp.
#
# Designed to run detached inside tmux session "gan_sweep". Survives detach.

set -u  # NOT -e: one failing cell must not kill the queue.

REPO="/mnt/ssd_1/yghu/Code/FlowTok"
EXP_ROOT="/mnt/ssd_2/yghu/Experiments/ae_recipe_sweep/gan"
ORCH_LOG="${EXP_ROOT}/orchestrator.log"
STATUS="${EXP_ROOT}/STATUS.txt"
MEM_FREE_REQUIRED=12000       # MiB — radar 1ch patch8 batch32 fits ~10 GiB
GPUS=(0 1 2 3)

# Keep ALL scratch off the full root FS.
export TMPDIR=/mnt/ssd_2/yghu/tmp
export MPLCONFIGDIR=/mnt/ssd_2/yghu/tmp/mpl
export TRITON_CACHE_DIR=/mnt/ssd_2/yghu/tmp/triton
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$EXP_ROOT"

# Anchor + TA-TiTok default (0.1) first so the headline comparison lands first,
# then the brackets (0.01, 1.0, 0.001).
QUEUE=(
  radar_gan_nogan
  radar_gan_w01
  radar_gan_w001
  radar_gan_w1
  radar_gan_w0001
)

log() { echo "[$(date '+%F %T')] $*" | tee -a "${ORCH_LOG}"; }

declare -A RUN_GPU RUN_PID GPU_BUSY
idx=0
total=${#QUEUE[@]}

log "GAN-sweep orchestrator start. ${total} cells. env=flowtok. tmux=gan_sweep. TMPDIR=${TMPDIR}."
log "Queue: ${QUEUE[*]}"

write_status() {
  {
    echo "Radar GAN sweep — $(date '+%F %T')"
    echo "root free: $(df -h / | awk 'NR==2{print $4}')   ssd_2 free: $(df -h /mnt/ssd_2 | awk 'NR==2{print $4}')"
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

  # ---- skip already-completed / externally-running cells (resume support) ----
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
        cfg="${REPO}/configs/${cell}.yaml"
        out="${EXP_ROOT}/${cell}"
        mkdir -p "$out"
        log "LAUNCH ${cell} on GPU ${g} (free ${free} MiB; $((idx+1))/${total})"
        CUDA_VISIBLE_DEVICES="$g" WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
        TMPDIR="$TMPDIR" MPLCONFIGDIR="$MPLCONFIGDIR" TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
        setsid bash -c \
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
