#!/bin/bash
# scripts/launch_sat_ae_token_sweep_lab2.sh
# Orchestrate 3-cell sat AE token sweep on lab2 with 2 available GPUs.
# Runs tok77 on GPU 0 + tok128 on GPU 2 in parallel; defers tok256 until one frees.

set -euo pipefail
FLOWTOK_ROOT="/mnt/ssd_1/yghu/Code/FlowTok"
EXP_ROOT="/mnt/ssd_2/yghu/Experiments"
mkdir -p "${EXP_ROOT}"

source "/home/yghu/miniconda3/etc/profile.d/conda.sh"
conda activate flowtok
cd "${FLOWTOK_ROOT}"

launch_cell () {
  local CELL_TAG="$1"; local GPU_IDX="$2"; local CONFIG="$3"
  local OUT_DIR="${EXP_ROOT}/sat10ch_ae_${CELL_TAG}_run1"
  mkdir -p "${OUT_DIR}"
  echo "[$(date '+%F %T')] Launch ${CELL_TAG} on GPU ${GPU_IDX} -> ${OUT_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU_IDX}" nohup \
    accelerate launch --num_processes 1 \
      scripts/train_flowtitok_ae.py --config "${CONFIG}" \
      > "${OUT_DIR}/training.log" 2>&1 &
  echo $! > "${OUT_DIR}/training.pid"
  echo "[$(date '+%F %T')] ${CELL_TAG} PID=$(cat ${OUT_DIR}/training.pid)"
}

wait_for_one () {
  local PID1="$1"; local PID2="$2"
  while kill -0 "${PID1}" 2>/dev/null && kill -0 "${PID2}" 2>/dev/null; do
    sleep 60
  done
  # Return whichever PID is alive (the survivor)
  if kill -0 "${PID1}" 2>/dev/null; then echo "${PID1}"; else echo "${PID2}"; fi
}

# Launch tok77 on GPU 0
launch_cell "tok77" 0 "${FLOWTOK_ROOT}/configs/sat10ch_ae_tok77_lab2.yaml"
PID_77=$(cat "${EXP_ROOT}/sat10ch_ae_tok77_run1/training.pid")
GPU_77=0

# Launch tok128 on GPU 2
launch_cell "tok128" 2 "${FLOWTOK_ROOT}/configs/sat10ch_ae_tok128_lab2.yaml"
PID_128=$(cat "${EXP_ROOT}/sat10ch_ae_tok128_run1/training.pid")
GPU_128=2

# Wait for one to finish
SURVIVOR=$(wait_for_one "${PID_77}" "${PID_128}")
if [ "${SURVIVOR}" = "${PID_77}" ]; then
  FREE_GPU=${GPU_128}
  echo "[$(date '+%F %T')] tok128 finished first; freeing GPU ${FREE_GPU}"
else
  FREE_GPU=${GPU_77}
  echo "[$(date '+%F %T')] tok77 finished first; freeing GPU ${FREE_GPU}"
fi

# Launch tok256 on the freed GPU
launch_cell "tok256" "${FREE_GPU}" "${FLOWTOK_ROOT}/configs/sat10ch_ae_tok256_lab2.yaml"
PID_256=$(cat "${EXP_ROOT}/sat10ch_ae_tok256_run1/training.pid")

# Wait for both remaining jobs
echo "[$(date '+%F %T')] Waiting for both remaining jobs to finish (${SURVIVOR}, ${PID_256})"
wait "${SURVIVOR}" 2>/dev/null || true
wait "${PID_256}"  2>/dev/null || true

echo "[$(date '+%F %T')] All three cells finished."
for TAG in tok77 tok128 tok256; do
  ls -la "${EXP_ROOT}/sat10ch_ae_${TAG}_run1/" | grep checkpoint || echo "  (no ckpts for ${TAG})"
done
