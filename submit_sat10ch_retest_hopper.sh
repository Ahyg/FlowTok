#!/bin/bash
# Resubmit the 6 sat10ch AE tests that timed out at 4h walltime.
# Template now uses walltime=10:00:00; FSS over 11 channels needs the extra time.
# The launcher checks `[[ -f metrics.json ]]` and skips already-finished tests,
# so this is safe to re-run.

set -euo pipefail

cd "$(dirname "$0")"
TEMPLATE="./test_scratch_ae_one_ckpt_hopper.sh"

submit() {
  local exp="$1" step="$2" cfg="$3" tag="$4"
  local jobname="ae_${tag}_${step}"
  echo "+ qsub -N ${jobname}  EXP=${exp} STEP=${step} CFG=${cfg}"
  qsub -N "${jobname}" \
       -v "EXP=${exp},STEP=${step},CFG=${cfg}" \
       "${TEMPLATE}"
}

# sat10ch run1 missing: 168186, 200000
SAT10_R1_EXP="sat10ch_flowtitok_ae_bl77_vae_scratch_run1_gadi"
SAT10_R1_CFG="sat10ch_flowtitok_ae_bl77_vae_scratch_gadi.yaml"
for s in 168186 200000; do submit "$SAT10_R1_EXP" "$s" "$SAT10_R1_CFG" "s10r1"; done

# sat10ch run3 missing: 50000, 100000, 150000, 200000
SAT10_R3_EXP="sat10ch_flowtitok_ae_bl77_vae_scratch_run3_gadi"
SAT10_R3_CFG="sat10ch_flowtitok_ae_bl77_vae_scratch_run3_gadi.yaml"
for s in 50000 100000 150000 200000; do submit "$SAT10_R3_EXP" "$s" "$SAT10_R3_CFG" "s10r3"; done

echo "6 sat10ch retest jobs submitted (walltime=10h)."
