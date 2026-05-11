#!/bin/bash
# Submit the 12 remaining AE tests as independent gpuvolta jobs.
# Each job tests one ckpt; the generic launcher handles cfg/ckpt/skip logic.

set -euo pipefail

cd "$(dirname "$0")"
TEMPLATE="./test_scratch_ae_one_ckpt_volta.sh"

submit() {
  local exp="$1" step="$2" cfg="$3" tag="$4"
  local jobname="ae_${tag}_${step}"
  echo "+ qsub -N ${jobname}  EXP=${exp} STEP=${step} CFG=${cfg}"
  qsub -N "${jobname}" \
       -v "EXP=${exp},STEP=${step},CFG=${cfg}" \
       "${TEMPLATE}"
}

# ── sat10ch run1 missing: 168186, 200000 ─────────────────────────
SAT10_R1_EXP="sat10ch_flowtitok_ae_bl77_vae_scratch_run1_gadi"
SAT10_R1_CFG="sat10ch_flowtitok_ae_bl77_vae_scratch_gadi.yaml"
for s in 168186 200000; do submit "$SAT10_R1_EXP" "$s" "$SAT10_R1_CFG" "s10r1"; done

# ── radar run3: 50000, 100000, 150000, 198063, 200000 ────────────
RADAR_R3_EXP="radar_flowtitok_ae_bl77_vae_scratch_run3_gadi"
RADAR_R3_CFG="radar_flowtitok_ae_bl77_vae_scratch_run3_gadi.yaml"
for s in 50000 100000 150000 198063 200000; do submit "$RADAR_R3_EXP" "$s" "$RADAR_R3_CFG" "rdr3"; done

# ── sat run3: 50000, 100000, 150000, 198063, 200000 ──────────────
SAT_R3_EXP="sat_flowtitok_ae_bl77_vae_scratch_run3_gadi"
SAT_R3_CFG="sat_flowtitok_ae_bl77_vae_scratch_run3_gadi.yaml"
for s in 50000 100000 150000 198063 200000; do submit "$SAT_R3_EXP" "$s" "$SAT_R3_CFG" "satr3"; done

# ── sat10ch run3: 50000, 100000, 150000, 200000 (no 198063 ckpt) ──
SAT10_R3_EXP="sat10ch_flowtitok_ae_bl77_vae_scratch_run3_gadi"
SAT10_R3_CFG="sat10ch_flowtitok_ae_bl77_vae_scratch_run3_gadi.yaml"
for s in 50000 100000 150000 200000; do submit "$SAT10_R3_EXP" "$s" "$SAT10_R3_CFG" "s10r3"; done

echo "All 16 jobs submitted."
