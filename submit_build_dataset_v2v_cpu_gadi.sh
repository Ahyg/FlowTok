#!/usr/bin/env bash
set -euo pipefail

cd "/scratch/kl02/$USER/Projv2v/FlowTok"

jid_train=$(qsub -v SPLIT=train build_dataset_v2v_cpu_gadi.sh)
jid_val=$(qsub -v SPLIT=val build_dataset_v2v_cpu_gadi.sh)
jid_test=$(qsub -v SPLIT=test build_dataset_v2v_cpu_gadi.sh)

echo "[SUBMIT] train job: ${jid_train}"
echo "[SUBMIT] val   job: ${jid_val}"
echo "[SUBMIT] test  job: ${jid_test}"

dep="afterok:${jid_train}:${jid_val}:${jid_test}"
jid_merge=$(qsub -W depend=${dep} merge_dataset_v2v_cpu_gadi.sh)

echo "[SUBMIT] merge job (after all success): ${jid_merge}"

