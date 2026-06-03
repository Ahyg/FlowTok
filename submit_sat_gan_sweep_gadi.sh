#!/bin/bash
# submit_sat_gan_sweep_gadi.sh — qsub all 5 sat10ch GAN-sweep cells on Gadi.
# Each is an independent single-GPU job (no PBS dependencies). Run from the
# FlowTok repo root on a Gadi login node AFTER `git pull`:
#
#   bash submit_sat_gan_sweep_gadi.sh
#
# nogan (GAN-off baseline) + ascending disc_weight. Each job picks its config
# from configs/<cell>_gadi.yaml via `-v CELL=<cell>`.

set -euo pipefail

CELLS=(
  s10_gan_nogan
  s10_gan_w0001
  s10_gan_w001
  s10_gan_w01
  s10_gan_w1
)

for cell in "${CELLS[@]}"; do
  jid=$(qsub -N "${cell}" -v "CELL=${cell}" train_sat_gan_sweep_gadi.sh)
  echo "submitted ${cell}  ->  ${jid}"
done
