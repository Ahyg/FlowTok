#!/bin/bash
#PBS -P kl02
#PBS -q normal
#PBS -l walltime=10:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m a
#PBS -N i2i_b_test_watcher
set -uo pipefail
FT=/scratch/kl02/$USER/Projv2v/FlowTok
EXP=/scratch/kl02/$USER/Projv2v/Experiments
mkdir -p /scratch/kl02/$USER/Projv2v/job_logs
LOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_i2i_b_test_watcher.log
cd $FT
{
echo "[$(date '+%F %T')] watcher start"

declare -A SUBMITTED
declare -A CKPT
CKPT[m2]=$EXP/sat2radar_flowtok_i2i_b_textvae_2021summer/ckpts/60000.ckpt
CKPT[m3]=$EXP/sat2radar_flowtok_i2i_b_diffusion_2021summer/ckpts/60000.ckpt
CKPT[m4]=$EXP/sat2radar_flowtok_i2i_b_xpred_2021summer/ckpts/60000.ckpt

while true; do
  ALL_DONE=1
  for M in m2 m3 m4; do
    [ "${SUBMITTED[$M]:-}" = "1" ] && continue
    ALL_DONE=0
    if [ -e "${CKPT[$M]}" ]; then
      echo "[$(date '+%F %T')] $M 60k ckpt detected at ${CKPT[$M]}"
      cd $FT && JID=$(qsub holdout_test_i2i_b_${M}_gadi.sh 2>&1)
      echo "[$(date '+%F %T')] $M holdout test submitted -> $JID"
      SUBMITTED[$M]=1
    fi
  done
  [ "$ALL_DONE" = "1" ] && { echo "[$(date '+%F %T')] all tests submitted, exit"; break; }
  sleep 300
done
echo "[$(date '+%F %T')] watcher done"
} > "$LOG" 2>&1
