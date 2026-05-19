#!/bin/bash
#PBS -P kl02
#PBS -q normal
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=8
#PBS -l mem=32GB
#PBS -l jobfs=20GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N i2i_ds_pipe
set -uo pipefail
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
DR=/g/data/kl02/yh0308/Data/71
SD=$DR/filelists
mkdir -p /scratch/kl02/$USER/Projv2v/job_logs "$SD"
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_i2i_ds_pipe.log
exec > "$JOBLOG" 2>&1
cd $FT
echo "[$(date '+%F %T')] build train 2021-05..10 ct005"
python3 -u scripts/build_dataset.py --data-root $DR --save-dir $SD --dataset-pkl-name dataset_filelist_i2i_b_train_202105_202110_ct005.pkl --start-date 20210501 --end-date 20211031 --split-mode days --split-ratio 1,0,0 --coverage-threshold 0.05
echo "[$(date '+%F %T')] build val 2024-06 nofilter"
python3 -u scripts/build_dataset.py --data-root $DR --save-dir $SD --dataset-pkl-name dataset_filelist_i2i_b_val_202406_nofilter.pkl --start-date 20240601 --end-date 20240630 --split-mode days --split-ratio 0,1,0 --coverage-threshold 0.0
echo "[$(date '+%F %T')] build test 2024-07 nofilter"
python3 -u scripts/build_dataset.py --data-root $DR --save-dir $SD --dataset-pkl-name dataset_filelist_i2i_b_test_202407_nofilter.pkl --start-date 20240701 --end-date 20240731 --split-mode days --split-ratio 0,0,1 --coverage-threshold 0.0
echo "[$(date '+%F %T')] merge -> single 3-slot filelist"
python3 scripts/merge_i2i_filelists.py --train $SD/dataset_filelist_i2i_b_train_202105_202110_ct005.pkl --val $SD/dataset_filelist_i2i_b_val_202406_nofilter.pkl --test $SD/dataset_filelist_i2i_b_test_202407_nofilter.pkl --out $SD/dataset_filelist_i2i_baseline_2021summer_merged.pkl
M=$SD/dataset_filelist_i2i_baseline_2021summer_merged.pkl
for pair in "m2:textvae" "m3:diffusion" "m4:xpred"; do
  mm=${pair%%:*}; dd=${pair##*:}
  TD=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_i2i_b_${dd}_2021summer_tiny
  mkdir -p "$TD"
  python3 scripts/make_overfit_filelist.py --src "$M" --out "$TD/dataset_filelist.pkl" --n 32 --put-into all
done
if [ ! -s "$M" ]; then
  echo "[$(date '+%F %T')] FATAL: merged filelist $M missing/empty — NOT fanning out"
  exit 1
fi
for pair in "m2:textvae" "m3:diffusion" "m4:xpred"; do
  dd=${pair##*:}
  TF=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_i2i_b_${dd}_2021summer_tiny/dataset_filelist.pkl
  if [ ! -s "$TF" ]; then
    echo "[$(date '+%F %T')] FATAL: tiny filelist $TF missing/empty — NOT fanning out"
    exit 1
  fi
done
echo "[$(date '+%F %T')] dataset ready; fanning out jobs"
cd $FT
qsub train_i2i_b_m1_full_gadi.sh
qsub train_i2i_b_m2_tiny_gadi.sh
qsub train_i2i_b_m3_tiny_gadi.sh
qsub train_i2i_b_m4_tiny_gadi.sh
echo "[$(date '+%F %T')] PIPELINE DONE: M1 full + M2/M3/M4 tiny submitted"
