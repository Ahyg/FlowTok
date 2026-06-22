#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=06:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=128GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N eval_pmm_nfe
# Eval-only (NO training): NFE sweep {20,50,100} + dump dBZ arrays for m8-i2i & fact-v2v
# on a fixed nofilt test subset, then offline PMM/quantile-mapping rescore (full metric vector).
set -uo pipefail
export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"; export TORCH_HOME="$HF_HOME"; export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1; export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
EVAL=/scratch/kl02/yh0308/Projv2v/Experiments/_eval_pmm_nfe_20260616
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_eval_pmm_nfe.log
mkdir -p "$EVAL" "$(dirname $JOBLOG)"
cd $FT
exec > "$JOBLOG" 2>&1

I2I_CFG=$FT/configs/Sat2Radar-i2i-cmp-m8align-B-bl128-cond3nan1_gadi.py
I2I_CKPT=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_i2i_cmp_m8align_B_bl128_cond3nan1/ckpts/200000.ckpt
I2I_PKL=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_test_202407_202507_nofilter_nan1_clip16.pkl

V2V_CFG=$FT/configs/Sat2Radar-v2v-cmp-fact-B-bl128-cond3nan1_gadi.py
V2V_CKPT=/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_cmp_fact_B_bl128_cond3nan1/ckpts/300000.ckpt
V2V_PKL=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_test_202407_202507_nofilter_nan1_clip16.pkl

run () {  # $1=tag $2=cfg $3=ckpt $4=pkl $5=mode $6=bs $7=maxb $8=nfe
  local OUT=$EVAL/$1
  mkdir -p "$OUT"
  echo "[$(date '+%F %T')] === $1  (mode=$5 nfe=$8 bs=$6 maxb=$7) ==="
  python3 -u scripts/test_sat2radar_v2v.py \
    --config "$2" --ckpt "$3" --out_dir "$OUT" \
    --split test --mode "$5" --filelist_path "$4" \
    --batch_size "$6" --max_batches_metrics "$7" --max_batches_images 0 \
    --flow_sample_steps "$8" --skip_gen_metrics \
    --dump_arrays --arrays_dir "$OUT/arrays" \
    --metrics_json "$OUT/metrics.json" --gpu 0
  echo "[$(date '+%F %T')] done $1 rc=$?"
}

# i2i m8align @200k : bs16, 48 batches (~768 frames)
for n in 20 50 100; do run "m8_i2i_nfe${n}" "$I2I_CFG" "$I2I_CKPT" "$I2I_PKL" i2i 16 48 "$n"; done
# v2v fact @300k : bs8, 16 batches (~128 clips x16 = 2048 frames)
for n in 20 50 100; do run "fact_v2v_nfe${n}" "$V2V_CFG" "$V2V_CKPT" "$V2V_PKL" v2v 8 16 "$n"; done

# clean training climatology (best-effort; rescorer falls back to oracle if missing)
echo "[$(date '+%F %T')] === build radar climatology ==="
python3 -u "$EVAL/build_radar_clim.py" \
  --train_pkl /g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_train_201906_202406_cond3nan1_clip16_p005_seed42.pkl \
  --out "$EVAL/radar_clim_train.npy" || echo "clim build failed -> oracle-only"

echo "[$(date '+%F %T')] === PMM / quantile-mapping rescore ==="
CLIM_ARG=""; [ -f "$EVAL/radar_clim_train.npy" ] && CLIM_ARG="--train_clim $EVAL/radar_clim_train.npy"
python3 -u "$EVAL/pmm_rescore.py" --eval_root "$EVAL" $CLIM_ARG
echo "[$(date '+%F %T')] ALL DONE"
