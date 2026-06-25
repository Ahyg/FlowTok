#!/bin/bash
#PBS -P ui54
#PBS -q gpuhopper
#PBS -l walltime=20:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N x1snap_fct600
# Full-set snap099 inference on the fact-v2v @600k ckpt (current best v2v-fact, avg_fss 0.4787).
# One extra sampler vs the standard Euler baseline (test_holdout_600000_nofilt = fixed_x0):
#   snap099 : --x1_snap_t 0.99   (at last step output model's clean-radar x1_hat, skip remaining integration)
# Replicates the production holdout invocation EXACTLY (max_batches_metrics=-1, gen metrics ON, same
# nofilt filelist/batch); differs ONLY in the snap flag. Compares snap099 against the fixed_x0 Euler baseline.
# (The earlier "canonical" sampler was a confirmed no-op == Euler and has been retired.)
set -uo pipefail
export HF_HOME="/scratch/kl02/$USER/hf_cache"; export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"; export XDG_CACHE_HOME="$HF_HOME"; export HF_HUB_OFFLINE=1; export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
EXP=/scratch/kl02/yh0308/Projv2v/Experiments
NF=/g/data/kl02/yh0308/Data/71/filelists

CFG=$FT/configs/Sat2Radar-v2v-cmp-fact-B-bl128-cond3nan1_gadi.py
WD=$EXP/sat2radar_flowtok_v2v_cmp_fact_B_bl128_cond3nan1
STEP=600000; MODE=v2v; BS=8
PKL=$NF/dataset_filelist_v2v_test_202407_202507_nofilter_nan1_clip16.pkl

CKPT=$WD/ckpts/${STEP}.ckpt
BASE_DIR=$WD/test_holdout_${STEP}_nofilt              # existing fixed_x0 Euler baseline (compare target, avg_fss 0.4787)
SNAP_DIR=$WD/test_holdout_${STEP}_nofilt_snap099      # snap099 output
mkdir -p "$SNAP_DIR" /scratch/kl02/$USER/Projv2v/job_logs
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_x1snap_factv2v600k.log
exec > "$JOBLOG" 2>&1
cd $FT
[ -e "$CKPT" ] || { echo "missing $CKPT"; exit 1; }

run_one () {  # OUTDIR  EXTRA_FLAG...
  local OUT="$1"; shift
  echo "[$(date '+%F %T')] fact-v2v @${STEP} -> $OUT  flags: $*"
  python3 -u scripts/test_sat2radar_v2v.py \
    --config "$CFG" --ckpt "$CKPT" --out_dir "$OUT" \
    --split test --mode $MODE --filelist_path "$PKL" \
    --max_batches_metrics -1 --max_batches_images 6 \
    --batch_size $BS "$@" \
    --dump_arrays --dump_pred_only --arrays_dir "$OUT/arrays" \
    --metrics_json "$OUT/metrics.json" --gpu 0
  echo "[$(date '+%F %T')] DONE $OUT"
}

run_one "$SNAP_DIR"  --x1_snap_t 0.99

compare () {  # LABEL SNAPJSON
  echo "[$(date '+%F %T')] === COMPARE $1 vs Euler baseline (fact-v2v @${STEP}) ==="
  python3 - "$BASE_DIR/metrics.json" "$2" <<'PY'
import json,sys
base=json.load(open(sys.argv[1])); snap=json.load(open(sys.argv[2]))
S=range(1,11)
def thr(d,t):
    try: return sum(d["fss_per_thr_scale"]["thr%d_scale%d"%(t,sc)] for sc in S)/10
    except: return float('nan')
def row(name,fb,fs,fmt="%.4f"):
    try: d=fs-fb
    except: d=float('nan')
    print(("  %-16s | "+fmt+" -> "+fmt+"  (delta %+.4f)")%(name,fb,fs,d))
print("  seen_samples base/new:", base.get("seen_samples"), snap.get("seen_samples"))
row("avg_fss(0-60)", base.get("avg_fss",float('nan')), snap.get("avg_fss",float('nan')))
row("weighted_fss",  base.get("weighted_fss",float('nan')), snap.get("weighted_fss",float('nan')))
for t in [5,15,25,35,45]:
    row("aFSS thr%d"%t, thr(base,t), thr(snap,t), "%.3f")
row("CSI35", base.get("csi35",float('nan')), snap.get("csi35",float('nan')))
row("HSS35", base.get("hss35",float('nan')), snap.get("hss35",float('nan')))
row("SSIM", base.get("ssim",float('nan')), snap.get("ssim",float('nan')))
row("PSNR_db", base.get("psnr_db",float('nan')), snap.get("psnr_db",float('nan')),"%.3f")
row("RMSE_dbz", base.get("rmse_dbz",float('nan')), snap.get("rmse_dbz",float('nan')),"%.3f")
row("bias_dbz", base.get("bias_dbz",float('nan')), snap.get("bias_dbz",float('nan')),"%+.3f")
print("  --- gen_metrics (lower=better) ---")
gb=base.get("gen_metrics",{}); gs=snap.get("gen_metrics",{})
for k in sorted(set(gb)|set(gs)):
    vb,vs=gb.get(k),gs.get(k)
    if isinstance(vb,dict): vb=vb.get("mean")
    if isinstance(vs,dict): vs=vs.get("mean")
    try: print("  %-16s | %.5f -> %.5f  (delta %+.5f)"%(k,vb,vs,vs-vb))
    except: print("  %-16s | %s -> %s"%(k,vb,vs))
PY
}
compare "snap099"   "$SNAP_DIR/metrics.json"
echo "[$(date '+%F %T')] DONE fact-v2v @${STEP} snap099"
