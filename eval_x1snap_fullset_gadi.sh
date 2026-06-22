#!/bin/bash
#PBS -P kl02
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
#PBS -N x1snapfull
# Full-set snap inference (x1_snap_t=0.99 = output clean-radar x1_hat at the LAST step), replicating
# the production holdout invocation EXACTLY (max_batches_metrics=-1, gen metrics ON, same filelist/batch)
# and differing ONLY in --x1_snap_t. Compares against the existing test_holdout_<STEP>_nofilt baseline.
# Submit twice: qsub -v MODEL=i2i ... ; qsub -v MODEL=v2v ...
set -uo pipefail
export HF_HOME="/scratch/kl02/$USER/hf_cache"; export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"; export XDG_CACHE_HOME="$HF_HOME"; export HF_HUB_OFFLINE=1; export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
EXP=/scratch/kl02/yh0308/Projv2v/Experiments
MODEL="${MODEL:?set MODEL=i2i or MODEL=v2v}"
NF=/g/data/kl02/yh0308/Data/71/filelists

if [ "$MODEL" = "i2i" ]; then
  CFG=$FT/configs/Sat2Radar-i2i-cmp-m8align-B-bl128-cond3nan1_gadi.py
  WD=$EXP/sat2radar_flowtok_i2i_cmp_m8align_B_bl128_cond3nan1
  STEP=200000; MODE=i2i; BS=16
  PKL=$NF/dataset_filelist_i2i_test_202407_202507_nofilter_nan1_clip16.pkl
elif [ "$MODEL" = "v2v" ]; then
  CFG=$FT/configs/Sat2Radar-v2v-cmp-fact-B-bl128-cond3nan1_gadi.py
  WD=$EXP/sat2radar_flowtok_v2v_cmp_fact_B_bl128_cond3nan1
  STEP=300000; MODE=v2v; BS=8
  PKL=$NF/dataset_filelist_v2v_test_202407_202507_nofilter_nan1_clip16.pkl
else echo "bad MODEL=$MODEL"; exit 2; fi

CKPT=$WD/ckpts/${STEP}.ckpt
BASE_DIR=$WD/test_holdout_${STEP}_nofilt              # existing production baseline (compare target)
SNAP_DIR=$WD/test_holdout_${STEP}_nofilt_snap099      # new snap output
mkdir -p "$SNAP_DIR" /scratch/kl02/$USER/Projv2v/job_logs
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_x1snapfull_${MODEL}.log
exec > "$JOBLOG" 2>&1
cd $FT
[ -e "$CKPT" ] || { echo "missing $CKPT"; exit 1; }

echo "[$(date '+%F %T')] $MODEL snap099 full-set -> $SNAP_DIR  (baseline=$BASE_DIR)"
# EXACT production recipe + --x1_snap_t 0.99 ; gen metrics ON (no --skip_gen_metrics)
python3 -u scripts/test_sat2radar_v2v.py \
  --config "$CFG" --ckpt "$CKPT" --out_dir "$SNAP_DIR" \
  --split test --mode $MODE --filelist_path "$PKL" \
  --max_batches_metrics -1 --max_batches_images 6 \
  --batch_size $BS --x1_snap_t 0.99 \
  --metrics_json "$SNAP_DIR/metrics.json" --gpu 0
echo "[$(date '+%F %T')] inference DONE"

echo "[$(date '+%F %T')] === COMPARE snap099 vs baseline ($MODEL) ==="
python3 - "$BASE_DIR/metrics.json" "$SNAP_DIR/metrics.json" <<'PY'
import json,sys,math
base=json.load(open(sys.argv[1])); snap=json.load(open(sys.argv[2]))
S=range(1,11); THR=list(range(0,61,5))
def thr(d,t):
    try: return sum(d["fss_per_thr_scale"]["thr%d_scale%d"%(t,sc)] for sc in S)/10
    except: return float('nan')
def mass5(d):
    h=d.get("refl_hist_60bins")
    if not h: return float('nan')
    e=h["bin_edges"]; pd=h["pred_density"]; gd=h["gt_density"]
    def fr(dn):
        tot=hi=0.0
        for i,v in enumerate(dn):
            w=e[i+1]-e[i]; m=v*w; tot+=m
            if e[i]>=5: hi+=m
        return hi/tot if tot else float('nan')
    return fr(pd)/fr(gd)
def sob(d):
    g=d.get("grad_sobel",{}).get("k1",{}); pr=g.get("pred_per_sample"); gt=g.get("gt_per_sample")
    return (sum(pr)/len(pr))/(sum(gt)/len(gt)) if pr and gt else float('nan')
def row(name,fb,fs,fmt="%.4f"):
    try: d=fs-fb
    except: d=float('nan')
    print(("  %-16s | "+fmt+" -> "+fmt+"  (Δ%+.4f)")%(name,fb,fs,d))
print("  seen_samples base/snap:", base.get("seen_samples"), snap.get("seen_samples"))
row("avg_fss(0-60)", base.get("avg_fss",float('nan')), snap.get("avg_fss",float('nan')))
row("weighted_fss",  base.get("weighted_fss",float('nan')), snap.get("weighted_fss",float('nan')))
for t in [5,15,20,25,30,35,40,45]:
    row("aFSS thr%d"%t, thr(base,t), thr(snap,t), "%.3f")
row("CSI35", base.get("csi35",float('nan')), snap.get("csi35",float('nan')),"%.4f")
row("HSS35", base.get("hss35",float('nan')), snap.get("hss35",float('nan')),"%.4f")
row("SSIM", base.get("ssim",float('nan')), snap.get("ssim",float('nan')),"%.4f")
row("PSNR_db", base.get("psnr_db",float('nan')), snap.get("psnr_db",float('nan')),"%.3f")
row("MAE_dbz", base.get("mae_dbz",float('nan')), snap.get("mae_dbz",float('nan')),"%.3f")
row("RMSE_dbz", base.get("rmse_dbz",float('nan')), snap.get("rmse_dbz",float('nan')),"%.3f")
row("bias_dbz", base.get("bias_dbz",float('nan')), snap.get("bias_dbz",float('nan')),"%+.3f")
row("area>=5/GT", mass5(base), mass5(snap),"%.3f")
row("sobel p/GT", sob(base), sob(snap),"%.3f")
print("  --- gen_metrics (lower=better for FID/sFID/KID/FVD/KVD) ---")
gb=base.get("gen_metrics",{}); gs=snap.get("gen_metrics",{})
for k in sorted(set(gb)|set(gs)):
    vb,vs=gb.get(k),gs.get(k)
    if isinstance(vb,dict): vb=vb.get("mean")
    if isinstance(vs,dict): vs=vs.get("mean")
    try: print("  %-16s | %.5f -> %.5f  (Δ%+.5f)"%(k,vb,vs,vs-vb))
    except: print("  %-16s | %s -> %s"%(k,vb,vs))
PY
echo "[$(date '+%F %T')] DONE $MODEL"
