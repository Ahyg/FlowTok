#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=05:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=128GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N sampler_ab
# A/B: x1 sampler fixed_x0 (legacy=time-avg, over-smooth) vs canonical (canonical data-pred) on
# m8-i2i@200k & fact-v2v@300k, SAME nofilt subset. Tests the confirmed FM sampling bug fix.
set -uo pipefail
export HF_HOME="/scratch/kl02/$USER/hf_cache"; export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"; export XDG_CACHE_HOME="$HF_HOME"; export HF_HUB_OFFLINE=1; export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
EX=/scratch/kl02/yh0308/Projv2v/Experiments
ROOT=$EX/_eval_sampler_ab_20260617
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_sampler_ab.log
mkdir -p "$ROOT"; exec > "$JOBLOG" 2>&1
cd $FT
I2I_PKL=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_test_202407_202507_nofilter_nan1_clip16.pkl
V2V_PKL=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_test_202407_202507_nofilter_nan1_clip16.pkl

run () {  # tag cfg ckpt pkl mode bs maxb sampler
  local OUT=$ROOT/$1; mkdir -p "$OUT"
  echo "[$(date '+%F %T')] === $1 (sampler=$8) ==="
  python3 -u scripts/test_sat2radar_v2v.py \
    --config $FT/configs/$2 --ckpt "$3" --out_dir "$OUT" \
    --split test --mode $5 --filelist_path "$4" \
    --batch_size $6 --max_batches_metrics $7 --max_batches_images 0 \
    --skip_gen_metrics --x1_sampler $8 \
    --metrics_json "$OUT/metrics.json" --gpu 0
}
M8=$EX/sat2radar_flowtok_i2i_cmp_m8align_B_bl128_cond3nan1/ckpts/200000.ckpt
FA=$EX/sat2radar_flowtok_v2v_cmp_fact_B_bl128_cond3nan1/ckpts/300000.ckpt
for s in fixed_x0 canonical; do
  run "m8_i2i_$s"  Sat2Radar-i2i-cmp-m8align-B-bl128-cond3nan1_gadi.py "$M8" "$I2I_PKL" i2i 16 40 $s
  run "fact_v2v_$s" Sat2Radar-v2v-cmp-fact-B-bl128-cond3nan1_gadi.py  "$FA" "$V2V_PKL" v2v 8 24 $s
done

echo "[$(date '+%F %T')] === COMPARE ==="
python3 - <<'PY'
import json,glob,os
ROOT="/scratch/kl02/yh0308/Projv2v/Experiments/_eval_sampler_ab_20260617"
S=range(1,11); THR=list(range(0,61,5))
def load(p):
    try: return json.load(open(p))
    except: return None
def afss(d,thrs):
    f=d.get("fss_per_thr_scale");
    if not f: return None
    return sum(sum(f[f"thr{t}_scale{s}"] for s in S)/10 for t in thrs)/len(thrs)
def mass5(d):
    h=d.get("refl_hist_60bins");
    if not h: return None
    e=h["bin_edges"]; pd=h["pred_density"]; gd=h["gt_density"]
    def fr(dn):
        tot=hi=0
        for i,v in enumerate(dn):
            w=e[i+1]-e[i]; m=v*w; tot+=m
            if e[i]>=5: hi+=m
        return hi/tot
    return fr(pd)/fr(gd)
def sob(d):
    g=d.get("grad_sobel",{}).get("k1",{});
    pr=g.get("pred_per_sample"); gt=g.get("gt_per_sample")
    if pr and gt: return (sum(pr)/len(pr))/(sum(gt)/len(gt))
    return None
for base in ["m8_i2i","fact_v2v"]:
    a=load(f"{ROOT}/{base}_fixed_x0/metrics.json"); b=load(f"{ROOT}/{base}_canonical/metrics.json")
    if not a or not b: print(f"{base}: missing"); continue
    print(f"\n=== {base}  (fixed_x0 -> canonical) ===")
    print(f"  aFSS0-60 : {afss(a,THR):.4f} -> {afss(b,THR):.4f}  (Δ{afss(b,THR)-afss(a,THR):+.4f})")
    print(f"  aFSS5-60 : {afss(a,[t for t in THR if t>=5]):.4f} -> {afss(b,[t for t in THR if t>=5]):.4f}")
    for t in [5,20,35,45]:
        fa=sum(a['fss_per_thr_scale'][f'thr{t}_scale{s}'] for s in S)/10
        fb=sum(b['fss_per_thr_scale'][f'thr{t}_scale{s}'] for s in S)/10
        print(f"    thr{t:2d} aFSS: {fa:.3f} -> {fb:.3f} ({fb-fa:+.3f})")
    print(f"  area>=5/GT: {mass5(a):.2f} -> {mass5(b):.2f}   bias: {a['bias_dbz']:+.3f} -> {b['bias_dbz']:+.3f}")
    print(f"  sobel(pred/GT): {sob(a):.2f} -> {sob(b):.2f}   MAE: {a['mae_dbz']:.3f} -> {b['mae_dbz']:.3f}  RMSE: {a['rmse_dbz']:.3f} -> {b['rmse_dbz']:.3f}")
PY
echo "[$(date '+%F %T')] DONE"
