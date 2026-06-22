#!/bin/bash
#PBS -P kl02
#PBS -q gpuhopper
#PBS -l walltime=03:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=128GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N x1snap
# Experiment: one-shot clean-radar readout (--x1_snap_t). Baseline (snap=-1 = current fixed_x0,
# should reproduce production aFSS ~0.42 m8 / ~0.45 fact) vs snapping x1_hat at late t in
# {0.7,0.8,0.9,0.95,0.99}. Same nofilt subset, same config/ckpt. Replaces the UNTRUSTWORTHY
# A/B 171486572 whose fixed_x0 arm (0.077) ran a transient broken code state.
set -uo pipefail
export HF_HOME="/scratch/kl02/$USER/hf_cache"; export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"; export XDG_CACHE_HOME="$HF_HOME"; export HF_HUB_OFFLINE=1; export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1
FT=/scratch/kl02/$USER/Projv2v/FlowTok
EX=/scratch/kl02/yh0308/Projv2v/Experiments
ROOT=$EX/_eval_x1snap_20260617
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_x1snap.log
mkdir -p "$ROOT"; exec > "$JOBLOG" 2>&1
cd $FT
I2I_PKL=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_test_202407_202507_nofilter_nan1_clip16.pkl
V2V_PKL=/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_test_202407_202507_nofilter_nan1_clip16.pkl
M8=$EX/sat2radar_flowtok_i2i_cmp_m8align_B_bl128_cond3nan1/ckpts/200000.ckpt
FA=$EX/sat2radar_flowtok_v2v_cmp_fact_B_bl128_cond3nan1/ckpts/300000.ckpt

run () {  # tag cfg ckpt pkl mode bs maxb snap
  local OUT=$ROOT/$1; mkdir -p "$OUT"
  echo "[$(date '+%F %T')] === $1 (x1_snap_t=$8) ==="
  python3 -u scripts/test_sat2radar_v2v.py \
    --config $FT/configs/$2 --ckpt "$3" --out_dir "$OUT" \
    --split test --mode $5 --filelist_path "$4" \
    --batch_size $6 --max_batches_metrics $7 --max_batches_images 0 \
    --skip_gen_metrics --x1_snap_t $8 \
    --metrics_json "$OUT/metrics.json" --gpu 0
}

I2I_CFG=Sat2Radar-i2i-cmp-m8align-B-bl128-cond3nan1_gadi.py
V2V_CFG=Sat2Radar-v2v-cmp-fact-B-bl128-cond3nan1_gadi.py

tag () { local s=$1; if [ "$s" = "-1" ]; then echo "baseline"; else echo "snap${s/./}"; fi; }
for s in -1 0.7 0.8 0.9 0.95 0.99; do
  run "m8_i2i_$(tag $s)"   $I2I_CFG "$M8" "$I2I_PKL" i2i 16 40 $s
  run "fact_v2v_$(tag $s)" $V2V_CFG "$FA" "$V2V_PKL" v2v 8  50 $s
done

echo "[$(date '+%F %T')] === COMPARE ==="
python3 - <<'PY'
import json,os
ROOT="/scratch/kl02/yh0308/Projv2v/Experiments/_eval_x1snap_20260617"
S=range(1,11); THR=list(range(0,61,5))
SNAPS=["-1","0.7","0.8","0.9","0.95","0.99"]
def tag(s): return "baseline" if s=="-1" else "snap"+s.replace(".","")
def load(p):
    try: return json.load(open(p))
    except: return None
def afss(d,thrs):
    f=d.get("fss_per_thr_scale")
    if not f: return None
    return sum(sum(f["thr%d_scale%d"%(t,sc)] for sc in S)/10 for t in thrs)/len(thrs)
def thr(d,t): return sum(d["fss_per_thr_scale"]["thr%d_scale%d"%(t,sc)] for sc in S)/10
def mass5(d):
    h=d.get("refl_hist_60bins")
    if not h: return None
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
    if pr and gt: return (sum(pr)/len(pr))/(sum(gt)/len(gt))
    return None
PROD={"m8_i2i":0.4224,"fact_v2v":0.4501}
for base in ["m8_i2i","fact_v2v"]:
    print("\n=== %s   (production fixed_x0 holdout aFSS = %.3f) ===" % (base, PROD[base]))
    print("  %-9s | aFSS0-60 | thr5  thr20 thr35 thr45 | area>=5/GT | sobel | bias" % "arm")
    for s in SNAPS:
        d=load("%s/%s_%s/metrics.json"%(ROOT,base,tag(s)))
        if not d: print("  %-9s | MISSING"%(tag(s))); continue
        a=afss(d,THR); m5=mass5(d); sb=sob(d)
        print("  %-9s |  %.4f  | %.3f %.3f %.3f %.3f |   %s   | %s | %+.3f" % (
            tag(s), a, thr(d,5),thr(d,20),thr(d,35),thr(d,45),
            ("%.2f"%m5) if m5 else "  -", ("%.2f"%sb) if sb else " - ", d.get("bias_dbz",float('nan'))))
PY
echo "[$(date '+%F %T')] DONE"
