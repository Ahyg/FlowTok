#!/bin/bash
#PBS -P ui54
#PBS -q gpuhopper
#PBS -l walltime=06:00:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N x1snap_pxf600
# Full-set snap099 inference on the pixfact @600k ckpt, FULL NOFILT test set (46,400 frames).
# Sampler lever, eval-only, NO retrain:
#   snap099 : --x1_snap_t 0.99  -> at the LAST flow node output the DiT's clean-radar x1_hat
#                                  directly and stop, instead of the fixed_x0 Euler integration
#                                  (which telescopes to a time-AVERAGE of x1_hat over all nodes).
# Grid check: eval nodes = linspace(0, 0.99999, sample_steps=20) -> t_18 = 0.94736, t_19 = 0.99999.
#   0.99 therefore fires at i=19 ONLY = exactly "read the last flow step". Same 20 model evals as
#   the baseline, so ~same runtime.
# Guard in diffusion/flow_matching.py:540 requires prediction_target=="radar_tokens" -- the pixfact
# config sets flow_prediction_target="radar_tokens" (line 125), so the switch does bite.
# Replicates the production holdout invocation (same ckpt / nofilt filelist / batch_size 8 /
# max_batches_metrics=-1 / seed 42 / gen metrics ON); differs ONLY in the snap flag and in dumping
# pred-only arrays (the baseline already wrote the shared gt_dbz.npy, so re-dumping it wastes 2.8 GiB).
# Per-batch reseeding (seed+batch_idx) makes this a PAIRED comparison: bit-identical initial noise.
# Baseline = test_holdout_600000_nofilt, written by holdout job 174224271 (finished 2026-07-20 22:05:37,
# Exit Status 0, n_frames 46400, avg_fss 0.5061).
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

CFG=$FT/configs/Sat2Radar-v2v-cmp-pixfact-B-p8-cond3nan1_gadi.py
WD=$EXP/sat2radar_flowtok_v2v_cmp_pixfact_B_p8_cond3nan1
STEP=600000; MODE=v2v; BS=8
PKL=$NF/dataset_filelist_v2v_test_202407_202507_nofilter_nan1_clip16.pkl

CKPT=$WD/ckpts/${STEP}.ckpt
BASE_DIR=$WD/test_holdout_${STEP}_nofilt            # fixed_x0 Euler baseline (compare target)
SNAP_DIR=$WD/test_holdout_${STEP}_nofilt_snap099    # snap099 output
mkdir -p "$SNAP_DIR" /scratch/kl02/$USER/Projv2v/job_logs
JOBLOG=/scratch/kl02/$USER/Projv2v/job_logs/${PBS_JOBID}_x1snap_pixfact600k.log
exec > "$JOBLOG" 2>&1
cd $FT
[ -e "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
[ -e "$PKL" ]  || { echo "missing $PKL"; exit 1; }
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "[$(date '+%F %T')] pixfact @${STEP} snap099 on FULL nofilt -> $SNAP_DIR"
python3 -u scripts/test_sat2radar_v2v.py \
  --config "$CFG" --ckpt "$CKPT" --out_dir "$SNAP_DIR" \
  --split test --mode $MODE --filelist_path "$PKL" \
  --max_batches_metrics -1 --max_batches_images 6 \
  --batch_size $BS --x1_snap_t 0.99 \
  --dump_arrays --dump_pred_only --arrays_dir "$SNAP_DIR/arrays" \
  --metrics_json "$SNAP_DIR/metrics.json" --gpu "$CUDA_VISIBLE_DEVICES"
RC=$?
# NOTE: judge success by metrics.json existing, NOT by any rc= line -- an
# echo "... rc=$?" after a $(date) substitution always prints 0.
echo "[$(date '+%F %T')] python exited rc=$RC"
if [ ! -s "$SNAP_DIR/metrics.json" ]; then
  echo "!! FAILED: $SNAP_DIR/metrics.json missing or empty -- snap099 run did not complete."
  exit 1
fi
echo "[$(date '+%F %T')] snap099 metrics.json written OK"

if [ ! -s "$BASE_DIR/metrics.json" ]; then
  echo "[$(date '+%F %T')] NOTE: baseline $BASE_DIR/metrics.json not present yet"
  echo "  (holdout job 174224271 may still be running). snap099 results are saved;"
  echo "  re-run the compare block later against the baseline."
  exit 0
fi

echo "[$(date '+%F %T')] === COMPARE snap099 vs fixed_x0 Euler baseline (pixfact @${STEP}, nofilt) ==="
python3 - "$BASE_DIR/metrics.json" "$SNAP_DIR/metrics.json" <<'PY'
import json, sys
base = json.load(open(sys.argv[1])); snap = json.load(open(sys.argv[2]))
S = range(1, 11)
nb, ns = base.get("n_frames"), snap.get("n_frames")
ok = (nb is not None and ns is not None and nb == ns)
print("  n_frames base/snap:", nb, ns, "" if ok else "  <<< MISSING or MISMATCHED: NOT comparable!")
print("  seen_samples base/snap:", base.get("seen_samples"), snap.get("seen_samples"))
def thr(d, t):
    try: return sum(d["fss_per_thr_scale"]["thr%d_scale%d" % (t, sc)] for sc in S) / 10
    except Exception: return float('nan')
def sob(d, k="k3", which="pred_mean"):
    # grad_sobel is a NESTED dict {k1,k3,k5,k7} -> {pred_per_sample, gt_per_sample,
    # pred_mean, pred_std, gt_mean, gt_std}, NOT a scalar. Reduce before printing.
    try: return float(d["grad_sobel"][k][which])
    except Exception: return float('nan')
def row(name, fb, fs, fmt="%.4f"):
    # print INSIDE the try: any non-scalar value degrades to a %s line instead of
    # raising TypeError and aborting the rest of the report (incl. gen_metrics).
    try:
        print(("  %-16s | " + fmt + " -> " + fmt + "  (delta %+.4f)") % (name, fb, fs, fs - fb))
    except Exception:
        print("  %-16s | %s -> %s" % (name, fb, fs))
row("avg_fss(0-60)", base.get("avg_fss", float('nan')), snap.get("avg_fss", float('nan')))
row("weighted_fss",  base.get("weighted_fss", float('nan')), snap.get("weighted_fss", float('nan')))
for t in [5, 15, 25, 35, 45]:
    row("aFSS thr%d" % t, thr(base, t), thr(snap, t), "%.3f")
row("CSI35",    base.get("csi35", float('nan')),    snap.get("csi35", float('nan')))
row("HSS35",    base.get("hss35", float('nan')),    snap.get("hss35", float('nan')))
row("POD35",    base.get("pod35", float('nan')),    snap.get("pod35", float('nan')))
row("FAR35",    base.get("far35", float('nan')),    snap.get("far35", float('nan')))
row("SSIM",     base.get("ssim", float('nan')),     snap.get("ssim", float('nan')))
row("PSNR_db",  base.get("psnr_db", float('nan')),  snap.get("psnr_db", float('nan')), "%.3f")
row("RMSE_dbz", base.get("rmse_dbz", float('nan')), snap.get("rmse_dbz", float('nan')), "%.3f")
row("MAE_dbz",  base.get("mae_dbz", float('nan')),  snap.get("mae_dbz", float('nan')), "%.3f")
row("R2",       base.get("r2", float('nan')),       snap.get("r2", float('nan')))
row("bias_dbz", base.get("bias_dbz", float('nan')), snap.get("bias_dbz", float('nan')), "%+.3f")
print("  --- sharpness (Sobel gradient magnitude; GT is the target to match) ---")
for k in ["k1", "k3", "k5", "k7"]:
    row("sobel_%s pred" % k, sob(base, k, "pred_mean"), sob(snap, k, "pred_mean"), "%.4f")
print("  %-16s | gt_mean k3 = %.4f (identical for both arms)" % ("sobel ref", sob(base, "k3", "gt_mean")))
print("  --- gen_metrics (lower=better) ---")
gb = base.get("gen_metrics", {}) or {}; gs = snap.get("gen_metrics", {}) or {}
for k in sorted(set(gb) | set(gs)):
    vb, vs = gb.get(k), gs.get(k)
    if isinstance(vb, dict): vb = vb.get("mean")
    if isinstance(vs, dict): vs = vs.get("mean")
    try: print("  %-16s | %.5f -> %.5f  (delta %+.5f)" % (k, vb, vs, vs - vb))
    except Exception: print("  %-16s | %s -> %s" % (k, vb, vs))
PY
CRC=$?
if [ "$CRC" -ne 0 ]; then
  echo "!! compare block failed rc=$CRC -- snap099 metrics.json IS still valid and complete;"
  echo "   only the side-by-side printout is missing. Re-run the compare by hand."
  exit "$CRC"
fi
echo "[$(date '+%F %T')] DONE pixfact @${STEP} nofilt snap099"
