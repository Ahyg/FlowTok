#!/bin/bash
#PBS -P ui54
#PBS -q normal
#PBS -l walltime=03:00:00
#PBS -l ncpus=8
#PBS -l mem=48GB
#PBS -l jobfs=10GB
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N crps_pixfact
# Accumulated CRPS for the pixfact arm from its 16-seed ensemble -- the last empty cell in the
# 11-model table. Same protocol as the other 8 stochastic arms (_crps_seedvar_20260709):
# pysteps-style accumulation over K members, thresholds {all,>=5,>=20,>=35,>=45,>=55}, lower=better dBZ.
# Members are written by seedvar600k_flowtok_arrays_gadi.sh MODEL=pixfact GRP=0..3.
set -uo pipefail
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1

CD=/scratch/kl02/yh0308/Projv2v/Experiments/_crps_seedvar_20260709
SV=/scratch/kl02/yh0308/Projv2v/Experiments/_seed_variance_600k_20260624
ARM=pixfact_v2v_600k_ns20
GT=$CD/gt_dbz_nofilt_46400.npy          # canonical, frame-order-verified against both i2i and v2v
OUT=$CD/crps_${ARM}.json
cd $CD
[ -e "$GT" ] || { echo "!! missing canonical GT $GT"; exit 1; }

# Refuse a partial ensemble. A K<16 CRPS is NOT comparable with the other arms, and
# compute_crps_accum.py would happily compute one from whatever happens to be on disk.
MISSING=""
for S in $(seq 0 15); do
  [ -s "$SV/$ARM/seed${S}/arrays/pred_dbz.npy" ] || MISSING="$MISSING $S"
done
if [ -n "$MISSING" ]; then
  echo "!! incomplete ensemble -- missing pred_dbz.npy for seeds:$MISSING"
  echo "   Re-run the group jobs (they are resumable: existing seeds are skipped), then resubmit this."
  exit 1
fi
echo "[$(date '+%F %T')] all 16 members present for $ARM"

# GT-alignment guard. verify_mse compares member0 (= seed0, glob sorts lexicographically) against
# the MSE the eval harness itself recorded for that seed; a mismatch means the GT frame order is
# wrong. Read it at runtime -- it is seed-specific and cannot be hardcoded.
VM=$(python3 -c "import json;print('%.4f'%json.load(open('$SV/$ARM/seed0/metrics.json'))['mse_dbz'])") || {
  echo "!! could not read seed0 mse_dbz"; exit 1; }
echo "[$(date '+%F %T')] seed0 harness mse_dbz = $VM (used as the GT-alignment assertion)"

python3 -u compute_crps_accum.py \
  --seed_glob "$SV/$ARM/seed*/arrays/pred_dbz.npy" \
  --gt "$GT" --label "$ARM" --verify_mse "$VM" \
  --out "$OUT"
RC=$?
# NOTE: judge by the json existing, never by an rc= line printed after a $(date) substitution.
echo "[$(date '+%F %T')] compute_crps_accum exited rc=$RC"
[ -s "$OUT" ] || { echo "!! FAILED: $OUT missing or empty"; exit 1; }

python3 - "$OUT" "$CD" <<'PY'
import json, sys, glob, os
d = json.load(open(sys.argv[1])); CD = sys.argv[2]
print("\n=== pixfact accumulated CRPS (K=%d, N=%d) ===" % (d["n_members"], d["n_frames"]))
for k in ["crps_all", "crps_ge5", "crps_ge20", "crps_ge35", "crps_ge45", "crps_ge55"]:
    print("  %-10s %8.4f   (npix %s)" % (k, d.get(k, float("nan")), f'{d.get(k+"_npix", 0):,}'))
print("\n=== all 9 stochastic arms, ranked by overall CRPS (lower = better) ===")
rows = []
for f in sorted(glob.glob(os.path.join(CD, "crps_*.json"))):
    if os.path.basename(f) == "crps_summary.json":
        continue
    try:
        r = json.load(open(f))
    except Exception:
        continue
    if "crps_all" in r:
        rows.append(r)
rows.sort(key=lambda r: r["crps_all"])
print("%-26s%4s%10s%9s%9s%9s%9s%9s" % ("model", "K", "CRPS", ">=5", ">=20", ">=35", ">=45", ">=55"))
for r in rows:
    print("%-26s%4d%10.4f%9.2f%9.2f%9.2f%9.2f%9.2f" % (
        r["label"], r["n_members"], r["crps_all"], r.get("crps_ge5", float("nan")),
        r.get("crps_ge20", float("nan")), r.get("crps_ge35", float("nan")),
        r.get("crps_ge45", float("nan")), r.get("crps_ge55", float("nan"))))
json.dump(rows, open(os.path.join(CD, "crps_summary.json"), "w"), indent=2)
print("\nrewrote crps_summary.json with %d arms" % len(rows))
PY
echo "[$(date '+%F %T')] DONE crps pixfact"
