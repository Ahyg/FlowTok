#!/bin/bash
# Decode-and-compare the two joint-tokenizer v2v pilots (Group A separate AE
# vs Group B joint AE) in PIXEL space — resolves the §5 confound that the
# overnight pilot only tracked token-space loss.
#
# Identical eval settings for both; same 2024/07 v2v test clips. Saves dBZ
# radar metrics (MSE/MAE/SSIM/FSS/CSI) + decoded radar image panels, then
# diffs A vs B into the results doc.
set -u
ROOT="/mnt/ssd_1/yghu/Code/FlowTok"
EXP="/mnt/ssd_2/yghu/Experiments"
TEST_PKL="/mnt/ssd_1/yghu/Data/71_3m/filelists/dataset_filelist_v2v_test_202407.pkl"
MLOG="/tmp/jointtok_v2v_compare.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "${MLOG}"; }

source /home/yghu/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

for G in A B; do
  CFG="configs/Sat2Radar-v2v-jointtok-${G}-FlowTiTok-S.py"
  CKPT="${EXP}/v2v_jointtok_${G}_run1/ckpts/8000.ckpt"
  OUT="${EXP}/v2v_jointtok_${G}_run1/test8000"
  log "=== START decode-eval Group ${G} ==="
  if python -u scripts/test_sat2radar_v2v.py \
      --config "${CFG}" --ckpt "${CKPT}" --out_dir "${OUT}" \
      --mode v2v --split test --filelist_path "${TEST_PKL}" \
      --batch_size 8 --max_batches_metrics 50 --max_batches_images 4 \
      --skip_gen_metrics --gpu 0 \
      --metrics_json "${OUT}/metrics.json" >>"${MLOG}" 2>&1; then
    log "=== OK Group ${G} -> ${OUT}/metrics.json ==="
  else
    log "=== FAIL Group ${G} (exit $?) ==="
  fi
done

log "=== diff A vs B ==="
python - <<'PY' 2>&1 | tee -a "${MLOG}"
import json, pathlib
EXP="/mnt/ssd_2/yghu/Experiments"
RES="/mnt/ssd_1/yghu/Code/FlowTok/docs/specs/results/2026-05-18-joint-tokenizer-results.md"
def load(g):
    p=f"{EXP}/v2v_jointtok_{g}_run1/test8000/metrics.json"
    try: return json.load(open(p))
    except Exception as e: return {"_err":str(e)}
A,B=load("A"),load("B")
if "_err" in A or "_err" in B:
    print("MISSING:", A.get("_err"), B.get("_err")); raise SystemExit
# lower better: mse/mae/rmse/far ; higher better: psnr/ssim/r2/fss/csi/pod
rows=[("mse_dbz","↓"),("mae_dbz","↓"),("rmse_dbz","↓"),("psnr_db","↑"),
      ("ssim","↑"),("r2","↑"),("avg_fss","↑"),("weighted_fss","↑"),
      ("csi35","↑"),("pod35","↑"),("far35","↓")]
def fmt(v):
    return f"{v:.4f}" if isinstance(v,(int,float)) else str(v)
lines=["","---","",
 "## 6. Decoded-radar pixel-space comparison (added — resolves §5 confound)","",
 f"`test_sat2radar_v2v.py`, 2024/07 v2v test, {A.get('seen_samples','?')} clips "
 f"({A.get('n_frames','?')} frames/clip), step-8000 pilots, identical settings. "
 "Metrics on dBZ radar after full decode (sat→flow→radar detokenizer).","",
 "| metric | dir | A (separate AE) | B (joint AE) | better |","|---|---|---|---|---|"]
better_B=better_A=0
for k,d in rows:
    a,b=A.get(k),B.get(k)
    if not isinstance(a,(int,float)) or not isinstance(b,(int,float)):
        lines.append(f"| {k} | {d} | {fmt(a)} | {fmt(b)} | — |"); continue
    impr_B = (b<a) if d=="↓" else (b>a)
    win = "**B**" if impr_B else ("**A**" if a!=b else "tie")
    if impr_B: better_B+=1
    elif a!=b: better_A+=1
    lines.append(f"| {k} | {d} | {a:.4f} | {b:.4f} | {win} |")
verdict=("B (joint) better" if better_B>better_A else
         "A (separate) better" if better_A>better_B else "mixed/tie")
lines += ["",
 f"**Tally:** B wins {better_B} / A wins {better_A} → **decoded radar: {verdict}**.","",
 "Read with the §4 caveat: B's tokenizer is collapsed/low-rank, so a lower "
 "token-loss did not necessarily mean better pixels. This table is the "
 "decisive check the pilot was missing. Panels: "
 "`/mnt/ssd_2/yghu/Experiments/v2v_jointtok_{A,B}_run1/test8000/`.",""]
doc=pathlib.Path(RES); txt=doc.read_text()
marker="## 6. Decoded-radar pixel-space comparison"
if marker in txt: txt=txt.split("\n---\n\n"+marker)[0].rstrip()
doc.write_text(txt+"\n"+"\n".join(lines)+"\n")
print(f"verdict={verdict} (B {better_B} / A {better_A}); wrote §6 into {RES}")
PY
log "ALL DONE."
