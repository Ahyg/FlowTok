#!/bin/bash
#PBS -P ui54
#PBS -q gpuhopper
#PBS -l walltime=01:30:00
#PBS -l storage=gdata/kl02+scratch/kl02
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=128GB
#PBS -l jobfs=90GB
#PBS -l wd
#PBS -M auhuyg@gmail.com
#PBS -m abe
#PBS -N pixf_i2i_gate
# Pre-flight gate for pixfact-i2i (pixel-space factorized DiT + flow, T=1) before ~100 GPU-h.
# Feasibility was proven on synthetic tensors (forward/loss/backward finite at T=1); this exercises
# the parts that were NOT: the REAL npy dataloader at num_frames=1 with pixel_space, an actual
# accelerate training step, checkpoint save, and KILL+RESUME (the self-resubmit chain a 600k run needs).
# It overrides only n_steps / save_interval / workdir via config_flags (lock_config=False), leaving the
# real config untouched. Measures s/step so the 9 kSU estimate becomes a fact.
set -uo pipefail
export HF_HOME="/scratch/kl02/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$HF_HOME"
export XDG_CACHE_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
conda activate flowtok
export PYTHONUNBUFFERED=1

FT=/scratch/kl02/$USER/Projv2v/FlowTok
CFG=$FT/configs/Sat2Radar-i2i-cmp-pixfact-B-p8-cond3nan1_gadi.py
GATE=/scratch/kl02/yh0308/Projv2v/Experiments/_pixfact_i2i_gate_20260723
mkdir -p "$GATE/ckpts" /scratch/kl02/$USER/Projv2v/job_logs
cd $FT
echo "=========================================================================="
echo "[$(date '+%F %T')] pixfact-i2i gate  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================================================="

# ---- 1. build + param match (device=cpu, quick) -------------------------
echo ""; echo "### 1. build model from the real config + param count"
python3 - "$CFG" <<'PY'
import sys, importlib.util
spec=importlib.util.spec_from_file_location("cfg", sys.argv[1])
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
c=m.get_config()
import flow_utils
nnet=flow_utils.get_nnet(**c.nnet)
n=sum(p.numel() for p in nnet.parameters())
print(f"  DiT params = {n:,}  (expect 229,735,008 = pixfact-v2v)")
assert n==229735008, "param count drifted from pixfact-v2v!"
print(f"  num_frames={c.dataset.num_frames}  pixel_space={c.get('pixel_space')}  "
      f"channels={c.nnet.model_args.channels}  cond_channels={c.nnet.model_args.cond_channels}  "
      f"num_latent_tokens={c.nnet.model_args.num_latent_tokens}")
assert c.dataset.num_frames==1 and c.get('pixel_space') is True
print("  OK")
PY
[ $? -ne 0 ] && { echo "!! GATE FAILED at step 1"; exit 1; }

OV="--config.workdir=$GATE --config.ckpt_root=$GATE/ckpts --config.sample_dir=$GATE/samples --config.train.save_interval=250 --config.train.eval_interval=100000"
latest_step() { ls -d "$GATE"/ckpts/*.ckpt 2>/dev/null | grep -oP '[0-9]+(?=\.ckpt)' | sort -n | tail -1; }

# ---- 2. leg 1: real data, real accelerate, to step 250 ------------------
echo ""; echo "### 2. leg 1 -> 250 steps on the REAL i2i cond3nan1 pkl"
accelerate launch --num_processes 1 scripts/train_sat2radar_v2v.py --config="$CFG" \
  $OV --config.train.n_steps=250 > "$GATE/leg1.log" 2>&1
RC=$?; S=$(latest_step); S=${S:-0}
echo "  rc=$RC  ckpt_step=$S"
if [ "$RC" -ne 0 ] || [ "$S" -lt 250 ]; then
  echo "!! GATE FAILED: leg1 rc=$RC step=$S"; tail -40 "$GATE/leg1.log"; exit 1
fi
echo "  it/s leg1: $(grep -oE '[0-9.]+(it/s|s/it)' "$GATE/leg1.log" | tail -1)"
echo "  loss trail: $(grep -oE 'loss[=: ]+[0-9.]+' "$GATE/leg1.log" | tail -3 | tr '\n' ' ')"

# ---- 3. leg 2: RESUME from 250 -> 500 -----------------------------------
echo ""; echo "### 3. leg 2 RESUME 250 -> 500 (the self-resubmit path)"
accelerate launch --num_processes 1 scripts/train_sat2radar_v2v.py --config="$CFG" \
  $OV --config.train.n_steps=500 > "$GATE/leg2.log" 2>&1
RC=$?; S2=$(latest_step); S2=${S2:-0}
echo "  rc=$RC  ckpt_step=$S2"
if [ "$RC" -ne 0 ] || [ "$S2" -le 250 ]; then
  echo "!! GATE FAILED: resume did not advance past 250 (rc=$RC step=$S2)"; tail -40 "$GATE/leg2.log"; exit 1
fi
grep -qiE "resume|loaded.*ckpt|restor" "$GATE/leg2.log" && echo "  resume confirmed in log"
echo "  it/s leg2: $(grep -oE '[0-9.]+(it/s|s/it)' "$GATE/leg2.log" | tail -1)"

# ---- 4. loss sanity across the resume boundary --------------------------
python3 - "$GATE" <<'PY'
import re, sys, os
def losses(p):
    txt=open(p,errors="ignore").read()
    return [float(x) for x in re.findall(r"loss[=: ]+([0-9.]+)", txt)]
l1=losses(os.path.join(sys.argv[1],"leg1.log"))[-30:]
l2=losses(os.path.join(sys.argv[1],"leg2.log"))[-30:]
if l1 and l2:
    a,b=sum(l1)/len(l1),sum(l2)/len(l2)
    print(f"  loss  leg1_end~{a:.4f}  leg2_end~{b:.4f}  ratio {b/a:.3f}")
    assert b < a*3.0, "loss jumped across resume -- optimizer state did not restore"
print("  loss sanity OK")
PY
[ $? -ne 0 ] && { echo "!! GATE FAILED: loss sanity"; exit 1; }

echo ""
echo "=========================================================================="
echo "[$(date '+%F %T')] PIXFACT-I2I GATE PASSED"
echo "throughput (use this for the SU estimate):"
echo "  leg2 rate: $(grep -oE '[0-9.]+(it/s|s/it)' "$GATE/leg2.log" | tail -1)"
echo "=========================================================================="
