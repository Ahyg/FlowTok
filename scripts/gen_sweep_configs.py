"""Generate the (reduced, big-budget) sim_weight-sweep configs.

The 8k/15k pilot was visually unconverged (results-doc §2 / the radar panels:
negative R², CSI35=0, low-freq blobs). Per the user's call we drop to a small
weight set and give every cell a much larger budget so the decoded-radar
verdict is trustworthy:

  cells   : w ∈ {0.0 (separate baseline), 0.05, 0.25}   (was 4 + 2 anchors)
  AE      : 25k steps   (was 15k)
  v2v     : 40k steps   (was 8k)

Each cell: clone of the Group-B templates, only joint.sim_weight + per-cell
paths + step budgets differ (one-knob ablation, shared seed/batch order).
After AE training, eval_joint_tokenizer.py also dumps a pure encode->decode
recon panel (recon_<tag>.png) — the tokenizer ceiling reference.
"""
import re
from pathlib import Path

CFG = Path("/mnt/ssd_1/yghu/Code/FlowTok/configs")
SWEEP = [("w000", 0.0), ("w005", 0.05), ("w025", 0.25)]
AE_STEPS = 25000
V2V_STEPS = 40000

ae_tpl = (CFG / "joint_ae_joint_lab2.yaml").read_text()
v2v_tpl = (CFG / "Sat2Radar-v2v-jointtok-B-FlowTiTok-S.py").read_text()

# Drop any stale sweep configs from the previous (4-cell, small-budget) run.
for p in list(CFG.glob("joint_ae_sweep_w*_lab2.yaml")) + \
        list(CFG.glob("Sat2Radar-v2v-jointtok-sweep-w*-FlowTiTok-S.py")):
    p.unlink()

for tag, w in SWEEP:
    run = f"joint_ae_sweep_{tag}_run1"
    kind = "SEPARATE baseline" if w == 0.0 else f"JOINT sim_weight={w}"

    # ── AE yaml ──────────────────────────────────────────────────────────────
    ae = ae_tpl
    ae = re.sub(r"\A(#.*\n)+",
                f"# sim_weight SWEEP cell {tag} ({kind}). Clone of "
                f"joint_ae_joint_lab2.yaml;\n# only experiment.* paths + "
                f"joint.sim_weight + step budget differ. See\n"
                f"# docs/specs/2026-05-18-joint-sat-radar-tokenizer-design.md\n",
                ae)
    ae = ae.replace("name: joint_ae_joint", f"name: joint_ae_sweep_{tag}")
    ae = ae.replace("/mnt/ssd_2/yghu/Experiments/joint_ae_joint_run1",
                    f"/mnt/ssd_2/yghu/Experiments/{run}")
    ae = re.sub(r"sim_weight:\s*[0-9.]+\s*#.*",
                f"sim_weight: {w}          # SWEEP cell ({kind})", ae, count=1)
    ae = re.sub(r"max_train_steps:\s*\d+",
                f"max_train_steps: {AE_STEPS}", ae, count=1)
    assert (f"sim_weight: {w} " in ae and f"name: joint_ae_sweep_{tag}" in ae
            and f"max_train_steps: {AE_STEPS}" in ae)
    (CFG / f"joint_ae_sweep_{tag}_lab2.yaml").write_text(ae)

    # ── v2v py ───────────────────────────────────────────────────────────────
    v = v2v_tpl
    v = v.replace(
        "# Joint-tokenizer experiment, Group B: FlowTok-S v2v on the joint_ae_joint_run1",
        f"# sim_weight SWEEP cell {tag} (w={w}, big-budget {V2V_STEPS}-step "
        f"v2v): FlowTok-S on {run}")
    v = v.replace(
        "/mnt/ssd_2/yghu/Experiments/joint_ae_joint_run1/sat/checkpoint-best_val/pytorch_model.bin",
        f"/mnt/ssd_2/yghu/Experiments/{run}/sat/checkpoint-best_val/pytorch_model.bin")
    v = v.replace(
        "/mnt/ssd_2/yghu/Experiments/joint_ae_joint_run1/radar/checkpoint-best_val/pytorch_model.bin",
        f"/mnt/ssd_2/yghu/Experiments/{run}/radar/checkpoint-best_val/pytorch_model.bin")
    v = v.replace('"/mnt/ssd_2/yghu/Experiments/v2v_jointtok_B_run1"',
                  f'"/mnt/ssd_2/yghu/Experiments/v2v_jointtok_sweep_{tag}_run1"')
    v = re.sub(r"n_steps=8_000,", f"n_steps={V2V_STEPS},", v)
    v = re.sub(r"save_interval=4_000,", "save_interval=20_000,", v)
    v = re.sub(r"eval_interval=2_000,", "eval_interval=10_000,", v)
    v = re.sub(r"warmup_steps=2000,", "warmup_steps=4000,", v)
    assert (run in v and f"v2v_jointtok_sweep_{tag}_run1" in v
            and f"n_steps={V2V_STEPS}" in v and "save_interval=20_000" in v)
    (CFG / f"Sat2Radar-v2v-jointtok-sweep-{tag}-FlowTiTok-S.py").write_text(v)
    print(f"wrote {tag}: sim_weight={w}  AE={AE_STEPS}  v2v={V2V_STEPS}")
