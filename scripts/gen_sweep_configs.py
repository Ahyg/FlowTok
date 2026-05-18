"""Generate the sim_weight-sweep configs by cloning the Group-B templates.

For each w in {0.02, 0.05, 0.1, 0.25} (tags w002/w005/w010/w025) emit:
  - configs/joint_ae_sweep_<tag>_lab2.yaml         (clone of joint_ae_joint_lab2.yaml)
  - configs/Sat2Radar-v2v-jointtok-sweep-<tag>-FlowTiTok-S.py (clone of the B v2v cfg)

One-knob ablation: identical to A/B except joint.sim_weight + per-cell paths,
so the sweep is directly comparable to results-doc §6. w=0 (=A) and w=0.5
(=B) are NOT regenerated — their existing runs are reused as anchors.
"""
import re
from pathlib import Path

CFG = Path("/mnt/ssd_1/yghu/Code/FlowTok/configs")
SWEEP = [("w002", 0.02), ("w005", 0.05), ("w010", 0.1), ("w025", 0.25)]

ae_tpl = (CFG / "joint_ae_joint_lab2.yaml").read_text()
v2v_tpl = (CFG / "Sat2Radar-v2v-jointtok-B-FlowTiTok-S.py").read_text()

for tag, w in SWEEP:
    run = f"joint_ae_sweep_{tag}_run1"
    ae = ae_tpl
    ae = ae.replace(
        "# Group A (SEPARATE baseline) — sat AE + radar AE, NO cross-modal loss.\n"
        "# Identical to joint_ae_joint_lab2.yaml except experiment.name/output_dir and\n"
        "# joint.sim_weight (=0.0). One-knob ablation; see",
        f"# sim_weight SWEEP cell: sim_weight={w}. Clone of joint_ae_joint_lab2.yaml;\n"
        f"# only experiment.* paths + joint.sim_weight differ. One-knob ablation; see")
    ae = ae.replace("name: joint_ae_joint", f"name: joint_ae_sweep_{tag}")
    ae = ae.replace(
        "/mnt/ssd_2/yghu/Experiments/joint_ae_joint_run1",
        f"/mnt/ssd_2/yghu/Experiments/{run}")
    ae = re.sub(r"sim_weight:\s*0\.5\b", f"sim_weight: {w}", ae, count=1)
    assert f"sim_weight: {w}" in ae and f"name: joint_ae_sweep_{tag}" in ae
    (CFG / f"joint_ae_sweep_{tag}_lab2.yaml").write_text(ae)

    v = v2v_tpl
    v = v.replace(
        "# Joint-tokenizer experiment, Group B: FlowTok-S v2v on the joint_ae_joint_run1",
        f"# sim_weight SWEEP cell {tag} (w={w}): FlowTok-S v2v on the {run}")
    v = v.replace(
        "/mnt/ssd_2/yghu/Experiments/joint_ae_joint_run1/sat/checkpoint-best_val/pytorch_model.bin",
        f"/mnt/ssd_2/yghu/Experiments/{run}/sat/checkpoint-best_val/pytorch_model.bin")
    v = v.replace(
        "/mnt/ssd_2/yghu/Experiments/joint_ae_joint_run1/radar/checkpoint-best_val/pytorch_model.bin",
        f"/mnt/ssd_2/yghu/Experiments/{run}/radar/checkpoint-best_val/pytorch_model.bin")
    v = v.replace(
        '"/mnt/ssd_2/yghu/Experiments/v2v_jointtok_B_run1"',
        f'"/mnt/ssd_2/yghu/Experiments/v2v_jointtok_sweep_{tag}_run1"')
    assert run in v and f"v2v_jointtok_sweep_{tag}_run1" in v
    (CFG / f"Sat2Radar-v2v-jointtok-sweep-{tag}-FlowTiTok-S.py").write_text(v)
    print(f"wrote {tag}: sim_weight={w} -> joint_ae_sweep_{tag}_lab2.yaml + "
          f"Sat2Radar-v2v-jointtok-sweep-{tag}-FlowTiTok-S.py")
