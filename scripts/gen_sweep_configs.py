"""Generate the per-index InfoNCE sim_weight-sweep configs (design §3.3).

Index-wise cosine was refuted (alignment-by-collapse: ~80% codebook dead even
at w=0.05/big-budget — results-doc §4/§6/§6.2). The follow-up swaps the sim
loss to per-index symmetric InfoNCE (anti-collapse; learnable CLIP log-temp;
1k-step sim-loss warmup) via the new `joint.sim_loss` selector. InfoNCE is
anti-collapse so higher weights are testable:

  cells   : w ∈ {0.0 (separate baseline), 0.1, 0.5, 1.0}
  AE      : 25k steps
  v2v     : 40k steps

Each cell: clone of the Group-B templates; only joint.sim_weight + the InfoNCE
knobs + per-cell paths + step budgets differ (one-knob ablation, shared
seed/batch order). Distinct `*_infonce_*` run dirs so nothing collides with the
old cosine-sweep artifacts (align_w0*.json / recon_w0*.png are kept).
"""
import re
from pathlib import Path

CFG = Path("/mnt/ssd_1/yghu/Code/FlowTok/configs")
SWEEP = [("w000", 0.0), ("w010", 0.1), ("w050", 0.5), ("w100", 1.0)]
AE_STEPS = 25000
V2V_STEPS = 40000
WARMUP = 1000
INIT_TEMP = 0.07

ae_tpl = (CFG / "joint_ae_joint_lab2.yaml").read_text()
v2v_tpl = (CFG / "Sat2Radar-v2v-jointtok-B-FlowTiTok-S.py").read_text()

# Drop any stale sweep configs (old cosine 4-cell run AND prior infonce runs).
for pat in ("joint_ae_sweep_w*_lab2.yaml",
            "Sat2Radar-v2v-jointtok-sweep-w*-FlowTiTok-S.py",
            "joint_ae_infonce_w*_lab2.yaml",
            "Sat2Radar-v2v-jointtok-infonce-w*-FlowTiTok-S.py"):
    for p in CFG.glob(pat):
        p.unlink()

for tag, w in SWEEP:
    run = f"joint_ae_infonce_{tag}_run1"
    kind = "SEPARATE baseline" if w == 0.0 else f"InfoNCE sim_weight={w}"

    # ── AE yaml ──────────────────────────────────────────────────────────────
    ae = ae_tpl
    ae = re.sub(r"\A(#.*\n)+",
                f"# InfoNCE sim_weight SWEEP cell {tag} ({kind}). Clone of "
                f"joint_ae_joint_lab2.yaml;\n# only experiment.* paths + "
                f"joint.{{sim_weight,sim_loss,...}} + step budget differ. See\n"
                f"# docs/specs/2026-05-18-joint-sat-radar-tokenizer-design.md §3.3\n",
                ae)
    ae = ae.replace("name: joint_ae_joint", f"name: joint_ae_infonce_{tag}")
    ae = ae.replace("/mnt/ssd_2/yghu/Experiments/joint_ae_joint_run1",
                    f"/mnt/ssd_2/yghu/Experiments/{run}")
    # Replace sim_weight AND inject the InfoNCE knobs right after it (still
    # inside the `joint:` block — 2-space indent for the continuation lines).
    ae = re.sub(r"sim_weight:\s*[0-9.]+\s*#.*",
                f"sim_weight: {w}          # InfoNCE SWEEP cell ({kind})\n"
                f"  sim_loss: infonce\n"
                f"  infonce_warmup_steps: {WARMUP}\n"
                f"  infonce_init_temp: {INIT_TEMP}",
                ae, count=1)
    ae = re.sub(r"max_train_steps:\s*\d+",
                f"max_train_steps: {AE_STEPS}", ae, count=1)
    assert (f"sim_weight: {w} " in ae
            and "sim_loss: infonce" in ae
            and f"infonce_warmup_steps: {WARMUP}" in ae
            and f"name: joint_ae_infonce_{tag}" in ae
            and f"max_train_steps: {AE_STEPS}" in ae)
    (CFG / f"joint_ae_infonce_{tag}_lab2.yaml").write_text(ae)

    # ── v2v py (tokenizers frozen at inference — no InfoNCE knowledge) ───────
    v = v2v_tpl
    v = v.replace(
        "# Joint-tokenizer experiment, Group B: FlowTok-S v2v on the joint_ae_joint_run1",
        f"# InfoNCE SWEEP cell {tag} (w={w}, big-budget {V2V_STEPS}-step "
        f"v2v): FlowTok-S on {run}")
    v = v.replace(
        "/mnt/ssd_2/yghu/Experiments/joint_ae_joint_run1/sat/checkpoint-best_val/pytorch_model.bin",
        f"/mnt/ssd_2/yghu/Experiments/{run}/sat/checkpoint-best_val/pytorch_model.bin")
    v = v.replace(
        "/mnt/ssd_2/yghu/Experiments/joint_ae_joint_run1/radar/checkpoint-best_val/pytorch_model.bin",
        f"/mnt/ssd_2/yghu/Experiments/{run}/radar/checkpoint-best_val/pytorch_model.bin")
    v = v.replace('"/mnt/ssd_2/yghu/Experiments/v2v_jointtok_B_run1"',
                  f'"/mnt/ssd_2/yghu/Experiments/v2v_jointtok_infonce_{tag}_run1"')
    v = re.sub(r"n_steps=8_000,", f"n_steps={V2V_STEPS},", v)
    v = re.sub(r"save_interval=4_000,", "save_interval=20_000,", v)
    v = re.sub(r"eval_interval=2_000,", "eval_interval=10_000,", v)
    v = re.sub(r"warmup_steps=2000,", "warmup_steps=4000,", v)
    assert (run in v and f"v2v_jointtok_infonce_{tag}_run1" in v
            and f"n_steps={V2V_STEPS}" in v and "save_interval=20_000" in v)
    (CFG / f"Sat2Radar-v2v-jointtok-infonce-{tag}-FlowTiTok-S.py").write_text(v)
    print(f"wrote {tag}: sim_weight={w} infonce  AE={AE_STEPS}  v2v={V2V_STEPS}")
