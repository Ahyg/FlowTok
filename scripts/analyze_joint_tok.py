"""Roll Group A vs Group B into the results doc, applying design §5.

--phase ae    : tokenizer-phase verdict from align_{A,B}.json (the primary result)
--phase final : also fold in the v2v pilot loss curves

Defensive by design: this runs unattended overnight; missing inputs degrade to
"NOT AVAILABLE" rather than crashing the pipeline.
"""
import argparse
import glob
import json
import re
from pathlib import Path

RESULTS = "docs/specs/results/2026-05-18-joint-tokenizer-results.md"


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception as e:                                  # noqa: BLE001
        return {"_error": f"{p}: {e}"}


def _pct(new, old):
    if old in (0, None):
        return float("nan")
    return 100.0 * (new - old) / abs(old)


def _v2v_loss_tail(workdir):
    """Best-effort: last few 'loss=' numbers from any log under workdir."""
    logs = glob.glob(f"{workdir}/**/*.log", recursive=True) + \
        glob.glob(f"{workdir}/*.txt") + glob.glob(f"{workdir}/log*")
    vals = []
    for lg in sorted(logs):
        try:
            txt = Path(lg).read_text(errors="ignore")
        except Exception:                                   # noqa: BLE001
            continue
        # v2v trainer logs dicts via dct2str -> "'loss': '0.0123'"; also
        # tolerate unquoted / "loss=0.0123" forms.
        for m in re.finditer(r"'loss':\s*'?([0-9][0-9.eE+-]*)'?", txt):
            vals.append(float(m.group(1)))
        if not vals:
            for m in re.finditer(r"\bloss[=: ]+([0-9][0-9.eE+-]*)", txt):
                vals.append(float(m.group(1)))
    if not vals:
        return None
    return {"n_points": len(vals), "first": vals[0], "last": vals[-1],
            "min": min(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["ae", "final"], required=True)
    ap.add_argument("--align_dir", required=True)
    ap.add_argument("--v2v_a_workdir", default="")
    ap.add_argument("--v2v_b_workdir", default="")
    args = ap.parse_args()

    A = _load(f"{args.align_dir}/align_A.json")
    B = _load(f"{args.align_dir}/align_B.json")
    L = ["# Joint Sat/Radar Tokenizer — Results",
         "",
         "Design + analysis: `docs/specs/2026-05-18-joint-sat-radar-tokenizer-design.md`",
         "",
         "**Read the spec §2 first** — \"positional correspondence\" was reframed to "
         "*shared latent-code alignment*; the realistic payoff is better flow "
         "*conditioning/convergence*, not added predictive power.",
         ""]

    if "_error" in A or "_error" in B:
        L += ["## ⚠ Tokenizer phase incomplete",
              f"- A: `{A.get('_error', 'ok')}`",
              f"- B: `{B.get('_error', 'ok')}`",
              "", "Training likely still running or failed — check "
              "`/mnt/ssd_2/yghu/Experiments/joint_ae_*_run1/log0.txt` and the "
              "orchestrator log.", ""]
        Path(RESULTS).parent.mkdir(parents=True, exist_ok=True)
        Path(RESULTS).write_text("\n".join(L))
        print("partial results written (alignment json missing)")
        return

    aa, ba = A["alignment"], B["alignment"]
    ar, br = A["reconstruction"], B["reconstruction"]
    d_cos = ba["index_wise_cosine_mean"] - aa["index_wise_cosine_mean"]
    d_cka = ba["linear_cka"] - aa["linear_cka"]
    sat_deg = _pct(br["sat_mse"], ar["sat_mse"])
    rad_deg = _pct(br["radar_mse"], ar["radar_mse"])

    aligned = (d_cos >= 0.10) or (d_cka >= 0.15)
    recon_ok = (sat_deg <= 10.0) and (rad_deg <= 10.0)
    helps_tok = aligned and recon_ok

    L += ["## 1. Tokenizer phase (primary result)", "",
          "| metric | Group A (separate) | Group B (joint) | Δ |",
          "|---|---|---|---|",
          f"| cross-modal cosine | {aa['index_wise_cosine_mean']:.4f} | "
          f"{ba['index_wise_cosine_mean']:.4f} | **{d_cos:+.4f}** |",
          f"| linear CKA | {aa['linear_cka']:.4f} | {ba['linear_cka']:.4f} | "
          f"**{d_cka:+.4f}** |",
          f"| per-index linfit err | {aa['per_index_linfit_norm_err']:.4f} | "
          f"{ba['per_index_linfit_norm_err']:.4f} | "
          f"{ba['per_index_linfit_norm_err']-aa['per_index_linfit_norm_err']:+.4f} |",
          f"| sat recon MSE | {ar['sat_mse']:.6f} | {br['sat_mse']:.6f} | "
          f"{sat_deg:+.1f}% |",
          f"| radar recon MSE | {ar['radar_mse']:.6f} | {br['radar_mse']:.6f} | "
          f"{rad_deg:+.1f}% |",
          f"| sat near-dead tok | {A['capacity']['sat']['near_dead_tokens']} | "
          f"{B['capacity']['sat']['near_dead_tokens']} | |",
          f"| radar near-dead tok | {A['capacity']['radar']['near_dead_tokens']} | "
          f"{B['capacity']['radar']['near_dead_tokens']} | |",
          "",
          "**Decision rule (design §5):** aligned if Δcosine ≥ 0.10 or ΔCKA ≥ "
          "0.15, AND ≤10% recon degradation on *both* modalities.",
          "",
          f"- alignment improved: **{aligned}** (Δcos={d_cos:+.4f}, ΔCKA={d_cka:+.4f})",
          f"- reconstruction preserved: **{recon_ok}** (sat {sat_deg:+.1f}%, "
          f"radar {rad_deg:+.1f}%)",
          "",
          f"### → Joint training helps the tokenizers: **{helps_tok}**", ""]

    if aligned and not recon_ok:
        L += ["**Trade-off detected:** the spaces aligned but reconstruction "
              "degraded. Per the spec, the right follow-up is a sim_weight "
              "sweep (0.1/0.25/0.5/1.0), not a verdict.", ""]
    elif not aligned:
        L += ["The similarity loss did **not** measurably align the latent "
              "spaces at sim_weight=0.5. Either it needs a higher weight, or "
              "the modalities resist index-wise alignment under reconstruction "
              "pressure. Recommend a weight sweep before concluding.", ""]

    if args.phase == "final":
        va = _v2v_loss_tail(args.v2v_a_workdir) if args.v2v_a_workdir else None
        vb = _v2v_loss_tail(args.v2v_b_workdir) if args.v2v_b_workdir else None
        L += ["## 2. v2v pilot (secondary — 8k steps, NOT converged)", ""]
        if va and vb:
            L += [f"- Group A v2v loss: first {va['first']:.4f} → last "
                  f"{va['last']:.4f} (min {va['min']:.4f}, {va['n_points']} pts)",
                  f"- Group B v2v loss: first {vb['first']:.4f} → last "
                  f"{vb['last']:.4f} (min {vb['min']:.4f}, {vb['n_points']} pts)",
                  "",
                  f"Lower/faster-dropping loss for B would support the "
                  f"*conditioning* hypothesis. B last − A last = "
                  f"**{vb['last']-va['last']:+.4f}**. Treat as directional only "
                  f"(single seed, 8k steps).", ""]
        else:
            L += ["v2v loss curves NOT parseable yet — inspect "
                  f"`{args.v2v_a_workdir}` / `{args.v2v_b_workdir}` logs "
                  "manually. The tokenizer-phase result above stands on its "
                  "own.", ""]

    L += ["## 3. What to do next", "",
          "- If *helps tokenizers* = True and v2v shows B converging "
          "faster/lower: proceed to a full-length v2v A/B at the chosen "
          "sim_weight to confirm.",
          "- If alignment up but recon down: run the sim_weight sweep.",
          "- If no alignment: token-index alignment isn't the lever — "
          "revisit (set-level / InfoNCE alternative in spec §3.1) or drop the "
          "idea.",
          "",
          "_Single seed, shared batch order across A/B (intentional for "
          "ablation cleanliness). Pilot step budgets — see spec §3.2/§7._"]

    Path(RESULTS).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULTS).write_text("\n".join(L))
    print(f"[analyze_joint_tok] verdict helps_tok={helps_tok}; wrote {RESULTS}")


if __name__ == "__main__":
    main()
