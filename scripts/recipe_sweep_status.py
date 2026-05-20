#!/usr/bin/env python3
"""Morning status for the AE loss-recipe sweep — parses each cell's training.log.

For every cell: max step reached, latest + best val reconstruction_loss (the
overnight signal; full eval via scripts/test_flowtitok_ae.py is the follow-on),
rFID, and whether it finished. Cells grouped by modality, deltas vs the b0
baseline at each cell's best step. No deps beyond stdlib.
"""
import re
from pathlib import Path

EXP = Path("/mnt/ssd_2/yghu/Experiments/ae_recipe_sweep")
CELLS = ["b0", "s1_kl05", "s1_kl10", "s1_pc", "s1_p06", "s1_p16"]
STEP_RE = re.compile(r"EVALUATION Step:\s*(\d+)")
RECON_RE = re.compile(r"'reconstruction_loss':\s*([0-9.eE+-]+)")
RFID_RE = re.compile(r"'rFID':\s*([0-9.eE+-]+)")
BEST_RE = re.compile(r"New best val L2 = ([0-9.eE+-]+) at step (\d+)")
TRAINSTEP_RE = re.compile(r"Step:\s*(\d+)\s+Total Loss")


def parse(log: Path):
    if not log.exists():
        return None
    txt = log.read_text(errors="ignore")
    evals = []  # (step, recon, rfid)
    steps = STEP_RE.findall(txt)
    recons = RECON_RE.findall(txt)
    rfids = RFID_RE.findall(txt)
    for s, r, f in zip(steps, recons, rfids):
        evals.append((int(s), float(r), float(f)))
    best = BEST_RE.findall(txt)
    best_l2, best_step = (float(best[-1][0]), int(best[-1][1])) if best else (None, None)
    tsteps = TRAINSTEP_RE.findall(txt)
    cur_step = int(tsteps[-1]) if tsteps else (evals[-1][0] if evals else 0)
    finished = "Finishing training" in txt
    return dict(evals=evals, best_l2=best_l2, best_step=best_step,
                cur_step=cur_step, finished=finished)


def main():
    for mod in ("radar", "s10"):
        print(f"\n=== {mod} ===")
        print(f"{'cell':10s} {'step':>6s} {'fin':>3s} {'best_val_L2':>12s} "
              f"{'@step':>6s} {'last_val_L2':>12s} {'last_rFID':>9s} {'Δ%vs b0':>8s}")
        base = parse(EXP / f"{mod}_b0" / "training.log")
        base_best = base["best_l2"] if base and base["best_l2"] else None
        for c in CELLS:
            d = parse(EXP / f"{mod}_{c}" / "training.log")
            if d is None:
                print(f"{c:10s} {'--':>6s} (not started)")
                continue
            lv = d["evals"][-1][1] if d["evals"] else float("nan")
            lf = d["evals"][-1][2] if d["evals"] else float("nan")
            delta = ""
            if base_best and d["best_l2"] and c != "b0":
                delta = f"{100*(d['best_l2']-base_best)/base_best:+.1f}"
            bl = f"{d['best_l2']:.6f}" if d["best_l2"] else "n/a"
            print(f"{c:10s} {d['cur_step']:>6d} {'Y' if d['finished'] else '·':>3s} "
                  f"{bl:>12s} {str(d['best_step'] or '-'):>6s} "
                  f"{lv:>12.6f} {lf:>9.1f} {delta:>8s}")
    print("\nLower val_L2 = better reconstruction. Δ% is best-val vs b0 baseline "
          "(negative = better than run1-recipe baseline).")
    print("Full fidelity/blur/lightning eval (test set + freq-edge + FAR + lgt-rich) "
          "is the follow-on per spec §7-8.")


if __name__ == "__main__":
    main()
