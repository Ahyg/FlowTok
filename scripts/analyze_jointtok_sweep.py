"""Roll the sim_weight sweep into results-doc §7.

Two tables:
  (a) AE trade-off curve  — cosine ↑ vs PCA effective rank vs recon MSE
      (the §5 question: largest w that doesn't collapse the latent).
  (b) Decoded-radar table — the DECISIVE ranking the user asked for:
      sat→flow→radar detokenizer, dBZ pixel metrics on 2024/07 v2v test.

Anchors w=0 (Group A) and w=0.5 (Group B) are reused from the existing
joint_tok_align/align_{A,B}.json + v2v_jointtok_{A,B}_run1/test8000/metrics.json.
Defensive: a missing/failed cell degrades to "pending", never crashes.
"""
import json
from pathlib import Path

EXP = Path("/mnt/ssd_2/yghu/Experiments")
ALIGN = EXP / "joint_tok_align"
RES = Path("/mnt/ssd_1/yghu/Code/FlowTok/docs/specs/results/"
           "2026-05-18-joint-tokenizer-results.md")

# (w, align tag, decoded test8000 dir) — A/B reused as the 0.0 / 0.5 anchors.
CELLS = [
    (0.00, "A",    EXP / "v2v_jointtok_A_run1"        / "test8000"),
    (0.02, "w002", EXP / "v2v_jointtok_sweep_w002_run1" / "test8000"),
    (0.05, "w005", EXP / "v2v_jointtok_sweep_w005_run1" / "test8000"),
    (0.10, "w010", EXP / "v2v_jointtok_sweep_w010_run1" / "test8000"),
    (0.25, "w025", EXP / "v2v_jointtok_sweep_w025_run1" / "test8000"),
    (0.50, "B",    EXP / "v2v_jointtok_B_run1"        / "test8000"),
]


def _jload(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:                                       # noqa: BLE001
        return None


def _g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _f(v, fmt="{:.4f}"):
    return fmt.format(v) if isinstance(v, (int, float)) else "—"


def main():
    rows = []
    for w, tag, tdir in CELLS:
        a = _jload(ALIGN / f"align_{tag}.json")
        m = _jload(tdir / "metrics.json")
        rows.append({
            "w": w, "tag": tag,
            "cos":  _g(a, "alignment", "index_wise_cosine_mean"),
            "cka":  _g(a, "alignment", "linear_cka"),
            "rk_s": _g(a, "capacity", "sat",   "pca_effective_rank"),
            "rk_r": _g(a, "capacity", "radar", "pca_effective_rank"),
            "re_s": _g(a, "reconstruction", "sat_mse"),
            "re_r": _g(a, "reconstruction", "radar_mse"),
            "mse":  _g(m, "mse_dbz"),
            "rmse": _g(m, "rmse_dbz"),
            "ssim": _g(m, "ssim"),
            "r2":   _g(m, "r2"),
            "fss":  _g(m, "avg_fss"),
            "csi":  _g(m, "csi35"),
            "have_ae": a is not None, "have_dec": m is not None,
        })

    base = next(r for r in rows if r["w"] == 0.0)
    rk_s0, rk_r0, re_s0 = base["rk_s"], base["rk_r"], base["re_s"]

    L = ["", "---", "",
         "## 7. sim_weight sweep `{0, 0.02, 0.05, 0.1, 0.25}` (+0.5 anchor)", "",
         "Same one-knob ablation, same budgets as §1/§6 (AE 15k, v2v 8k, seed 42, "
         "shared batch order). w=0 reuses Group A, w=0.5 reuses Group B. "
         "**Final ranking is the decoded-radar table — token-space loss is "
         "deliberately not used here (see §4).**", ""]

    # (a) AE trade-off curve
    L += ["### 7a. AE trade-off (the §5 collapse question)", "",
          "| sim_weight | x-modal cosine ↑ | linear CKA | PCA rank sat | "
          "PCA rank radar | sat recon MSE | radar recon MSE | sat recon Δ vs w0 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["have_ae"]:
            L.append(f"| {r['w']:.2f} | _pending_ | | | | | | |"); continue
        dre = (f"{100.0*(r['re_s']-re_s0)/abs(re_s0):+.1f}%"
               if isinstance(r["re_s"], (int, float)) and re_s0 else "—")
        L.append(f"| {r['w']:.2f} | {_f(r['cos'])} | {_f(r['cka'])} | "
                 f"{_f(r['rk_s'],'{:.1f}')} | {_f(r['rk_r'],'{:.1f}')} | "
                 f"{_f(r['re_s'],'{:.6f}')} | {_f(r['re_r'],'{:.6f}')} | {dre} |")

    # (b) decoded-radar — decisive
    L += ["",
          "### 7b. Decoded-radar pixel-space (DECISIVE — 2024/07 v2v test)", "",
          "| sim_weight | mse_dbz ↓ | rmse_dbz ↓ | ssim ↑ | r² ↑ | avg_fss ↑ | "
          "csi35 ↑ |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["have_dec"]:
            L.append(f"| {r['w']:.2f} | _pending_ | | | | | |"); continue
        L.append(f"| {r['w']:.2f} | {_f(r['mse'])} | {_f(r['rmse'])} | "
                 f"{_f(r['ssim'])} | {_f(r['r2'])} | {_f(r['fss'])} | "
                 f"{_f(r['csi'])} |")

    dec = [r for r in rows if isinstance(r["mse"], (int, float))]
    L += [""]
    if len(dec) >= 2:
        best = min(dec, key=lambda r: r["mse"])
        # §5 goal: largest w whose sat PCA rank stays >= 0.9*w0 AND sat recon
        # degradation <= 10%, among cells that didn't collapse.
        ok = [r for r in rows if isinstance(r["rk_s"], (int, float))
              and isinstance(rk_s0, (int, float)) and rk_s0
              and r["rk_s"] >= 0.9 * rk_s0
              and isinstance(r["re_s"], (int, float)) and re_s0
              and (r["re_s"] - re_s0) / abs(re_s0) <= 0.10]
        no_collapse = max(ok, key=lambda r: r["w"]) if ok else None
        L += [f"**Decoded-radar winner (min mse_dbz): sim_weight = "
              f"{best['w']:.2f}** (mse_dbz {best['mse']:.4f}, "
              f"ssim {_f(best['ssim'])}, r² {_f(best['r2'])}).", ""]
        if no_collapse is not None:
            L += [f"**§5 non-collapse operating point:** largest w keeping sat "
                  f"PCA rank ≥ 0.9·w0 and sat recon ≤ +10% is "
                  f"**w = {no_collapse['w']:.2f}**. "
                  + ("Coincides with the decoded-radar winner — clean result."
                     if no_collapse['w'] == best['w'] else
                     "Differs from the decoded-radar winner: trust the decoded "
                     "table (the user's stated arbiter); the AE proxy only "
                     "explains *why*."), ""]
        else:
            L += ["**§5 non-collapse operating point:** no swept w holds PCA "
                  "rank ≥ 0.9·w0 with recon ≤ +10% — every positive weight "
                  "tested collapses the latent to some degree; the useful "
                  "window (if any) is below 0.02.", ""]
        if best["w"] == 0.0:
            L += ["→ **No positive sim_weight beats the separate baseline on "
                  "decoded radar.** Index-wise cosine alignment, at every "
                  "weight tested, costs more reconstruction than it returns in "
                  "flow conditioning. Recommend dropping index-wise alignment "
                  "and revisiting the set-level / InfoNCE alternative (spec "
                  "§3.1) before further sweeping.", ""]
        else:
            L += [f"→ **A positive sim_weight ({best['w']:.2f}) beats the "
                  "separate baseline on decoded radar.** The idea works below "
                  "the collapse threshold; next step is a longer-budget v2v "
                  f"A/B at w={best['w']:.2f} vs w=0 to confirm at convergence.",
                  ""]
    else:
        L += ["_Sweep still running — decoded-radar table fills in per cell._",
              ""]

    txt = RES.read_text()
    marker = "## 7. sim_weight sweep"
    if marker in txt:
        txt = txt.split("\n---\n\n" + marker)[0].rstrip()
    RES.write_text(txt + "\n" + "\n".join(L) + "\n")
    print(f"[analyze_jointtok_sweep] wrote §7 ({len(dec)}/{len(CELLS)} decoded)")


if __name__ == "__main__":
    main()
