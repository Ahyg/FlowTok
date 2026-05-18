"""Roll the (reduced, big-budget) sim_weight sweep into results-doc §7.

Cells: w ∈ {0.0 separate, 0.05, 0.25}, AE 25k, v2v 40k. The 8k Group-A/B
pilots are NOT mixed in here (different budget, visually unconverged — see
§2/§6); they stay in §4/§6 as the pilot record.

Two tables:
  7a AE trade-off  — cosine ↑ vs PCA effective rank vs recon MSE
     (+ the tokenizer-ceiling recon panels recon_<tag>.png).
  7b Decoded-radar — the DECISIVE ranking: sat→flow→radar detokenizer,
     dBZ pixel metrics on 2024/07 v2v test.

Defensive: a missing/failed cell degrades to "pending", never crashes.
"""
import json
from pathlib import Path

EXP = Path("/mnt/ssd_2/yghu/Experiments")
ALIGN = EXP / "joint_tok_align"
RES = Path("/mnt/ssd_1/yghu/Code/FlowTok/docs/specs/results/"
           "2026-05-18-joint-tokenizer-results.md")

# (sim_weight, align tag, decoded test dir)
CELLS = [
    (0.00, "w000", EXP / "v2v_jointtok_sweep_w000_run1" / "test_final"),
    (0.05, "w005", EXP / "v2v_jointtok_sweep_w005_run1" / "test_final"),
    (0.25, "w025", EXP / "v2v_jointtok_sweep_w025_run1" / "test_final"),
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
    rk_s0, re_s0 = base["rk_s"], base["re_s"]

    L = ["", "---", "",
         "## 7. sim_weight sweep — reduced set, big budget", "",
         "`w ∈ {0.0 separate, 0.05, 0.25}`, **AE 25k / v2v 40k** (the §6 8k "
         "pilot was visually unconverged — neg R², CSI35=0, low-freq blobs; "
         "those 8k A/B numbers stay in §4/§6 and are *not* mixed into this "
         "table). One-knob ablation, seed 42, shared batch order. "
         "**Ranking = the decoded-radar table 7b** — token-space loss is "
         "deliberately excluded (§4 collapse confound). Tokenizer-ceiling "
         "recon panels (pure encode→decode, no flow): "
         "`joint_tok_align/recon_{w000,w005,w025}.png` (also copied to each "
         "`joint_ae_sweep_*_run1/recon_test.png`) — the decoded radar in 7b "
         "can never beat that ceiling.", ""]

    L += ["### 7a. AE trade-off + tokenizer ceiling", "",
          "| sim_weight | x-modal cosine ↑ | linear CKA | PCA rank sat | "
          "PCA rank radar | sat recon MSE | radar recon MSE | sat recon Δ vs w0 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["have_ae"]:
            L.append(f"| {r['w']:.2f} | _pending_ | | | | | | |"); continue
        dre = (f"{100.0*(r['re_s']-re_s0)/abs(re_s0):+.1f}%"
               if isinstance(r["re_s"], (int, float))
               and isinstance(re_s0, (int, float)) and re_s0 else "—")
        L.append(f"| {r['w']:.2f} | {_f(r['cos'])} | {_f(r['cka'])} | "
                 f"{_f(r['rk_s'],'{:.1f}')} | {_f(r['rk_r'],'{:.1f}')} | "
                 f"{_f(r['re_s'],'{:.6f}')} | {_f(r['re_r'],'{:.6f}')} | {dre} |")

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
        ok = [r for r in rows if isinstance(r["rk_s"], (int, float))
              and isinstance(rk_s0, (int, float)) and rk_s0
              and r["rk_s"] >= 0.9 * rk_s0
              and isinstance(r["re_s"], (int, float))
              and isinstance(re_s0, (int, float)) and re_s0
              and (r["re_s"] - re_s0) / abs(re_s0) <= 0.10]
        no_collapse = max(ok, key=lambda r: r["w"]) if ok else None
        L += [f"**Decoded-radar winner (min mse_dbz): sim_weight = "
              f"{best['w']:.2f}** (mse_dbz {best['mse']:.4f}, "
              f"ssim {_f(best['ssim'])}, r² {_f(best['r2'])}).", ""]
        if no_collapse is not None:
            same = no_collapse["w"] == best["w"]
            L += [f"**§5 non-collapse operating point:** largest w keeping sat "
                  f"PCA rank ≥ 0.9·w0 and sat recon ≤ +10% is "
                  f"**w = {no_collapse['w']:.2f}**. "
                  + ("Coincides with the decoded-radar winner — clean."
                     if same else
                     "Differs from the decoded winner: trust 7b (the stated "
                     "arbiter); 7a only explains *why*."), ""]
        else:
            L += ["**§5 non-collapse operating point:** no swept w>0 holds PCA "
                  "rank ≥ 0.9·w0 with recon ≤ +10% — even 0.05 collapses the "
                  "latent; the useful window (if any) is below 0.05.", ""]
        if best["w"] == 0.0:
            L += ["→ **No positive sim_weight beats the separate baseline on "
                  "decoded radar, even at the big budget.** Index-wise cosine "
                  "alignment costs more reconstruction than it returns in flow "
                  "conditioning. Recommend dropping index-wise alignment and "
                  "revisiting the set-level / InfoNCE alternative (spec §3.1).",
                  ""]
        else:
            L += [f"→ **A positive sim_weight ({best['w']:.2f}) beats the "
                  "separate baseline on decoded radar at the big budget.** "
                  "The idea works below the collapse threshold; promote "
                  f"w={best['w']:.2f} to the main config and confirm on a "
                  "held-out month / second seed.", ""]
        # Convergence sanity — at big budget the models should beat mean (r²>0)
        # and show some convective skill; flag if not.
        if all(isinstance(r["r2"], (int, float)) and r["r2"] < 0 for r in dec):
            L += ["⚠ **Still under-converged:** every cell has r² < 0 (worse "
                  "than predicting the mean) even at 40k v2v. The A/B ranking "
                  "is internally consistent but the absolute model is not yet "
                  "usable — compare the 7a recon panels: if the *tokenizer* "
                  "recon is already poor, the ceiling (not the flow budget) is "
                  "the blocker and the next lever is a larger tokenizer.", ""]
    else:
        L += ["_Sweep running — 7b fills in per cell (~3.3 h/cell: AE 25k + "
              "v2v 40k + decode)._", ""]

    txt = RES.read_text()
    marker = "## 7. sim_weight sweep"
    if marker in txt:
        txt = txt.split("\n---\n\n" + marker)[0].rstrip()
    RES.write_text(txt + "\n" + "\n".join(L) + "\n")
    print(f"[analyze_jointtok_sweep] §7 ({len(dec)}/{len(CELLS)} decoded)")


if __name__ == "__main__":
    main()
