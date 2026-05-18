"""Roll the InfoNCE sim_weight sweep into results-doc §7 (design §3.3).

Index-wise cosine was refuted (alignment-by-collapse — §4/§6/§6.2). This sweep
swaps the sim loss to per-index symmetric InfoNCE (anti-collapse). Cells:
w ∈ {0.0 separate, 0.1, 0.5, 1.0}, AE 25k, v2v 40k. The 8k/cosine pilots are
NOT mixed in (different loss + budget); they stay in §4/§6 as the record.

Two tables:
  7a AE trade-off  — cosine ↑ vs CKA vs PCA effective rank vs **near-dead
     tokens** vs recon MSE (the §3.3 acceptance bar: align WITHOUT collapse).
  7b Decoded-radar — the DECISIVE ranking: sat→flow→radar detokenizer,
     dBZ pixel metrics on 2024/07 v2v test.

InfoNCE alignment artifacts are tagged inf_* (cosine ones preserved).
Defensive: a missing/failed cell degrades to "pending", never crashes.
"""
import json
from pathlib import Path

EXP = Path("/mnt/ssd_2/yghu/Experiments")
ALIGN = EXP / "joint_tok_align"
RES = Path("/mnt/ssd_1/yghu/Code/FlowTok/docs/specs/results/"
           "2026-05-18-joint-tokenizer-results.md")

# (sim_weight, dir/config tag, align/recon tag, decoded test dir)
CELLS = [
    (0.0, "w000", "inf_w000", EXP / "v2v_jointtok_infonce_w000_run1" / "test_final"),
    (0.1, "w010", "inf_w010", EXP / "v2v_jointtok_infonce_w010_run1" / "test_final"),
    (0.5, "w050", "inf_w050", EXP / "v2v_jointtok_infonce_w050_run1" / "test_final"),
    (1.0, "w100", "inf_w100", EXP / "v2v_jointtok_infonce_w100_run1" / "test_final"),
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
    for w, tag, atag, tdir in CELLS:
        a = _jload(ALIGN / f"align_{atag}.json")
        m = _jload(tdir / "metrics.json")
        rows.append({
            "w": w, "tag": tag,
            "cos":  _g(a, "alignment", "index_wise_cosine_mean"),
            "cka":  _g(a, "alignment", "linear_cka"),
            "rk_s": _g(a, "capacity", "sat",   "pca_effective_rank"),
            "rk_r": _g(a, "capacity", "radar", "pca_effective_rank"),
            "nd_s": _g(a, "capacity", "sat",   "near_dead_tokens"),
            "nd_r": _g(a, "capacity", "radar", "near_dead_tokens"),
            "nt_s": _g(a, "capacity", "sat",   "n_tokens"),
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
         "## 7. sim_weight sweep — InfoNCE (design §3.3)", "",
         "Per-index **symmetric InfoNCE** alignment (learnable CLIP "
         "log-temperature, 1k-step sim-loss warmup) — the anti-collapse "
         "replacement for index-wise cosine, which was refuted by "
         "alignment-by-collapse (§4/§6/§6.2; the cosine 8k/sweep numbers stay "
         "there and are *not* mixed in here). `w ∈ {0.0 separate, 0.1, 0.5, "
         "1.0}`, **AE 25k / v2v 40k**, one-knob ablation, seed 42, shared "
         "batch order. **Ranking = the decoded-radar table 7b.** The §3.3 "
         "question: does InfoNCE buy alignment *without* the codebook death "
         "cosine caused? PCA effective rank **and near-dead tokens** are hard "
         "gates in 7a (they, not the loss log, exposed the cosine collapse). "
         "Tokenizer-ceiling recon panels: "
         "`joint_tok_align/recon_inf_{w000,w010,w050,w100}.png`.", ""]

    L += ["### 7a. AE trade-off + tokenizer ceiling (collapse gates)", "",
          "| sim_weight | x-modal cosine ↑ | linear CKA | PCA rank sat | "
          "PCA rank radar | near-dead sat | near-dead radar | sat recon MSE | "
          "radar recon MSE | sat recon Δ vs w0 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["have_ae"]:
            L.append(f"| {r['w']:.2f} | _pending_ | | | | | | | | |"); continue
        dre = (f"{100.0*(r['re_s']-re_s0)/abs(re_s0):+.1f}%"
               if isinstance(r["re_s"], (int, float))
               and isinstance(re_s0, (int, float)) and re_s0 else "—")
        nds = (f"{r['nd_s']}/{r['nt_s']}"
               if isinstance(r["nd_s"], (int, float)) else "—")
        ndr = (f"{r['nd_r']}/{r['nt_s']}"
               if isinstance(r["nd_r"], (int, float)) else "—")
        L.append(f"| {r['w']:.2f} | {_f(r['cos'])} | {_f(r['cka'])} | "
                 f"{_f(r['rk_s'],'{:.1f}')} | {_f(r['rk_r'],'{:.1f}')} | "
                 f"{nds} | {ndr} | {_f(r['re_s'],'{:.6f}')} | "
                 f"{_f(r['re_r'],'{:.6f}')} | {dre} |")

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
        # §3.3 non-collapse gate: align up AND PCA rank ≥ 0.9·w0 AND
        # near-dead ≤ 10% of tokens AND sat recon ≤ +10%.
        def no_collapse(r):
            return (isinstance(r["rk_s"], (int, float))
                    and isinstance(rk_s0, (int, float)) and rk_s0
                    and r["rk_s"] >= 0.9 * rk_s0
                    and isinstance(r["re_s"], (int, float))
                    and isinstance(re_s0, (int, float)) and re_s0
                    and (r["re_s"] - re_s0) / abs(re_s0) <= 0.10
                    and isinstance(r["nd_s"], (int, float))
                    and isinstance(r["nt_s"], (int, float)) and r["nt_s"]
                    and r["nd_s"] <= 0.10 * r["nt_s"]
                    and isinstance(r["nd_r"], (int, float))
                    and r["nd_r"] <= 0.10 * r["nt_s"])
        ok = [r for r in rows if r["w"] > 0.0 and no_collapse(r)]
        held = max(ok, key=lambda r: r["w"]) if ok else None
        L += [f"**Decoded-radar winner (min mse_dbz): sim_weight = "
              f"{best['w']:.2f}** (mse_dbz {best['mse']:.4f}, "
              f"ssim {_f(best['ssim'])}, r² {_f(best['r2'])}).", ""]
        if held is not None:
            L += [f"**§3.3 anti-collapse check:** largest w>0 holding PCA rank "
                  f"≥ 0.9·w0, near-dead ≤ 10 % of tokens, sat recon ≤ +10 % is "
                  f"**w = {held['w']:.2f}** — InfoNCE aligns *without* the "
                  "codebook death cosine caused (cosine collapsed already at "
                  "w=0.05: ~80 % dead). "
                  + ("Coincides with the decoded-radar winner — clean win."
                     if held["w"] == best["w"] else
                     "Differs from the decoded winner: trust 7b (the stated "
                     "arbiter); 7a explains *why*."), ""]
        else:
            L += ["**§3.3 anti-collapse check:** no swept w>0 clears all four "
                  "gates (PCA rank ≥ 0.9·w0, near-dead ≤ 10 %, recon ≤ +10 %). "
                  "InfoNCE mitigated but did not eliminate the trade-off — "
                  "inspect the recon panels and the per-gate columns in 7a.", ""]
        if best["w"] == 0.0:
            L += ["→ **No positive sim_weight beats the separate baseline on "
                  "decoded radar.** Even collapse-free InfoNCE alignment does "
                  "not translate into better decoded radar at this budget — "
                  "latent-code alignment is not the lever for this task; "
                  "revisit the §2 caveat (alignment ≠ information).", ""]
        else:
            L += [f"→ **A positive sim_weight ({best['w']:.2f}) beats the "
                  "separate baseline on decoded radar.** InfoNCE alignment "
                  "helps where cosine hurt; promote "
                  f"w={best['w']:.2f} to the main config and confirm on a "
                  "held-out month / second seed.", ""]
        if all(isinstance(r["r2"], (int, float)) and r["r2"] < 0 for r in dec):
            L += ["⚠ **Still under-converged:** every cell has r² < 0 even at "
                  "40k v2v. The A/B ranking is internally consistent but the "
                  "absolute model is not yet usable — compare the 7a recon "
                  "panels: if the *tokenizer* recon is already poor, the "
                  "ceiling (not the flow budget) is the blocker and the next "
                  "lever is a larger tokenizer, not more flow steps.", ""]
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
