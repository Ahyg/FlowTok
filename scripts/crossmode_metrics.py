#!/usr/bin/env python3
"""Post-hoc CROSS-MODE generation/temporal metrics for the sat->radar benchmark.

Goal: give i2i and v2v models the *identical* full metric set so every metric is
directly comparable.

Native test scripts compute (per mode):
  i2i  -> FID / sFID / KID   (InceptionV3, per-frame image distribution)
  v2v  -> FVD / KVD          (I3D-Kinetics, clip distribution)
       +  tc_dbz_sq          (RAFT optical-flow temporal consistency)

This script fills the cross-mode gap by reusing the *same* `_generation_metrics`
module on the dumped fp32-dBZ prediction arrays (no model re-run):
  * i2i holdout -> ADD fvd / kvd / tc_dbz_sq / tc_count_pairs
                   (regroup the per-frame preds into 16-frame clips; the dumped
                    frames are in filelist order == v2v clip order, verified)
  * v2v holdout -> ADD fid / sfid / kid
                   (flatten clips into per-frame images)

GT is canonical (clip(channel-11, 0, 60) dBZ, filelist order) and shared across
ALL models at a given (mode, cond) -> the FID/FVD "real" distribution is identical,
so cross-project (FlowTok vs Diffi2i) numbers are comparable.

Usage:
  # 1) one-time GT cache per (mode, cond)  [I/O heavy -> run in a job]
  python crossmode_metrics.py --mode i2i --cond ct --gt-cache <gt.npy> --build-gt-only
  # 2) per-holdout cross-mode compute (needs the cache from step 1)
  python crossmode_metrics.py --mode i2i --cond ct --gt-cache <gt.npy> \
      --pred <pred_dbz.npy> --json <holdout.json> --device cuda
"""
import argparse, json, os, pickle, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FLDIR = "/g/data/kl02/yh0308/Data/71/filelists"
DBZ_MAX = 60.0
RADAR_CH = 11
T_CLIP = 16


def filelist_path(mode, cond):
    if cond == "ct":
        return f"{FLDIR}/dataset_filelist_{mode}_test_202407_202507_cond3nan1_clip16_p005_seed42.pkl"
    if cond == "nofilt":
        return f"{FLDIR}/dataset_filelist_{mode}_test_202407_202507_nofilter_nan1_clip16.pkl"
    raise ValueError(cond)


def target_paths_in_order(mode, cond):
    """All radar target .npy paths for the test split, in dataloader (== dump) order."""
    # First-party filelist (built by this repo's build pipeline under /g/data; the same
    # pkl the test scripts load). Trusted source -> pickle.load is safe here.
    te = pickle.load(open(filelist_path(mode, cond), "rb"))[2]
    paths = []
    for item in te:                       # item = (inputs, targets); targets = list of paths
        tgt = item[1]
        paths.extend(str(p) for p in tgt)  # i2i: 1 path; v2v: 16 paths (clip, in order)
    return paths


def build_gt(mode, cond, cache):
    """Canonical GT dBZ array [M, H, W] in [0,60], filelist order. Cached to disk."""
    lock = cache + ".lock"
    if os.path.exists(cache):
        return np.load(cache, mmap_mode="r")
    # crude cross-process lock to avoid two jobs building the same cache
    for _ in range(600):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd); break
        except FileExistsError:
            if os.path.exists(cache):
                return np.load(cache, mmap_mode="r")
            time.sleep(2)
    try:
        if os.path.exists(cache):
            return np.load(cache, mmap_mode="r")
        paths = target_paths_in_order(mode, cond)
        M = len(paths)
        gt = np.empty((M, 128, 128), np.float32)
        for i, p in enumerate(paths):
            a = np.load(p)                                  # (12,128,128) raw
            gt[i] = np.clip(np.nan_to_num(a[RADAR_CH], nan=0.0), 0.0, DBZ_MAX)
            if i % 4000 == 0:
                print(f"[gt] {mode}/{cond} {i}/{M}", flush=True)
        tmp = cache + ".tmp.npy"
        np.save(tmp, gt); os.replace(tmp, cache)
        print(f"[gt] wrote {cache}  shape={gt.shape}", flush=True)
        return gt
    finally:
        try: os.remove(lock)
        except OSError: pass


def _to01(arr):
    import torch
    return torch.from_numpy((np.asarray(arr, np.float32) / DBZ_MAX)).clamp_(0, 1)


def add_clip_metrics(pred_dbz, gt_dbz, device, clip_batch=8):
    """i2i -> FVD / KVD / tc (regroup frames into clips of 16)."""
    import torch
    import _generation_metrics as gm
    M = (len(pred_dbz) // T_CLIP) * T_CLIP
    H, W = pred_dbz.shape[-2:]
    n = M // T_CLIP
    prc = _to01(pred_dbz[:M]).reshape(n, T_CLIP, 1, H, W)
    gtc = _to01(gt_dbz[:M]).reshape(n, T_CLIP, 1, H, W)
    mets = gm.make_v2v_metrics(device)                      # {fvd, kvd, _i3d}
    tc = gm.TemporalConsistency(device, dbz_scale=DBZ_MAX)
    for s in range(0, n, clip_batch):
        gtb = gtc[s:s + clip_batch].to(device)
        prb = prc[s:s + clip_batch].to(device)
        gt3 = gm.radar_to_3ch(gtb); pr3 = gm.radar_to_3ch(prb)
        for k in ("fvd", "kvd"):
            if k in mets:
                mets[k].update(gt3, real=True); mets[k].update(pr3, real=False)
        tc.update(gtb, prb, valid_mask=None)
    out = {}
    for k in ("fvd", "kvd"):
        if k in mets:
            r = mets[k].compute()
            out[k] = ({"mean": float(r[0].item()), "std": float(r[1].item())}
                      if isinstance(r, (tuple, list)) else float(r.item()))
    out["tc_dbz_sq"] = tc.compute(); out["tc_count_pairs"] = int(tc.tc_count)
    return out


def gt_self_tc(gt_dbz, device, clip_batch=8):
    """GT self temporal-consistency: warp GT_t by GT flow, compare to GT_{t+1}.
    The 'irreducible' TC reference (flow-estimation error + real growth/decay).
    A prediction with LOWER tc than this is smoother than reality (over-smoothed)."""
    import _generation_metrics as gm
    M = (len(gt_dbz) // T_CLIP) * T_CLIP
    H, W = gt_dbz.shape[-2:]
    n = M // T_CLIP
    gtc = _to01(gt_dbz[:M]).reshape(n, T_CLIP, 1, H, W)
    tc = gm.TemporalConsistency(device, dbz_scale=DBZ_MAX)
    for s in range(0, n, clip_batch):
        gtb = gtc[s:s + clip_batch].to(device)
        tc.update(gtb, gtb, valid_mask=None)            # pred == gt
    return {"tc_dbz_sq": tc.compute(), "tc_count_pairs": int(tc.tc_count)}


def add_frame_metrics(pred_dbz, gt_dbz, device, frame_batch=256):
    """v2v -> FID / sFID / KID (flatten clips into per-frame images)."""
    import torch
    import _generation_metrics as gm
    H, W = pred_dbz.shape[-2:]
    pr = _to01(np.asarray(pred_dbz).reshape(-1, 1, H, W))
    gt = _to01(np.asarray(gt_dbz).reshape(-1, 1, H, W))
    M = min(len(pr), len(gt)); pr = pr[:M]; gt = gt[:M]
    mets = gm.make_i2i_metrics(device)                      # {fid, sfid, kid}
    for s in range(0, M, frame_batch):
        gt3 = gm.radar_to_3ch(gt[s:s + frame_batch].to(device))
        pr3 = gm.radar_to_3ch(pr[s:s + frame_batch].to(device))
        for k in ("fid", "sfid", "kid"):
            if k in mets:
                mets[k].update(gt3, real=True); mets[k].update(pr3, real=False)
    out = {}
    for k in ("fid", "sfid", "kid"):
        if k in mets:
            r = mets[k].compute()
            out[k] = ({"mean": float(r[0].item()), "std": float(r[1].item())}
                      if isinstance(r, (tuple, list)) else float(r.item()))
    return out


def merge_json(json_path, added, mode):
    with open(json_path) as f:
        d = json.load(f)
    gen = d.get("gen_metrics") or {}
    gen.update(added)
    prov = sorted(added.keys())
    note = gen.get("_crossmode_added", [])
    gen["_crossmode_added"] = sorted(set(note) | set(prov))
    d["gen_metrics"] = gen
    tmp = json_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, json_path)
    print(f"[merge] {json_path}  +{prov}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["i2i", "v2v"])
    ap.add_argument("--cond", required=True, choices=["ct", "nofilt"])
    ap.add_argument("--gt-cache", required=True)
    ap.add_argument("--pred", help="pred_dbz.npy (omit with --build-gt-only)")
    ap.add_argument("--json", help="holdout metrics json to merge into")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip-batch", type=int, default=8)
    ap.add_argument("--frame-batch", type=int, default=256)
    ap.add_argument("--build-gt-only", action="store_true")
    ap.add_argument("--gt-tc-baseline", help="compute GT self-TC and write to this json, then exit")
    a = ap.parse_args()

    gt = build_gt(a.mode, a.cond, a.gt_cache)
    print(f"[gt] {a.mode}/{a.cond} ready: shape={gt.shape}", flush=True)
    if a.build_gt_only:
        return
    if a.gt_tc_baseline:
        import torch
        dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
        gt = np.asarray(gt).reshape(-1, gt.shape[-2], gt.shape[-1])
        base = gt_self_tc(gt, dev, a.clip_batch)
        with open(a.gt_tc_baseline, "w") as f:
            json.dump({"mode": a.mode, "cond": a.cond, **base}, f, indent=2)
        print(f"[gt-tc-baseline] {a.cond}: {base} -> {a.gt_tc_baseline}", flush=True)
        return
    assert a.pred and a.json, "need --pred and --json (or --build-gt-only)"
    pred = np.load(a.pred, mmap_mode="r")
    print(f"[pred] {a.pred} shape={pred.shape}", flush=True)
    # pred dumped as frames [M,H,W] (i2i) or clips flattened to frames; normalize to [M,H,W]
    pred = np.asarray(pred).reshape(-1, pred.shape[-2], pred.shape[-1])
    gt = np.asarray(gt).reshape(-1, gt.shape[-2], gt.shape[-1])
    M = min(len(pred), len(gt))
    if len(pred) != len(gt):
        print(f"[WARN] pred frames {len(pred)} != gt frames {len(gt)}; truncating to {M}")
    pred = pred[:M]; gt = gt[:M]
    import torch
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    if a.mode == "i2i":
        added = add_clip_metrics(pred, gt, dev, a.clip_batch)     # fvd/kvd/tc
    else:
        added = add_frame_metrics(pred, gt, dev, a.frame_batch)   # fid/sfid/kid
    print(f"[compute] {a.mode}/{a.cond} added={added} ({time.time()-t0:.0f}s)", flush=True)
    merge_json(a.json, added, a.mode)


if __name__ == "__main__":
    main()
