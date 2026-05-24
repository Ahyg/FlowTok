"""Filter the full v2v train pkl to the summer-2021 date range used by the i2i
M-series, producing a comparable v2v small-set filelist. A v2v clip is kept iff
ALL its frame timestamps fall within [DMIN, DMAX]."""
import pickle, re, argparse

def clip_paths(entry):
    p = entry[0] if isinstance(entry, (list, tuple)) else entry
    return p if isinstance(p, (list, tuple)) else [p]

def all_dates(entry):
    ds = []
    for fp in clip_paths(entry):
        m = re.search(r'(20\d{6})', str(fp))
        if m:
            ds.append(int(m.group(1)))
    return ds

def keep(entry, dmin, dmax):
    ds = all_dates(entry)
    return bool(ds) and all(dmin <= d <= dmax for d in ds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_train_201906_202312.pkl")
    ap.add_argument("--out", default="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_baseline_2021summer.pkl")
    ap.add_argument("--dmin", type=int, required=True)
    ap.add_argument("--dmax", type=int, required=True)
    args = ap.parse_args()
    with open(args.src, "rb") as f:
        tr, va, te = pickle.load(f)
    tr2 = [e for e in tr if keep(e, args.dmin, args.dmax)]
    va2 = [e for e in va if keep(e, args.dmin, args.dmax)]
    te2 = [e for e in te if keep(e, args.dmin, args.dmax)]
    assert len(tr2) > 0, "no train clips in range -- check path date regex / range"
    with open(args.out, "wb") as f:
        pickle.dump((tr2, va2, te2), f)
    print(f"wrote {args.out}: train={len(tr2)} val={len(va2)} test={len(te2)} (range {args.dmin}-{args.dmax})")

if __name__ == "__main__":
    main()
