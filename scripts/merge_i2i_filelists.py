#!/usr/bin/env python3
"""Merge per-split filelists into one (train, val, test) filelist.

Mirrors the logic of merge_dataset_v2v_cpu_gadi.sh: takes the train slot from
--train, the val slot from --val, the test slot from --test, writes the 3-slot
tuple to --out. `.pkl` is this repo's pre-existing filelist format (produced/
consumed only by its own build_dataset.py / dataset loader; not external input).
"""
import argparse
import pickle


def _slot(path, idx):
    with open(path, "rb") as f:
        return pickle.load(f)[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    tr, va, te = _slot(a.train, 0), _slot(a.val, 1), _slot(a.test, 2)
    with open(a.out, "wb") as f:
        pickle.dump((tr, va, te), f)
    print(f"merged train={len(tr)} val={len(va)} test={len(te)} -> {a.out}")


if __name__ == "__main__":
    main()
