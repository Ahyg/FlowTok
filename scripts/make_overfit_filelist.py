#!/usr/bin/env python3
"""Build a tiny N-sample overfit filelist from a source train pkl.

Works for both i2i (each entry = ([path], [path])) and v2v (each entry = (path_list_T,
path_list_T)) — just slices the first N entries from train and writes them to a new pkl.

The output preserves the (train, val, test) tuple structure used by DatasetBuilder.

Examples
--------

# i2i (single-frame samples):
python make_overfit_filelist.py \
    --src /g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_train_201906_202312_ct005.pkl \
    --out /scratch/.../my-overfit/models/dataset_filelist.pkl \
    --n 32

# v2v (16-frame clips):
python make_overfit_filelist.py \
    --src /g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_v2v_train_201906_202312_halfvalid50_ct005.pkl \
    --out /scratch/.../my-v2v-overfit/models/dataset_filelist.pkl \
    --n 32

# overfit-test (after the 32-sample model trains, test it on the same 32 samples):
# put the 32 train samples into the test slot.
python make_overfit_filelist.py \
    --src dataset_filelist.pkl \
    --out dataset_filelist_overfittest.pkl \
    --n 32 \
    --put-into test
"""
import argparse
import os
import pickle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source train pkl (tuple of train/val/test lists).")
    p.add_argument("--out", required=True, help="Output pkl path.")
    p.add_argument("--n", type=int, default=32, help="How many samples/clips to include.")
    p.add_argument("--put-into", default="train",
                   choices=("train", "val", "test", "all"),
                   help="Which slot(s) of (train, val, test) to put the N samples into. "
                        "'all' puts the same N into every slot (overfit memorization).")
    args = p.parse_args()

    with open(args.src, "rb") as f:
        train_files, val_files, test_files = pickle.load(f)
    sub = train_files[: args.n]
    print(f"src train={len(train_files)}  → sub n={len(sub)}  (slot={args.put_into})")

    if args.put_into == "train":
        new = (sub, val_files, test_files)
    elif args.put_into == "val":
        new = ([], sub, test_files)
    elif args.put_into == "test":
        new = ([], val_files, sub)
    else:  # all
        new = (sub, sub, sub)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(new, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
