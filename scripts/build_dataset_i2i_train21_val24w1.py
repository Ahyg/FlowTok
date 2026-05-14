"""Combine existing train and val i2i pkls into one with both train and val splits.

The FlowTok pkl format is a 3-tuple ``(train_files, val_files, test_files)``,
loaded by ``data.dataset.SatelliteRadarNpyDataset._collect_files`` as::

    train_files, val_files, test_files = pickle.load(f)

Existing split-specific pkls have data in only one slot (e.g. the train pkl
has files in ``elem[0]`` and empty lists in the other two). This script merges
a train-slot pkl and a val-slot pkl into a single pkl where both slots are
populated, so the AE dataloader can request ``filelist_split="train"`` and
``filelist_split="val"`` from the same file.
"""
import argparse
import pickle
from pathlib import Path


def _extract_nonempty(triple, preferred_idx, label):
    """Return the populated list from a (train, val, test) triple.

    Prefers ``triple[preferred_idx]`` if non-empty, otherwise falls back to
    whichever slot has data (with a warning), and raises if all are empty.
    """
    if not (isinstance(triple, tuple) and len(triple) == 3):
        raise ValueError(
            f"{label} pkl is not a 3-tuple; got type={type(triple).__name__}"
            f" len={len(triple) if hasattr(triple, '__len__') else '?'}"
        )
    primary = triple[preferred_idx]
    if len(primary) > 0:
        return primary
    # Fallback: pick any non-empty slot.
    for i, slot in enumerate(triple):
        if len(slot) > 0:
            print(
                f"WARNING: {label} pkl had empty slot {preferred_idx}; "
                f"falling back to slot {i} with {len(slot)} entries."
            )
            return slot
    raise ValueError(f"{label} pkl has all three slots empty.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_pkl", required=True,
                   help="Existing pkl with train files in slot 0 of the 3-tuple.")
    p.add_argument("--val_pkl", required=True,
                   help="Existing pkl with val files in slot 1 of the 3-tuple.")
    p.add_argument("--out_pkl", required=True,
                   help="Output path for the combined 3-tuple pkl.")
    args = p.parse_args()

    with open(args.train_pkl, "rb") as f:
        train_obj = pickle.load(f)
    with open(args.val_pkl, "rb") as f:
        val_obj = pickle.load(f)

    train_files = _extract_nonempty(train_obj, preferred_idx=0, label="train")
    val_files = _extract_nonempty(val_obj, preferred_idx=1, label="val")

    # Preserve test slot from val pkl if it has one, else empty list.
    test_files = []
    if isinstance(val_obj, tuple) and len(val_obj) == 3 and len(val_obj[2]) > 0:
        test_files = val_obj[2]
    elif isinstance(train_obj, tuple) and len(train_obj) == 3 and len(train_obj[2]) > 0:
        test_files = train_obj[2]

    combined = (train_files, val_files, test_files)
    Path(args.out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(combined, f)

    print(
        f"Wrote {args.out_pkl}: "
        f"train={len(train_files)} files, "
        f"val={len(val_files)} files, "
        f"test={len(test_files)} files"
    )


if __name__ == "__main__":
    main()
