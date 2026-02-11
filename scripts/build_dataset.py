import os
import json
import argparse
import logging
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.DatasetBuilder import NpyDatasetBuilder


def parse_float_tuple(value: str):
    try:
        return tuple(map(float, value.strip().strip("()").replace(" ", "").split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Tuple must be a string of numbers separated by commas, like '0.7,0.1,0.2'."
        ) from exc


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.save_dir, "build_dataset.log")),
        ],
    )
    logger = logging.getLogger("DatasetBuilder")

    with open(os.path.join(args.save_dir, "build_dataset_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    builder = NpyDatasetBuilder(
        data_root=args.data_root,
        start_date=args.start_date,
        end_date=args.end_date,
        history_frames=args.history_frames,
        future_frame=args.future_frame,
        refresh_rate=args.refresh_rate,
        coverage_threshold=args.coverage_threshold,
        seed=args.seed,
    )

    if args.v2v:
        # Real-time v2v: fixed-length clips, optional strict_consecutive and filter_radar
        if args.split_mode == "blocks":
            train_files, val_files, test_files = builder.build_filelist_v2v_by_blocks(
                save_dir=args.save_dir,
                file_name=args.dataset_pkl_name,
                block_size=args.block_size,
                split_ratio=args.split_ratio,
                drop_last=args.drop_last,
                clip_length=args.clip_length,
                strict_consecutive=args.strict_consecutive,
                filter_radar=args.filter_radar,
            )
        else:
            train_files, val_files, test_files = builder.build_filelist_v2v_by_days(
                save_dir=args.save_dir,
                file_name=args.dataset_pkl_name,
                split_ratio=args.split_ratio,
                fixed_test_days=args.fixed_test_days,
                clip_length=args.clip_length,
                strict_consecutive=args.strict_consecutive,
                filter_radar=args.filter_radar,
            )
    elif args.split_mode == "blocks":
        train_files, val_files, test_files = builder.build_filelist_by_blocks(
            save_dir=args.save_dir,
            file_name=args.dataset_pkl_name,
            block_size=args.block_size,
            split_ratio=args.split_ratio,
            drop_last=args.drop_last,
        )
    else:
        train_files, val_files, test_files = builder.build_filelist_by_days(
            save_dir=args.save_dir,
            file_name=args.dataset_pkl_name,
            split_ratio=args.split_ratio,
            fixed_test_days=args.fixed_test_days,
        )

    logger.info("Done. train=%d, val=%d, test=%d", len(train_files), len(val_files), len(test_files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build FlowTok dataset filelist.")
    parser.add_argument("--data-root", required=True, type=str, help="Root directory containing YYYY/MM/DD/*.npy")
    parser.add_argument("--save-dir", required=True, type=str, help="Directory to save dataset_filelist.pkl")
    parser.add_argument("--dataset-pkl-name", default="dataset_filelist.pkl", type=str)

    parser.add_argument("--start-date", default="", type=str, help="YYYYMMDD")
    parser.add_argument("--end-date", default="", type=str, help="YYYYMMDD")
    parser.add_argument("--history-frames", default=0, type=int)
    parser.add_argument("--future-frame", default=0, type=int)
    parser.add_argument("--refresh-rate", default=10, type=int)
    parser.add_argument("--coverage-threshold", default=0.05, type=float)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--split-mode", default="blocks", choices=("blocks", "days"))
    parser.add_argument("--block-size", default=96, type=int, help="(blocks split only) samples per block for train/val/test split; not video length")
    parser.add_argument("--split-ratio", default=(0.7, 0.2, 0.1), type=parse_float_tuple)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--fixed-test-days", default=None, type=lambda s: s.split(","))

    parser.add_argument("--v2v", action="store_true", help="Build aligned (sat_paths, radar_paths) for real-time v2v with fixed-length clips")
    parser.add_argument("--clip-length", default=16, type=int, help="V2V: frames per video (each sample = one clip of this many frames)")
    parser.add_argument("--no-strict-consecutive", dest="strict_consecutive", action="store_false", help="V2V: do not discard blocks with non-consecutive frames")
    parser.add_argument("--no-filter-radar", dest="filter_radar", action="store_false", help="V2V: disable radar sparse filtering")
    parser.set_defaults(strict_consecutive=True, filter_radar=True)

    main(parser.parse_args())

