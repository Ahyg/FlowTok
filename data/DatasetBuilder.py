import os
import re
import glob
import pickle
import random
from datetime import datetime, timedelta

import numpy as np
import logging

logger = logging.getLogger(__name__)

TIME_FMT = "%Y%m%dT%H%M"   # e.g. 20240701T0000

class NpyDatasetBuilder:
    """
    Dataset builder for fused .npy samples stored as:

        <data_root>/YYYY/MM/DD/20240712T0000.npy

    Each npy file:
        shape: (C, H, W)
        C = 10 satellite channels + 1 lightning + 1 radar (last channel = radar)

    The builder:
        - collects all .npy files
        - groups by day (YYYYMMDD)
        - builds (history sequence, target) pairs in time
        - filters out scenes with insufficient radar coverage
        - splits into train/val/test (by days or by blocks)
    """

    def __init__(self,
                 data_root,
                 start_date="",          # e.g. "20240701"
                 end_date="",            # e.g. "20241231"
                 history_frames=0,
                 future_frame=0,
                 refresh_rate=10,        # minutes
                 coverage_threshold=0.05,
                 seed=42):
        self.data_root = data_root
        self.start_date = start_date
        self.end_date = end_date
        self.history_frames = history_frames
        self.future_frame = future_frame
        self.refresh_rate = refresh_rate
        self.coverage_threshold = coverage_threshold
        self.seed = seed

    # ---------- Basic tools ----------

    def extract_time(self, filename):
        """
        Extract timestamp from filename, e.g. 20240701T0000.npy -> '20240701T0000'
        """
        m = re.search(r"(\d{8}T\d{4})", filename)
        return m.group(1) if m else None

    def _walk_all_npy(self):
        """
        Walk through all .npy files under data_root, return [(time_str, full_path), ...],
        Sorted by time.
        """
        all_files = []
        for dirpath, dirnames, filenames in os.walk(self.data_root):
            dirnames[:] = [d for d in dirnames if d not in ["_neg", "figs"]]

            for fn in filenames:
                if not fn.endswith(".npy"):
                    continue
                t = self.extract_time(fn)
                if t is None:
                    continue
                day = t[:8]
                if self.start_date and day < self.start_date:
                    continue
                if self.end_date and day > self.end_date:
                    continue
                full = os.path.join(dirpath, fn)
                all_files.append((t, full))

        if not all_files:
            raise RuntimeError(f"No .npy files found under {self.data_root} in the given date range.")

        # Sort by time    
        all_files.sort(key=lambda x: x[0])
        return all_files

    def _group_by_day(self, all_files):
        """
        Input all_files = [(time_str, path), ...]
        Output:
            day_to_times: { 'YYYYMMDD': [time_str1, time_str2, ...] }
            time_to_path: { time_str: full_path }
        """
        day_to_times = {}
        time_to_path = {}

        for t, path in all_files:
            day = t[:8]
            time_to_path[t] = path
            day_to_times.setdefault(day, []).append(t)

        # Sort times within each day
        for day in day_to_times:
            day_to_times[day].sort()

        return day_to_times, time_to_path

    # ---------- Coverage filtering (using the last channel radar in npy) ----------

    def is_radar_sparse_npy(self, npy_path, precip_thr=0.0, conv_thr=33.0):
        """
        Return True if the radar scene is sparse (to be filtered out),
        False if valid.

        Here we assume:
            arr = np.load(npy_path)  # shape: (C, H, W)
            radar = arr[-1]          # last channel = radar reflectivity (dBZ)

            - AF/CAF/SAF
            - coverage_threshold for PAF
        """
        if self.coverage_threshold == 0.0:
            return False

        arr = np.load(npy_path)   # (C, H, W)
        radar = arr[-1]           # last channel = radar reflectivity (dBZ)

        radar_flat = radar.ravel()
        total_pixels = radar_flat.size
        valid = ~np.isnan(radar_flat)

        precip_mask = (radar_flat > precip_thr) & valid
        conv_mask   = (radar_flat > conv_thr) & valid
        strat_mask  = (radar_flat > 0) & (radar_flat <= conv_thr) & valid

        paf = precip_mask.sum() / total_pixels
        caf = conv_mask.sum()    / total_pixels
        saf = strat_mask.sum()   / total_pixels

        # Popcorn convection:
        if (saf < 0.05) and (caf >= 0.01):
            return False

        # Sparse filtering
        is_sparse = (paf < self.coverage_threshold) and (caf < 0.005) and (saf < 0.05)
        return is_sparse

    # ---------- Build (hist_seq, target) pairs from time series ----------

    def _get_paired_from_times(self, times, time_to_path, return_time=False):
        """
        times: A list of time_str for a certain group (e.g., several days), already sorted by time.
        time_to_path: Global dictionary {time_str: full_path}
        Returns:
            If return_time=False:
                [(hist_seq_paths, [target_path]), ...]
            If return_time=True:
                [(anchor_datetime, hist_seq_paths, [target_path]), ...]
        """
        paired = []
        for t0 in times:
            try:
                t0_dt = datetime.strptime(t0, TIME_FMT)

                # Historical frames: t0 - k*refresh_rate
                hist_times = [
                    (t0_dt - timedelta(minutes=self.refresh_rate * i)).strftime(TIME_FMT)
                    for i in reversed(range(self.history_frames + 1))
                ]
                if not all(ht in time_to_path for ht in hist_times):
                    continue

                hist_paths = [time_to_path[ht] for ht in hist_times]

                # Future target frame
                target_dt = t0_dt + timedelta(minutes=self.refresh_rate * self.future_frame)
                target_ts = target_dt.strftime(TIME_FMT)
                if target_ts not in time_to_path:
                    continue

                target_path = time_to_path[target_ts]

                # Coverage filtering
                if self.is_radar_sparse_npy(target_path):
                    continue

                if return_time:
                    paired.append((target_dt, hist_paths, [target_path]))
                else:
                    paired.append((hist_paths, [target_path]))
            except ValueError:
                logger.info(f"Failed to parse time: {t0}")
                continue
        return paired

    # ---------- Split train/val/test by "days" ----------

    def build_filelist_by_days(self, save_dir,
                               file_name="dataset_filelist.pkl",
                               split_ratio=(0.7, 0.2, 0.1),
                               fixed_test_days=None):
        """
        - First split by days: train_days / val_days / test_days
        - Then within each split, build (hist_seq, target) pairs based on time windows
        """
        random.seed(self.seed)

        all_files = self._walk_all_npy()
        day_to_times, time_to_path = self._group_by_day(all_files)
        all_days = sorted(day_to_times.keys())

        # ---- Split train/val/test by days ----
        if fixed_test_days is not None:
            fixed_test_days = set(str(d) for d in fixed_test_days)
            test_days = [d for d in all_days if d in fixed_test_days]
            remaining_days = [d for d in all_days if d not in fixed_test_days]

            random.shuffle(remaining_days)
            total_remaining = len(remaining_days)
            train_days = round(split_ratio[0] / (split_ratio[0] + split_ratio[1]) * total_remaining)
            val_days   = total_remaining - train_days

            train_days = remaining_days[:train_days]
            val_days   = remaining_days[train_days:train_days+val_days]
        else:
            random.shuffle(all_days)
            total_days = len(all_days)
            n_train = round(split_ratio[0] * total_days)
            n_val   = round(split_ratio[1] * total_days)
            n_test  = round(split_ratio[2] * total_days)

            train_days = all_days[:n_train]
            val_days   = all_days[n_train:n_train+n_val]
            test_days  = all_days[n_train+n_val:n_train+n_val+n_test]

        # ---- Build paired file lists based on days in each split ----
        train_files = self._get_paired_from_times(
            [t for d in train_days for t in day_to_times[d]],
            time_to_path
        )
        val_files = self._get_paired_from_times(
            [t for d in val_days for t in day_to_times[d]],
            time_to_path
        )
        test_files = self._get_paired_from_times(
            [t for d in test_days for t in day_to_times[d]],
            time_to_path
        )

        # ---- Save ----
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, 'wb') as f:
            pickle.dump((train_files, val_files, test_files), f)

        logger.info(f"Saved dataset to: {save_path}")
        logger.info("Total paired files: %d | Total days: %d",
                    len(train_files)+len(val_files)+len(test_files), len(all_days))
        logger.info("Split (days): train=%d, val=%d, test=%d",
                    len(train_days), len(val_days), len(test_days))
        logger.info("Split (files):  train=%d, val=%d, test=%d",
                    len(train_files), len(val_files), len(test_files))

        return train_files, val_files, test_files

    # ---------- Split train/val/test by "time blocks" ----------

    def build_filelist_by_blocks(self, save_dir,
                                 file_name="dataset_filelist.pkl",
                                 block_size=96,
                                 split_ratio=(0.7, 0.2, 0.1),
                                 drop_last=False):
        """
        - First generate all paired files (with anchor time)
        - Sort by time
        - Split into blocks
        - Randomly assign blocks to train/val/test
        """
        random.seed(self.seed)

        all_files = self._walk_all_npy()
        _, time_to_path = self._group_by_day(all_files)
        all_times = [t for t, _ in all_files]

        # All paired files (with timestamps)
        files_with_time = self._get_paired_from_times(all_times, time_to_path, return_time=True)
        if not files_with_time:
            raise RuntimeError("No paired files found after filtering. Please check thresholds / history / future settings.")

        # Already sorted by time (because all_files is sorted)
        # files_with_time: [(anchor_dt, hist_paths, [target_path]), ...]

        # Split into blocks
        blocks = [files_with_time[i:i+block_size] for i in range(0, len(files_with_time), block_size)]
        if drop_last and len(blocks) > 0 and len(blocks[-1]) < block_size:
            blocks = blocks[:-1]

        total_blocks = len(blocks)
        if total_blocks == 0:
            raise RuntimeError("No blocks formed. Try smaller block_size or drop_last=False.")

        indices = list(range(total_blocks))
        random.shuffle(indices)

        n_train = round(split_ratio[0] * total_blocks)
        n_val   = round(split_ratio[1] * total_blocks)
        n_test  = round(split_ratio[2] * total_blocks)

        train_idx = indices[:n_train]
        val_idx   = indices[n_train:n_train+n_val]
        test_idx  = indices[n_train+n_val:n_train+n_val+n_test]

        def flatten_blocks(idx_list):
            out = []
            for bi in idx_list:
                for (ts, hist_paths, radar_list) in blocks[bi]:
                    out.append((hist_paths, radar_list))
            return out

        train_files = flatten_blocks(train_idx)
        val_files   = flatten_blocks(val_idx)
        test_files  = flatten_blocks(test_idx)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, 'wb') as f:
            pickle.dump((train_files, val_files, test_files), f)

        logger.info("Saved block-based dataset to: %s", save_path)
        logger.info("Total paired files: %d | Block size: %d | Total blocks: %d",
                    len(files_with_time), block_size, total_blocks)
        logger.info("Split (blocks): train=%d, val=%d, test=%d",
                    len(train_idx), len(val_idx), len(test_idx))
        logger.info("Split (files):  train=%d, val=%d, test=%d",
                    len(train_files), len(val_files), len(test_files))

        return train_files, val_files, test_files

    def load_filelist(self, path):
        with open(path, 'rb') as f:
            logger.info(f"Loaded dataset from: {path}")
            return pickle.load(f)
