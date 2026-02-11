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

    def _check_strict_consecutive(self, path_list):
        """
        Return True iff every consecutive pair of paths has exactly refresh_rate minutes apart.
        Uses filename timestamp; if any gap != refresh_rate, return False (block should be discarded).
        """
        if len(path_list) <= 1:
            return True
        for i in range(len(path_list) - 1):
            t0 = self.extract_time(os.path.basename(path_list[i]))
            t1 = self.extract_time(os.path.basename(path_list[i + 1]))
            if t0 is None or t1 is None:
                return False
            try:
                d0 = datetime.strptime(t0, TIME_FMT)
                d1 = datetime.strptime(t1, TIME_FMT)
                delta_min = (d1 - d0).total_seconds() / 60.0
                if abs(delta_min - self.refresh_rate) > 0.1:
                    return False
            except ValueError:
                return False
        return True

    def _are_times_consecutive(self, time_list):
        """
        Return True iff time_list (ordered) has every pair exactly refresh_rate minutes apart.
        time_list: list of time_str.
        """
        if len(time_list) <= 1:
            return True
        for i in range(len(time_list) - 1):
            try:
                d0 = datetime.strptime(time_list[i], TIME_FMT)
                d1 = datetime.strptime(time_list[i + 1], TIME_FMT)
                delta_min = (d1 - d0).total_seconds() / 60.0
                if abs(delta_min - self.refresh_rate) > 0.1:
                    return False
            except ValueError:
                return False
        return True

    def _build_v2v_clips_no_overlap(self, sorted_times, time_to_path,
                                    clip_length=16,
                                    strict_consecutive=True,
                                    filter_radar=True):
        """
        Sort by time first, then split into fixed-length clips with no overlapping frames between samples.

        - sorted_times: Full list of timestamps, already sorted by time.
        - Scan left to right: if segment [i, i+clip_length) exists in time_to_path and is strictly
          consecutive, emit one clip and set i += clip_length (no overlap); else i += 1 and retry.
        - strict_consecutive: when True, require adjacent frames in the segment to be exactly refresh_rate apart.
        - filter_radar: when True and coverage_threshold > 0, keep clip only if last frame passes is_radar_sparse_npy.
        Returns: [(path_list, path_list), ...], each path_list of length clip_length; no overlap between samples.
        """
        paired = []
        i = 0
        while i + clip_length <= len(sorted_times):
            segment = sorted_times[i:i + clip_length]
            if not all(tt in time_to_path for tt in segment):
                i += 1
                continue
            if strict_consecutive and not self._are_times_consecutive(segment):
                i += 1
                continue
            path_list = [time_to_path[tt] for tt in segment]
            if filter_radar and self.coverage_threshold > 0:
                if self.is_radar_sparse_npy(path_list[-1]):
                    i += 1
                    continue
            paired.append((path_list, path_list))
            i += clip_length  # no overlap: next segment starts after this clip
        return paired

    def _get_aligned_v2v_from_times(self, times, time_to_path,
                                    clip_length=16,
                                    strict_consecutive=True,
                                    filter_radar=True):
        """
        [Deprecated: samples overlap] Kept for compatibility. Use _build_v2v_clips_no_overlap on sorted times instead.
        """
        paired = []
        for t0 in times:
            try:
                t0_dt = datetime.strptime(t0, TIME_FMT)
                times_clip = [
                    (t0_dt + timedelta(minutes=self.refresh_rate * i)).strftime(TIME_FMT)
                    for i in range(clip_length)
                ]
                if not all(tt in time_to_path for tt in times_clip):
                    continue
                path_list = [time_to_path[tt] for tt in times_clip]
                if strict_consecutive and not self._check_strict_consecutive(path_list):
                    continue
                if filter_radar and self.coverage_threshold > 0:
                    if self.is_radar_sparse_npy(path_list[-1]):
                        continue
                paired.append((path_list, path_list))
            except ValueError:
                logger.info(f"Failed to parse time: {t0}")
                continue
        return paired

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

    # ---------- V2V: sort all by time -> fixed-length non-overlapping clips -> then split ----------

    def build_filelist_v2v_by_days(self, save_dir,
                                   file_name="dataset_filelist_v2v.pkl",
                                   split_ratio=(0.7, 0.2, 0.1),
                                   fixed_test_days=None,
                                   clip_length=16,
                                   strict_consecutive=True,
                                   filter_radar=True):
        """
        1) Sort full dataset by time; 2) split into fixed-length non-overlapping clips; 3) split train/val/test by day.
        A clip's day is determined by the first frame's timestamp.
        """
        random.seed(self.seed)

        all_files = self._walk_all_npy()
        day_to_times, time_to_path = self._group_by_day(all_files)
        all_days = sorted(day_to_times.keys())
        # Full timeline sorted by time (continuous across days)
        sorted_times = sorted(time_to_path.keys())

        # Fixed-length non-overlapping clips (already in time order)
        all_clips = self._build_v2v_clips_no_overlap(
            sorted_times, time_to_path,
            clip_length=clip_length,
            strict_consecutive=strict_consecutive,
            filter_radar=filter_radar,
        )
        if not all_clips:
            raise RuntimeError(
                "No V2V clips found (no-overlap). Check clip_length, strict_consecutive, filter_radar and date range."
            )

        # Assign each clip to a day by the first frame's timestamp (first 8 chars = YYYYMMDD)
        def clip_day(clip_item):
            path0 = clip_item[0][0]
            t = self.extract_time(os.path.basename(path0))
            return t[:8] if t else ""

        clips_by_day = {}
        for c in all_clips:
            d = clip_day(c)
            clips_by_day.setdefault(d, []).append(c)

        if fixed_test_days is not None:
            fixed_test_days = set(str(d) for d in fixed_test_days)
            test_days = [d for d in all_days if d in fixed_test_days]
            remaining_days = [d for d in all_days if d not in fixed_test_days]
            random.shuffle(remaining_days)
            total_remaining = len(remaining_days)
            train_days = round(split_ratio[0] / (split_ratio[0] + split_ratio[1]) * total_remaining)
            val_days = total_remaining - train_days
            train_days = remaining_days[:train_days]
            val_days = remaining_days[train_days:train_days + val_days]
        else:
            random.shuffle(all_days)
            total_days = len(all_days)
            n_train = round(split_ratio[0] * total_days)
            n_val = round(split_ratio[1] * total_days)
            n_test = round(split_ratio[2] * total_days)
            train_days = all_days[:n_train]
            val_days = all_days[n_train:n_train + n_val]
            test_days = all_days[n_train + n_val:n_train + n_val + n_test]

        train_files = [c for d in train_days for c in clips_by_day.get(d, [])]
        val_files = [c for d in val_days for c in clips_by_day.get(d, [])]
        test_files = [c for d in test_days for c in clips_by_day.get(d, [])]

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, 'wb') as f:
            pickle.dump((train_files, val_files, test_files), f)

        logger.info("Saved V2V dataset (no-overlap) to: %s", save_path)
        logger.info("V2V clip_length=%d, strict_consecutive=%s, filter_radar=%s | total clips=%d",
                    clip_length, strict_consecutive, filter_radar, len(all_clips))
        logger.info("Split (days): train=%d, val=%d, test=%d | Split (files): train=%d, val=%d, test=%d",
                    len(train_days), len(val_days), len(test_days), len(train_files), len(val_files), len(test_files))

        return train_files, val_files, test_files

    # ---------- V2V: sort all by time -> fixed-length non-overlapping clips -> then split by blocks ----------

    def build_filelist_v2v_by_blocks(self, save_dir,
                                    file_name="dataset_filelist_v2v.pkl",
                                    block_size=96,
                                    split_ratio=(0.7, 0.2, 0.1),
                                    drop_last=False,
                                    clip_length=16,
                                    strict_consecutive=True,
                                    filter_radar=True):
        """
        1) Sort full dataset by time; 2) split into fixed-length non-overlapping clips; 3) split train/val/test by blocks.

        clip_length: Frames per sample (length of one video), e.g. 16 means 16-frame clips.
        block_size:  Number of clips per block for split; whole blocks are assigned to train/val/test to avoid leakage.
        """
        random.seed(self.seed)

        all_files = self._walk_all_npy()
        _, time_to_path = self._group_by_day(all_files)
        sorted_times = sorted(time_to_path.keys())

        # Fixed-length non-overlapping clips (output order is time order, no extra sort needed)
        files_v2v = self._build_v2v_clips_no_overlap(
            sorted_times, time_to_path,
            clip_length=clip_length,
            strict_consecutive=strict_consecutive,
            filter_radar=filter_radar,
        )
        if not files_v2v:
            raise RuntimeError(
                "No V2V clips found (no-overlap). Check clip_length, strict_consecutive, filter_radar and date range."
            )

        blocks = [files_v2v[i:i + block_size] for i in range(0, len(files_v2v), block_size)]
        if drop_last and len(blocks) > 0 and len(blocks[-1]) < block_size:
            blocks = blocks[:-1]

        total_blocks = len(blocks)
        if total_blocks == 0:
            raise RuntimeError("No blocks formed. Try smaller block_size or drop_last=False.")

        indices = list(range(total_blocks))
        random.shuffle(indices)

        n_train = round(split_ratio[0] * total_blocks)
        n_val = round(split_ratio[1] * total_blocks)
        n_test = round(split_ratio[2] * total_blocks)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:n_train + n_val + n_test]

        def flatten(idx_list):
            out = []
            for bi in idx_list:
                out.extend(blocks[bi])
            return out

        train_files = flatten(train_idx)
        val_files = flatten(val_idx)
        test_files = flatten(test_idx)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, 'wb') as f:
            pickle.dump((train_files, val_files, test_files), f)

        logger.info("Saved V2V block-based dataset (no-overlap) to: %s", save_path)
        logger.info("V2V clip_length=%d, strict_consecutive=%s, filter_radar=%s | total clips=%d",
                    clip_length, strict_consecutive, filter_radar, len(files_v2v))
        logger.info("Blocks: train=%d, val=%d, test=%d | Split (files): train=%d, val=%d, test=%d",
                    len(train_idx), len(val_idx), len(test_idx), len(train_files), len(val_files), len(test_files))

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
