import glob
import os
import pickle
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


def collate_sat2radar_v2v(batch):
    """
    Collate variable-length (sat_video, radar_video) from sat2radar_v2v mode.
    Pads to max T in the batch. Returns valid_mask [B, T_max] (True = real frame).
    """
    if not batch:
        return {}
    sat_list = [b["sat_video"] for b in batch]
    radar_list = [b["radar_video"] for b in batch]
    n_frames = [b["n_frames"] for b in batch]
    T_max = max(n_frames)
    B = len(batch)
    _, C_sat, H, W = sat_list[0].shape
    _, C_rad, _, _ = radar_list[0].shape

    sat_padded = torch.zeros(B, T_max, C_sat, H, W, dtype=sat_list[0].dtype)
    radar_padded = torch.zeros(B, T_max, C_rad, H, W, dtype=radar_list[0].dtype)
    valid_mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i in range(B):
        T_i = n_frames[i]
        sat_padded[i, :T_i] = sat_list[i]
        radar_padded[i, :T_i] = radar_list[i]
        valid_mask[i, :T_i] = True

    return {
        "sat_video": sat_padded,
        "radar_video": radar_padded,
        "valid_mask": valid_mask,
        "n_frames": n_frames,
    }


class SatelliteRadarNpyDataset(Dataset):
    """
    Load fused .npy frames with shape (12, H, W).

    Channels:
        0-9  : Satellite IR channels
        10   : Lightning
        11   : Radar reflectivity
    """

    def __init__(
        self,
        base_dir: str | None = None,
        years: Iterable[str] | None = None,
        mode: str = "satellite",
        ir_band_indices: Iterable[int] | None = None,
        use_lightning: bool = True,
        filelist_path: str | None = None,
        filelist_split: str = "train",
        files: list | None = None,
        *,
        # Extra options for sequence / video use-cases.
        history_frames: int | None = None,
        future_frames: int | None = None,
        frame_stride: int = 1,
        # For real-time v2v: aligned (sat, radar) pairs, variable T (T=1 => i2i).
        # num_frames: int = fixed T; (min_t, max_t) = sample T in [min_t, max_t]; None = use all.
        num_frames: int | tuple[int, int] | None = None,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.years = [str(y) for y in years] if years is not None else []
        self.mode = mode
        self.ir_band_indices = list(range(10)) if ir_band_indices is None else sorted(ir_band_indices)
        self.use_lightning = use_lightning
        self.filelist_path = filelist_path
        self.filelist_split = filelist_split
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.frame_stride = max(int(frame_stride), 1)
        self.num_frames = num_frames
        self.files = self._collect_files(files)

    def _collect_files(self, files_override):
        if files_override is not None:
            return files_override
        if self.filelist_path:
            with open(self.filelist_path, "rb") as f:
                train_files, val_files, test_files = pickle.load(f)
            split_map = {"train": train_files, "val": val_files, "test": test_files}
            if self.filelist_split not in split_map:
                raise ValueError(f"Unknown filelist_split: {self.filelist_split}")
            return split_map[self.filelist_split]

        files = []
        if self.base_dir is None or not self.years:
            raise ValueError("base_dir and years are required when filelist_path is not provided.")
        for year in self.years:
            pattern = os.path.join(self.base_dir, year, "**", "*.npy")
            files.extend(glob.glob(pattern, recursive=True))
        files = sorted(files)
        if not files:
            raise ValueError(f"No npy files found under {self.base_dir} for years={self.years}")
        return files

    @staticmethod
    def scale_sat_lgt_img(ir_img, lgt_img=None):
        ir_img = np.nan_to_num(ir_img, nan=0.0, copy=False)
        ir_min, ir_max = 200.0, 320.0
        np.clip(ir_img, ir_min, ir_max, out=ir_img)
        ir_scaled = (ir_img - ir_min) / (ir_max - ir_min)

        lgt_scaled = None
        if lgt_img is not None:
            lgt_img = np.nan_to_num(lgt_img, nan=0.0, copy=False)
            l_min, l_max = 0.1, 50.0
            np.clip(lgt_img, l_min, l_max, out=lgt_img)
            lgt_scaled = (lgt_img - l_min) / (l_max - l_min)
        return ir_scaled, lgt_scaled

    @staticmethod
    def scale_radar_img(mask):
        mask = np.nan_to_num(mask, nan=0.0, copy=False)
        z_min, z_max = 0.0, 60.0
        np.clip(mask, z_min, z_max, out=mask)
        return (mask - z_min) / (z_max - z_min)

    def __len__(self):
        return len(self.files)

    def _select_paths(self, paths, max_frames: int | None):
        """Helper to subsample a list of paths for sequence / video.

        - Applies frame_stride first.
        - If max_frames is not None, keeps only the last max_frames elements.
        """
        if not isinstance(paths, (list, tuple)):
            return [paths]

        # Apply stride (controls effective frame rate).
        paths = paths[:: self.frame_stride]

        if max_frames is not None and max_frames > 0 and len(paths) > max_frames:
            paths = paths[-max_frames:]
        return paths

    def _load_sat_image(self, path: str):
        """Load and scale one satellite(+lightning) frame as (C, H, W)."""
        arr = np.load(path)  # (12, H, W)
        ir = arr[self.ir_band_indices]
        lgt = arr[-2] if self.use_lightning else None
        ir_scaled, lgt_scaled = self.scale_sat_lgt_img(ir, lgt)
        if self.use_lightning and lgt_scaled is not None:
            img = np.concatenate([ir_scaled, lgt_scaled[None, ...]], axis=0)
        else:
            img = ir_scaled
        return img

    def _load_radar_image(self, path: str):
        """Load and scale one radar frame as (1, H, W)."""
        arr = np.load(path)  # (12, H, W)
        radar = self.scale_radar_img(arr[-1])
        img = radar[None, ...]
        return img

    def __getitem__(self, idx):
        item = self.files[idx]

        # Case 1: paired (hist_paths, radar_paths) from DatasetBuilder.
        if isinstance(item, (tuple, list)) and len(item) == 2:
            hist_paths, radar_paths = item

            # Simple single-frame modes (backward-compatible with original behavior).
            if self.mode == "satellite":
                path = hist_paths[-1]
                img = self._load_sat_image(path)
                return {
                    "image": torch.from_numpy(img).float(),
                    "path": path,
                }
            if self.mode == "radar":
                path = radar_paths[0]
                img = self._load_radar_image(path)
                return {
                    "image": torch.from_numpy(img).float(),
                    "path": path,
                }

            # New: satellite history sequence video.
            if self.mode == "satellite_sequence":
                sat_paths = self._select_paths(hist_paths, self.history_frames)
                frames = [self._load_sat_image(p) for p in sat_paths]
                # (T, C, H, W)
                video = np.stack(frames, axis=0)
                return {
                    "video": torch.from_numpy(video).float(),
                    "hist_paths": sat_paths,
                }

            # New: radar future sequence video (if multiple future frames are present).
            if self.mode == "radar_sequence":
                radar_paths_seq = self._select_paths(radar_paths, self.future_frames)
                frames = [self._load_radar_image(p) for p in radar_paths_seq]
                video = np.stack(frames, axis=0)
                return {
                    "video": torch.from_numpy(video).float(),
                    "radar_paths": radar_paths_seq,
                }

            # New: paired satellite history video and radar future video (fixed history→future).
            if self.mode == "sat2radar_video":
                sat_paths = self._select_paths(hist_paths, self.history_frames)
                radar_paths_seq = self._select_paths(radar_paths, self.future_frames)

                sat_frames = [self._load_sat_image(p) for p in sat_paths]
                radar_frames = [self._load_radar_image(p) for p in radar_paths_seq]

                sat_video = np.stack(sat_frames, axis=0)   # (T_sat, C_sat, H, W)
                radar_video = np.stack(radar_frames, axis=0)  # (T_rad, 1, H, W)

                return {
                    "sat_video": torch.from_numpy(sat_video).float(),
                    "radar_video": torch.from_numpy(radar_video).float(),
                    "hist_paths": sat_paths,
                    "radar_paths": radar_paths_seq,
                }

            # Real-time v2v: aligned (sat_t, radar_t) per timestep, variable T (T=1 => i2i).
            # Filelist must be list of (sat_paths, radar_paths) with len(sat_paths)==len(radar_paths)
            # (same timesteps for both domains). num_frames controls segment length: int, (min,max), or None.
            if self.mode == "sat2radar_v2v":
                sat_paths = list(hist_paths)
                radar_paths_seq = list(radar_paths)
                if len(sat_paths) != len(radar_paths_seq):
                    raise ValueError(
                        f"sat2radar_v2v requires aligned pairs: len(sat)={len(sat_paths)} != len(radar)={len(radar_paths_seq)}. "
                        "Use a filelist where each item is (sat_paths, radar_paths) with same length."
                    )
                # Apply stride (same for both)
                sat_paths = sat_paths[:: self.frame_stride]
                radar_paths_seq = radar_paths_seq[:: self.frame_stride]
                n_avail = len(sat_paths)
                if n_avail == 0:
                    raise ValueError("No frames after frame_stride for sat2radar_v2v.")

                # Sample segment length T (support i2i and v2v)
                if self.num_frames is None:
                    T = n_avail
                    start = 0
                elif isinstance(self.num_frames, int):
                    T = min(self.num_frames, n_avail)
                    start = int(np.random.randint(0, n_avail - T + 1)) if n_avail > T else 0
                else:
                    min_t, max_t = self.num_frames
                    T = min(max_t, n_avail)
                    T = max(min_t, min(T, n_avail))
                    start = int(np.random.randint(0, n_avail - T + 1)) if n_avail > T else 0

                sat_paths = sat_paths[start : start + T]
                radar_paths_seq = radar_paths_seq[start : start + T]
                sat_frames = [self._load_sat_image(p) for p in sat_paths]
                radar_frames = [self._load_radar_image(p) for p in radar_paths_seq]
                sat_video = np.stack(sat_frames, axis=0)
                radar_video = np.stack(radar_frames, axis=0)
                return {
                    "sat_video": torch.from_numpy(sat_video).float(),
                    "radar_video": torch.from_numpy(radar_video).float(),
                    "n_frames": T,
                }

            raise ValueError(f"Unsupported mode for paired (hist, radar): {self.mode}")

        # Case 2: unpaired, original single-frame npy list.
        path = item
        arr = np.load(path)  # (12, H, W)

        if self.mode == "satellite":
            ir = arr[self.ir_band_indices]
            lgt = arr[-2] if self.use_lightning else None
            ir_scaled, lgt_scaled = self.scale_sat_lgt_img(ir, lgt)
            if self.use_lightning and lgt_scaled is not None:
                img = np.concatenate([ir_scaled, lgt_scaled[None, ...]], axis=0)
            else:
                img = ir_scaled
            return {
                "image": torch.from_numpy(img).float(),
                "path": path,
            }
        elif self.mode == "radar":
            radar = self.scale_radar_img(arr[-1])
            img = radar[None, ...]
            return {
                "image": torch.from_numpy(img).float(),
                "path": path,
            }
        else:
            raise ValueError(f"Unsupported mode for single-frame item: {self.mode}")
