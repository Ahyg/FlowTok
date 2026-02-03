import glob
import os
import pickle
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


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
    ):
        super().__init__()
        self.base_dir = base_dir
        self.years = [str(y) for y in years] if years is not None else []
        self.mode = mode
        self.ir_band_indices = list(range(10)) if ir_band_indices is None else sorted(ir_band_indices)
        self.use_lightning = use_lightning
        self.filelist_path = filelist_path
        self.filelist_split = filelist_split
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

    def __getitem__(self, idx):
        item = self.files[idx]
        if isinstance(item, (tuple, list)) and len(item) == 2:
            hist_paths, radar_paths = item
            if self.mode == "satellite":
                path = hist_paths[-1]
            elif self.mode == "radar":
                path = radar_paths[0]
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
        else:
            path = item

        arr = np.load(path)  # (12, H, W)

        if self.mode == "satellite":
            ir = arr[self.ir_band_indices]
            lgt = arr[-2] if self.use_lightning else None
            ir_scaled, lgt_scaled = self.scale_sat_lgt_img(ir, lgt)
            if self.use_lightning:
                img = np.concatenate([ir_scaled, lgt_scaled[None, ...]], axis=0)
            else:
                img = ir_scaled
        elif self.mode == "radar":
            radar = self.scale_radar_img(arr[-1])
            img = radar[None, ...]
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return {
            "image": torch.from_numpy(img).float(),
            "path": path,
        }
