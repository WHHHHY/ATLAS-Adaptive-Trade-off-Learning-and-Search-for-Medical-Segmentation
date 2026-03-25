from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess import crop_or_pad, load_case, sample_patch_bbox


def _normalize_class_key(class_key) -> int | tuple:
    if isinstance(class_key, str):
        try:
            return int(class_key)
        except ValueError:
            return class_key
    return class_key


def _select_class_locations(class_locations: dict | None, target_label: int | None) -> dict | None:
    if class_locations is None or target_label is None:
        return class_locations

    filtered = {}
    for raw_key, locations in class_locations.items():
        key = _normalize_class_key(raw_key)
        if key == target_label:
            filtered[1] = locations
    return filtered


def _remap_segmentation(seg: np.ndarray, target_label: int | None) -> np.ndarray:
    if target_label is None:
        return seg.astype(np.int16)
    return (seg == target_label).astype(np.int16)


def load_split(split_file: str | Path, fold: int) -> tuple[list[str], list[str]]:
    with Path(split_file).open("r", encoding="utf-8") as handle:
        splits = json.load(handle)
    split = splits[fold]
    return list(split["train"]), list(split["val"])


class CaseStore:
    def __init__(self, case_dir: str | Path, cache_data: bool = False):
        self.case_dir = Path(case_dir)
        self.cache_data = cache_data
        self.cache: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}

    def load(self, case_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
        if self.cache_data and case_id in self.cache:
            return self.cache[case_id]

        case = load_case(self.case_dir, case_id)
        if self.cache_data:
            self.cache[case_id] = case
        return case


class BTCVPatchDataset(Dataset):
    def __init__(
        self,
        case_dir: str | Path,
        case_ids: list[str],
        patch_size: list[int],
        samples_per_epoch: int,
        oversample_foreground_percent: float,
        cache_data: bool = False,
        validate_mode: bool = False,
        seed: int = 1234,
        target_label: int | None = None,
    ):
        self.case_ids = list(case_ids)
        self.patch_size = list(patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.oversample_foreground_percent = float(oversample_foreground_percent)
        self.store = CaseStore(case_dir, cache_data=cache_data)
        self.validate_mode = validate_mode
        self.rng = np.random.default_rng(seed)
        self.target_label = target_label

    def set_worker_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        case_id = self.case_ids[int(self.rng.integers(0, len(self.case_ids)))]
        data, seg, properties = self.store.load(case_id)
        seg = _remap_segmentation(seg, self.target_label)
        force_fg = False if self.validate_mode else self.rng.random() < self.oversample_foreground_percent
        bbox = sample_patch_bbox(
            data.shape[1:],
            self.patch_size,
            _select_class_locations(properties.get("class_locations"), self.target_label),
            force_fg,
            self.rng,
        )

        image = crop_or_pad(data, bbox, fill_value=0.0)
        label = crop_or_pad(seg, bbox, fill_value=0)

        return {
            "image": torch.from_numpy(image).float(),
            "label": torch.from_numpy(label).long(),
            "case_id": case_id,
        }


class BTCVVolumeDataset(Dataset):
    def __init__(self, case_dir: str | Path, case_ids: list[str], cache_data: bool = False, target_label: int | None = None):
        self.case_ids = list(case_ids)
        self.store = CaseStore(case_dir, cache_data=cache_data)
        self.target_label = target_label

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        case_id = self.case_ids[index]
        data, seg, properties = self.store.load(case_id)
        return {
            "image": torch.from_numpy(data).float(),
            "label": torch.from_numpy(_remap_segmentation(seg, self.target_label)).long(),
            "case_id": case_id,
            "properties": properties,
        }
