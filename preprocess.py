from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Iterable

if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"

import blosc2
import numpy as np


blosc2.set_nthreads(1)


def load_pickle(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _load_b2nd(path: Path) -> np.ndarray:
    array = blosc2.open(urlpath=str(path), mode="r")
    return np.asarray(array[:])


def load_case_array(case_dir: str | Path, case_id: str, kind: str) -> np.ndarray:
    case_dir = Path(case_dir)
    suffix = "_seg" if kind == "seg" else ""
    b2nd_path = case_dir / f"{case_id}{suffix}.b2nd"
    npy_path = case_dir / f"{case_id}{suffix}.npy"
    npz_path = case_dir / f"{case_id}.npz"

    if b2nd_path.is_file():
        return _load_b2nd(b2nd_path)

    if npy_path.is_file():
        return np.load(npy_path)

    if npz_path.is_file():
        key = "seg" if kind == "seg" else "data"
        return np.load(npz_path)[key]

    raise FileNotFoundError(f"Cannot find {kind} array for {case_id} in {case_dir}")


def load_case(case_dir: str | Path, case_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
    case_dir = Path(case_dir)
    data = load_case_array(case_dir, case_id, "data")
    seg = load_case_array(case_dir, case_id, "seg")
    properties = load_pickle(case_dir / f"{case_id}.pkl")
    return data.astype(np.float32), seg.astype(np.int16), properties


def _normalize_class_key(class_key) -> int | tuple:
    if isinstance(class_key, str):
        try:
            return int(class_key)
        except ValueError:
            return class_key
    return class_key


def select_foreground_voxel(class_locations: dict, rng: np.random.Generator) -> np.ndarray | None:
    eligible = []
    for raw_key, locations in class_locations.items():
        key = _normalize_class_key(raw_key)
        if len(locations) == 0:
            continue
        if key == 0:
            continue
        if isinstance(key, tuple) and len(key) > 0 and key[0] == -1:
            continue
        eligible.append(np.asarray(locations))

    if not eligible:
        return None

    locations = eligible[int(rng.integers(0, len(eligible)))]
    return locations[int(rng.integers(0, len(locations)))]


def sample_patch_bbox(
    image_shape: Iterable[int],
    patch_size: Iterable[int],
    class_locations: dict | None,
    oversample_foreground: bool,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    image_shape = np.asarray(list(image_shape), dtype=int)
    patch_size = np.asarray(list(patch_size), dtype=int)

    lower_bounds = []
    upper_bounds = []
    for dimension, patch in zip(image_shape, patch_size):
        if dimension <= patch:
            lower_bounds.append(0)
            upper_bounds.append(0)
        else:
            lower_bounds.append(0)
            upper_bounds.append(dimension - patch)

    if oversample_foreground and class_locations is not None:
        voxel = select_foreground_voxel(class_locations, rng)
        if voxel is not None:
            starts = []
            for axis in range(len(patch_size)):
                center = int(voxel[axis + 1])
                proposed = center - patch_size[axis] // 2
                starts.append(int(np.clip(proposed, lower_bounds[axis], upper_bounds[axis])))
            return [(start, start + size) for start, size in zip(starts, patch_size)]

    starts = []
    for axis in range(len(patch_size)):
        if upper_bounds[axis] == 0:
            starts.append(0)
        else:
            starts.append(int(rng.integers(lower_bounds[axis], upper_bounds[axis] + 1)))
    return [(start, start + size) for start, size in zip(starts, patch_size)]


def crop_or_pad(array: np.ndarray, bbox: list[tuple[int, int]], fill_value: float) -> np.ndarray:
    channels = array.shape[0]
    patch_size = [upper - lower for lower, upper in bbox]
    result = np.full((channels, *patch_size), fill_value, dtype=array.dtype)

    src_slices = []
    dst_slices = []
    for axis, (lower, upper) in enumerate(bbox, start=1):
        src_lower = max(0, lower)
        src_upper = min(array.shape[axis], upper)
        dst_lower = max(0, -lower)
        dst_upper = dst_lower + (src_upper - src_lower)
        src_slices.append(slice(src_lower, src_upper))
        dst_slices.append(slice(dst_lower, dst_upper))

    result[(slice(None), *dst_slices)] = array[(slice(None), *src_slices)]
    return result


def zscore_if_needed(volume: np.ndarray) -> np.ndarray:
    mean = float(volume.mean())
    std = float(volume.std())
    if std < 1e-8:
        return volume
    return (volume - mean) / std
