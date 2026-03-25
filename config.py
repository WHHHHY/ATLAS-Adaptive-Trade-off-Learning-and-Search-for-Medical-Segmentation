from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


DEFAULT_CONFIG = {
    "experiment_name": "slimformer_btcv_minimal",
    "organ": "pancreas",
    "seed": 1234,
    "split_id": 0,
    "device": "cuda",
    "save_dir": "runs",
    "output_dir": "outputs/slimformer_btcv_minimal",
    "pruning_ratio": None,
    "quantization": {
        "enabled": False,
        "mode": "none",
    },
    "data": {
        "root": "data/Dataset002_BTCV",
        "dataset_json": "data/Dataset002_BTCV/dataset.json",
        "plans_file": "data/Dataset002_BTCV/nnUNetPlans.json",
        "split_file": "data/Dataset002_BTCV/splits_final.json",
        "configuration": "3d_fullres",
        "data_identifier": "nnUNetPlans_3d_fullres",
        "patch_size": None,
        "cache_data": False,
        "train_samples_per_epoch": 250,
        "val_samples_per_epoch": 50,
        "oversample_foreground_percent": 0.33,
        "num_workers": 4,
        "persistent_workers": True,
        "pin_memory": True,
    },
    "model": {
        "base_channels": 48,
        "patch_size": None,
        "num_heads": [4, 8, 16, 32],
        "metaformer_layers": [2, 2, 2],
        "token_mixer": "mamba",
        "drop": 0.2,
        "drop_path": 0.2,
        "prune_flags_list": None,
    },
    "train": {
        "batch_size": None,
        "epochs": 1500,
        "lr": 1e-2,
        "weight_decay": 3e-5,
        "momentum": 0.99,
        "grad_clip": 12.0,
        "amp": True,
        "save_every": 50,
        "full_validation_every": 50,
    },
    "validation": {
        "overlap": 0.0,
        "use_full_volume": True,
    },
}


def _has_nested_key(mapping: dict[str, Any], keys: tuple[str, ...]) -> bool:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def _apply_top_level_aliases(user_config: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    alias_override: dict[str, Any] = {}

    dataset_path = user_config.get("dataset_path")
    if dataset_path is not None:
        alias_override.setdefault("data", {})["root"] = dataset_path
        dataset_root = Path(dataset_path)
        if not _has_nested_key(user_config, ("data", "dataset_json")):
            alias_override["data"]["dataset_json"] = str(dataset_root / "dataset.json")
        if not _has_nested_key(user_config, ("data", "plans_file")):
            alias_override["data"]["plans_file"] = str(dataset_root / "nnUNetPlans.json")
        if "split_file" not in user_config and not _has_nested_key(user_config, ("data", "split_file")):
            alias_override["data"]["split_file"] = str(dataset_root / "splits_final.json")

    if "split_file" in user_config:
        alias_override.setdefault("data", {})["split_file"] = user_config["split_file"]
    if "batch_size" in user_config:
        alias_override.setdefault("train", {})["batch_size"] = user_config["batch_size"]
    if "epochs" in user_config:
        alias_override.setdefault("train", {})["epochs"] = user_config["epochs"]
    if "learning_rate" in user_config:
        alias_override.setdefault("train", {})["lr"] = user_config["learning_rate"]
    if "weight_decay" in user_config:
        alias_override.setdefault("train", {})["weight_decay"] = user_config["weight_decay"]
    if "num_workers" in user_config:
        alias_override.setdefault("data", {})["num_workers"] = user_config["num_workers"]

    return deep_update(config, alias_override)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_model_patch_size(spacing: list[float]) -> list[int]:
    if len(spacing) >= 3 and float(spacing[0]) > 3.0 * float(spacing[1]):
        return [1, 2, 2]
    return [2, 2, 2]


def resolve_target_label(labels: dict[str, int], organ: str | None) -> int | None:
    if organ is None:
        return None
    normalized = organ.strip().lower()
    if normalized in {"", "all", "multi_organ", "multi-organ"}:
        return None
    for name, index in labels.items():
        if name.strip().lower() == normalized:
            return int(index)
    raise ValueError(f"Unknown organ '{organ}' in dataset labels.")


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}

    config = deep_update(DEFAULT_CONFIG, user_config)
    config = _apply_top_level_aliases(user_config, config)
    config["config_path"] = str(config_path)

    for section, key in (
        ("data", "root"),
        ("data", "dataset_json"),
        ("data", "plans_file"),
        ("data", "split_file"),
    ):
        config[section][key] = str(resolve_path(config[section][key]))

    if config.get("save_dir") is None:
        config["save_dir"] = config["output_dir"]
    config["save_dir"] = str(resolve_path(config["save_dir"]))
    config["output_dir"] = str(resolve_path(config["output_dir"]))

    dataset_json = _load_json(Path(config["data"]["dataset_json"]))
    plans = _load_json(Path(config["data"]["plans_file"]))
    configuration_name = config["data"]["configuration"]
    plan_config = plans["configurations"][configuration_name]

    data_root = Path(config["data"]["root"])
    preprocessed_dir = data_root / config["data"]["data_identifier"]

    config["data"]["preprocessed_dir"] = str(preprocessed_dir)
    config["data"]["dataset_name"] = plans["dataset_name"]
    config["data"]["spacing"] = plan_config["spacing"]
    config["data"]["plan_patch_size"] = config["data"]["patch_size"] or plan_config["patch_size"]
    config["data"]["channels"] = len(dataset_json.get("channel_names", {"0": "CT"}))
    config["data"]["labels"] = dataset_json["labels"]
    config["data"]["target_label"] = resolve_target_label(dataset_json["labels"], config.get("organ"))
    config["data"]["num_classes"] = 2 if config["data"]["target_label"] is not None else len(dataset_json["labels"])

    if config["train"]["batch_size"] is None:
        config["train"]["batch_size"] = plan_config["batch_size"]

    if config["model"]["patch_size"] is None:
        config["model"]["patch_size"] = infer_model_patch_size(plan_config["spacing"])

    config["dataset_json_obj"] = dataset_json
    config["plans_obj"] = plans
    return config


def save_resolved_config(config: dict[str, Any], destination: str | Path) -> None:
    destination = resolve_path(destination)
    serializable = copy.deepcopy(config)
    serializable.pop("dataset_json_obj", None)
    serializable.pop("plans_obj", None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False)
