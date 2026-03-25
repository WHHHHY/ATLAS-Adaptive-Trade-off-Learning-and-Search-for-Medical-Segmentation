from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"

import torch
import numpy as np

from checkpoint import load_checkpoint
from config import load_config
from dataset import BTCVVolumeDataset, load_split
from engine import build_dataloader, validate_full_volumes
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config.")
    parser.add_argument("--fold", type=int, default=None, help="BTCV fold index.")
    parser.add_argument("--ckpt", "--checkpoint", dest="checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--device", default=None, help="Override device.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested_device = config["device"]
    if requested_device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(requested_device)
    return torch.device("cpu")


def run_validation(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    fold: int = 0,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(int(config["seed"]))
    device = resolve_device(config)

    _, val_ids = load_split(config["data"]["split_file"], fold)
    dataset = BTCVVolumeDataset(
        config["data"]["preprocessed_dir"],
        val_ids,
        cache_data=config["data"]["cache_data"],
        target_label=config["data"].get("target_label"),
    )
    loader = build_dataloader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        shuffle=False,
        seed=int(config["seed"]) + 2,
    )

    model = build_model(config["model"], config["data"]["channels"], config["data"]["num_classes"]).to(device)
    load_checkpoint(checkpoint_path, model, device=device)

    output_dir = Path(output_dir) if output_dir is not None else Path(config["output_dir"]) / f"fold_{fold}" / "validation"
    return validate_full_volumes(
        model,
        loader,
        device,
        num_classes=config["data"]["num_classes"],
        patch_size=config["data"]["plan_patch_size"],
        overlap=config["validation"]["overlap"],
        output_dir=output_dir,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device

    fold = args.fold if args.fold is not None else int(config.get("split_id", 0))
    summary = run_validation(config, args.checkpoint, fold=fold)
    print(f"Mean validation Dice: {summary['foreground_mean']['Dice']:.4f}")


if __name__ == "__main__":
    main()
