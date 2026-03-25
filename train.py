from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from time import perf_counter
from typing import Any

if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch

from checkpoint import load_checkpoint, save_checkpoint
from config import load_config, save_resolved_config
from dataset import BTCVPatchDataset, BTCVVolumeDataset, load_split
from engine import PolyLRScheduler, append_metrics, build_dataloader, train_one_epoch, validate_full_volumes, validate_patches
from losses import DiceCrossEntropyLoss
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config.")
    parser.add_argument("--fold", type=int, default=None, help="BTCV fold index.")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from.")
    parser.add_argument("--device", default=None, help="Override device from config.")
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


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def run_training(
    config: dict[str, Any],
    fold: int = 0,
    resume: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    seed = int(config["seed"])
    set_seed(seed)
    device = resolve_device(config)

    train_ids, val_ids = load_split(config["data"]["split_file"], fold)
    preprocessed_dir = config["data"]["preprocessed_dir"]
    output_dir = Path(output_dir) if output_dir is not None else Path(config["output_dir"]) / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = BTCVPatchDataset(
        preprocessed_dir,
        train_ids,
        patch_size=config["data"]["plan_patch_size"],
        samples_per_epoch=config["data"]["train_samples_per_epoch"],
        oversample_foreground_percent=config["data"]["oversample_foreground_percent"],
        cache_data=config["data"]["cache_data"],
        validate_mode=False,
        seed=seed,
        target_label=config["data"].get("target_label"),
    )
    val_patch_dataset = BTCVPatchDataset(
        preprocessed_dir,
        val_ids,
        patch_size=config["data"]["plan_patch_size"],
        samples_per_epoch=config["data"]["val_samples_per_epoch"],
        oversample_foreground_percent=0.0,
        cache_data=config["data"]["cache_data"],
        validate_mode=True,
        seed=seed + 1,
        target_label=config["data"].get("target_label"),
    )
    val_volume_dataset = BTCVVolumeDataset(
        preprocessed_dir,
        val_ids,
        cache_data=config["data"]["cache_data"],
        target_label=config["data"].get("target_label"),
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        persistent_workers=config["data"]["persistent_workers"],
        shuffle=True,
        seed=seed,
    )
    val_patch_loader = build_dataloader(
        val_patch_dataset,
        batch_size=config["train"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        persistent_workers=config["data"]["persistent_workers"],
        shuffle=False,
        seed=seed + 1,
    )
    val_volume_loader = build_dataloader(
        val_volume_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        shuffle=False,
        seed=seed + 2,
    )

    model = build_model(config["model"], config["data"]["channels"], config["data"]["num_classes"]).to(device)
    params_total, trainable_params = count_parameters(model)
    loss_fn = DiceCrossEntropyLoss(num_classes=config["data"]["num_classes"])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
        momentum=config["train"]["momentum"],
        nesterov=True,
    )
    scheduler = PolyLRScheduler(optimizer, config["train"]["lr"], config["train"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=config["train"]["amp"] and device.type == "cuda")

    start_epoch = 0
    best_epoch = 0
    best_metric = -1.0
    if resume:
        checkpoint = load_checkpoint(resume, model, optimizer, scheduler, scaler, device=device)
        start_epoch = int(checkpoint.get("epoch", 0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_metric = float(checkpoint.get("best_metric", -1.0))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    metrics_log = output_dir / "train_metrics.jsonl"
    last_checkpoint_path = checkpoints_dir / "last.pt"
    best_checkpoint_path = checkpoints_dir / "best.pt"

    train_loss_last = None
    val_loss_last = None
    val_dice = None
    infer_time_ms_per_case = None
    best_metric_name = "val_dice" if config["validation"]["use_full_volume"] else "pseudo_dice"
    epoch_times = []
    train_start_time = perf_counter()
    train_epochs_completed = start_epoch

    for epoch in range(start_epoch, config["train"]["epochs"]):
        epoch_start_time = perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            scaler,
            device,
            grad_clip=config["train"]["grad_clip"],
            amp_enabled=config["train"]["amp"],
        )
        val_patch_metrics = validate_patches(
            model,
            val_patch_loader,
            loss_fn,
            device,
            num_classes=config["data"]["num_classes"],
            amp_enabled=config["train"]["amp"],
        )

        scheduler.step()

        full_summary = None
        should_run_full = config["validation"]["use_full_volume"] and (
            (epoch + 1) % config["train"]["full_validation_every"] == 0 or epoch == config["train"]["epochs"] - 1
        )
        current_metric = float(val_patch_metrics["pseudo_dice"])
        if not config["validation"]["use_full_volume"]:
            val_dice = current_metric

        should_update_best = not config["validation"]["use_full_volume"]
        if should_run_full:
            full_summary = validate_full_volumes(
                model,
                val_volume_loader,
                device,
                num_classes=config["data"]["num_classes"],
                patch_size=config["data"]["plan_patch_size"],
                overlap=config["validation"]["overlap"],
                output_dir=output_dir / "validation",
            )
            current_metric = float(full_summary["foreground_mean"]["Dice"])
            val_dice = current_metric
            infer_time_ms_per_case = float(full_summary.get("infer_time_ms_per_case", 0.0))
            should_update_best = True

        train_loss_last = float(train_metrics["loss"])
        val_loss_last = float(val_patch_metrics["loss"])
        train_epochs_completed = epoch + 1
        epoch_time_sec = perf_counter() - epoch_start_time
        epoch_times.append(epoch_time_sec)

        payload = {
            "epoch": epoch + 1,
            "train_loss": train_loss_last,
            "val_loss": val_loss_last,
            "pseudo_dice": float(val_patch_metrics["pseudo_dice"]),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time_sec,
        }
        if full_summary is not None:
            payload["val_dice"] = float(full_summary["foreground_mean"]["Dice"])
            payload["infer_time_ms_per_case"] = infer_time_ms_per_case
        append_metrics(metrics_log, payload)

        if should_update_best and current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, scaler, epoch + 1, best_metric, best_epoch, config)
        save_checkpoint(last_checkpoint_path, model, optimizer, scheduler, scaler, epoch + 1, best_metric, best_epoch, config)

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    best_metric_value = best_metric if best_metric >= 0.0 else None
    return {
        "best_epoch": best_epoch,
        "train_epochs_completed": train_epochs_completed,
        "val_dice": val_dice,
        "val_loss": val_loss_last,
        "train_loss_last": train_loss_last,
        "params_total": params_total,
        "trainable_params": trainable_params,
        "flops_g": None,
        "macs_g": None,
        "peak_vram_mb": peak_vram_mb,
        "infer_time_ms_per_case": infer_time_ms_per_case,
        "elapsed_time_sec": perf_counter() - train_start_time,
        "avg_epoch_time_sec": float(np.mean(epoch_times)) if epoch_times else 0.0,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "ckpt_best_path": str(best_checkpoint_path),
        "ckpt_last_path": str(last_checkpoint_path),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device

    fold = args.fold if args.fold is not None else int(config.get("split_id", 0))
    output_dir = Path(config["output_dir"]) / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir / "resolved_config.yaml")
    run_training(config, fold=fold, resume=args.resume, output_dir=output_dir)


if __name__ == "__main__":
    main()
