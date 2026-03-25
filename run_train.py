from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"

import torch
import yaml

from config import load_config, save_resolved_config
from experiment_schema import (
    STATUS_INVALID_TRIAL,
    STATUS_SUCCESS,
    build_default_metrics,
    build_history_entry,
    classify_status_for_phase,
    enrich_metrics_record,
    format_error,
    now_timestamp,
    validate_candidate_config,
)
from train import run_training


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_output(log_path: str | Path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(original_stdout, handle)
        sys.stderr = TeeStream(original_stderr, handle)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config.")
    parser.add_argument("--fold", type=int, default=None, help="Override split_id from config.")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from.")
    parser.add_argument("--device", default=None, help="Override device from config.")
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return normalized or "run"


def build_run_id(save_root: Path, organ: str, seed: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_run_id = f"{timestamp}-{sanitize_name(organ)}-seed{seed}"
    run_id = base_run_id
    suffix = 1
    while (save_root / run_id).exists():
        run_id = f"{base_run_id}-{suffix:02d}"
        suffix += 1
    return run_id


def write_failed_config_snapshot(path: str | Path, config: dict[str, Any], error: Exception) -> None:
    payload = deepcopy(config)
    payload["status"] = STATUS_INVALID_TRIAL
    payload["error"] = f"{type(error).__name__}: {error}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_checkpoint_epoch(checkpoint_path: Path, key: str) -> int | None:
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    value = checkpoint.get(key)
    if value is None:
        return None
    return int(value)


def load_last_epoch_metrics(metrics_log_path: Path) -> dict[str, Any]:
    if not metrics_log_path.exists():
        return {}
    lines = [line for line in metrics_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def append_history(history_path: str | Path, metrics: dict[str, Any]) -> None:
    payload = build_history_entry(metrics)
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def finalize_metrics(metrics: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    metrics_log = run_dir / "train_metrics.jsonl"
    last_epoch_metrics = load_last_epoch_metrics(metrics_log)
    if metrics.get("elapsed_time_sec") is None and metrics.get("train_time_sec_total") is not None:
        metrics["elapsed_time_sec"] = float(metrics["train_time_sec_total"])
    metrics.pop("train_time_sec_total", None)
    if metrics["train_loss_last"] is None and "train_loss" in last_epoch_metrics:
        metrics["train_loss_last"] = float(last_epoch_metrics["train_loss"])
    if metrics["val_loss"] is None and "val_loss" in last_epoch_metrics:
        metrics["val_loss"] = float(last_epoch_metrics["val_loss"])
    if metrics["val_dice"] is None:
        if "val_dice" in last_epoch_metrics:
            metrics["val_dice"] = float(last_epoch_metrics["val_dice"])
        elif metrics.get("best_metric_name") != "val_dice" and "pseudo_dice" in last_epoch_metrics:
            metrics["val_dice"] = float(last_epoch_metrics["pseudo_dice"])
    if metrics["infer_time_ms_per_case"] is None and "infer_time_ms_per_case" in last_epoch_metrics:
        metrics["infer_time_ms_per_case"] = float(last_epoch_metrics["infer_time_ms_per_case"])

    last_epoch = load_checkpoint_epoch(run_dir / "checkpoints" / "last.pt", "epoch")
    if last_epoch is not None:
        metrics["train_epochs_completed"] = last_epoch
    best_epoch = load_checkpoint_epoch(run_dir / "checkpoints" / "best.pt", "best_epoch")
    if best_epoch is not None:
        metrics["best_epoch"] = best_epoch
    if metrics.get("best_metric_value") is None and metrics.get("best_metric_name") == "val_dice":
        metrics["best_metric_value"] = metrics.get("val_dice")
    return enrich_metrics_record(metrics)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_load_error = None
    preflight_error = None
    config: dict[str, Any] = {
        "organ": "unknown",
        "seed": 0,
        "split_id": 0,
        "save_dir": str(project_root / "runs"),
        "validation": {"use_full_volume": True},
        "config_path": str((project_root / args.config).resolve()),
    }

    try:
        config = load_config(args.config)
        if args.device is not None:
            config["device"] = args.device
    except Exception as exc:
        config_load_error = exc

    fold = args.fold if args.fold is not None else int(config.get("split_id", 0))
    save_root = Path(config["save_dir"])
    save_root.mkdir(parents=True, exist_ok=True)
    run_id = build_run_id(save_root, str(config.get("organ", "pancreas")), int(config["seed"]))
    run_dir = save_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    runtime_config = deepcopy(config)
    runtime_config["run_id"] = run_id
    runtime_config["split_id"] = fold
    runtime_config["output_dir"] = str(run_dir)
    runtime_context = deepcopy(runtime_config.get("runtime") or {})
    runtime_context.update({
        "run_id": run_id,
        "run_dir": str(run_dir),
        "history_path": str(save_root / "history.jsonl"),
    })
    runtime_config["runtime"] = runtime_context
    if config_load_error is None:
        save_resolved_config(runtime_config, run_dir / "config_used.yaml")
        try:
            validate_candidate_config(runtime_config)
        except Exception as exc:
            preflight_error = exc
    else:
        write_failed_config_snapshot(run_dir / "config_used.yaml", runtime_config, config_load_error)

    metrics = build_default_metrics(runtime_config, run_id, run_dir, runtime_config["config_path"])
    metrics["best_metric_name"] = "val_dice" if runtime_config["validation"]["use_full_volume"] else "pseudo_dice"

    with tee_output(run_dir / "train.log"):
        try:
            if config_load_error is not None:
                metrics["failure_stage"] = "config_load"
                raise config_load_error
            if preflight_error is not None:
                metrics["failure_stage"] = "preflight_validation"
                raise preflight_error
            summary = run_training(runtime_config, fold=fold, resume=args.resume, output_dir=run_dir)
            metrics.update(summary)
            metrics["status"] = STATUS_SUCCESS
        except Exception as exc:
            if metrics.get("failure_stage") is None:
                metrics["failure_stage"] = "training"
            metrics["status"] = classify_status_for_phase(exc, metrics.get("failure_stage"))
            metrics["error"] = format_error(exc)
            traceback.print_exc()
        finally:
            metrics = finalize_metrics(metrics, run_dir)
            metrics["timestamp"] = now_timestamp()
            metrics = enrich_metrics_record(metrics)
            try:
                write_json(run_dir / "metrics.json", metrics)
            except Exception as write_exc:
                print(f"Failed to write metrics.json: {type(write_exc).__name__}: {write_exc}", file=sys.stderr)
            try:
                append_history(save_root / "history.jsonl", metrics)
            except Exception as history_exc:
                print(f"Failed to append history.jsonl: {type(history_exc).__name__}: {history_exc}", file=sys.stderr)


if __name__ == "__main__":
    main()