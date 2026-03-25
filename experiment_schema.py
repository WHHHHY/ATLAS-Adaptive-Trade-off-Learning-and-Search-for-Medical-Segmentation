from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "experiment_schema_v1"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_INVALID_TRIAL = "invalid_trial"
VALID_STATUSES = {STATUS_SUCCESS, STATUS_FAILED, STATUS_INVALID_TRIAL}


def now_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_quantization(config: dict[str, Any]) -> dict[str, Any]:
    quantization = config.get("quantization") or {}
    return {
        "enabled": quantization.get("enabled"),
        "mode": quantization.get("mode"),
    }


def get_dataset_name(config: dict[str, Any]) -> str | None:
    data_config = config.get("data") or {}
    dataset_name = data_config.get("dataset_name")
    if dataset_name:
        return str(dataset_name)
    root = data_config.get("root")
    if root:
        return Path(str(root)).name
    return None


def get_search_method(config: dict[str, Any]) -> str:
    runtime = config.get("runtime") or {}
    return str(runtime.get("search_method") or config.get("search_method") or "manual")


def get_parent_run_id(config: dict[str, Any]) -> str | None:
    runtime = config.get("runtime") or {}
    parent_run_id = runtime.get("parent_run_id")
    if parent_run_id is not None:
        return str(parent_run_id)
    parent_run_id = config.get("parent_run_id")
    if parent_run_id is None:
        return None
    return str(parent_run_id)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _params_to_millions(params_total: Any) -> float | None:
    if params_total is None:
        return None
    return float(params_total) / 1_000_000.0


def classify_status_for_phase(exception: Exception, phase: str | None) -> str:
    if phase in {"config_load", "preflight_validation"}:
        return STATUS_INVALID_TRIAL
    return STATUS_FAILED


def format_error(exception: Exception) -> str:
    return f"{type(exception).__name__}: {exception}"


def validate_candidate_config(config: dict[str, Any]) -> None:
    train_config = config.get("train") or {}
    data_config = config.get("data") or {}
    validation_config = config.get("validation") or {}

    epochs = int(train_config.get("epochs", 0))
    if epochs <= 0:
        raise ValueError("train.epochs must be > 0")

    batch_size = int(train_config.get("batch_size", 0))
    if batch_size <= 0:
        raise ValueError("train.batch_size must be > 0")

    patch_size = data_config.get("plan_patch_size")
    if not isinstance(patch_size, list) or not patch_size or any(int(size) <= 0 for size in patch_size):
        raise ValueError("data.plan_patch_size must be a non-empty list of positive integers")

    overlap = float(validation_config.get("overlap", 0.0))
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError("validation.overlap must be in [0, 1)")


def build_default_metrics(config: dict[str, Any], run_id: str, run_dir: Path, config_path: str) -> dict[str, Any]:
    checkpoints_dir = run_dir / "checkpoints"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp": now_timestamp(),
        "organ": config.get("organ"),
        "dataset": get_dataset_name(config),
        "search_method": get_search_method(config),
        "status": STATUS_FAILED,
        "seed": int(config.get("seed", 0)),
        "parent_run_id": get_parent_run_id(config),
        "val_dice": None,
        "latency_ms": None,
        "params_m": None,
        "peak_vram_mb": None,
        "train_time_s": None,
        "utility": None,
        "failure_stage": None,
        "best_epoch": None,
        "train_epochs_completed": 0,
        "val_loss": None,
        "train_loss_last": None,
        "save_dir": str(run_dir),
        "config_path": config_path,
        "ckpt_best_path": str(checkpoints_dir / "best.pt"),
        "ckpt_last_path": str(checkpoints_dir / "last.pt"),
        "elapsed_time_sec": None,
        "params_total": None,
        "trainable_params": None,
        "avg_epoch_time_sec": None,
        "pruning_ratio": config.get("pruning_ratio"),
        "quantization": normalize_quantization(config),
        "error": None,
        "infer_time_ms_per_case": None,
        "best_metric_name": "val_dice",
        "best_metric_value": None,
        "flops_g": None,
        "macs_g": None,
        "summary": {},
        "metrics": {},
        "timing": {},
        "resources": {},
        "artifacts": {},
        "context": {},
        "error_info": None,
    }


def enrich_metrics_record(metrics: dict[str, Any]) -> dict[str, Any]:
    latency_ms = _as_float(metrics.get("latency_ms"))
    if latency_ms is None:
        latency_ms = _as_float(metrics.get("infer_time_ms_per_case"))
    metrics["latency_ms"] = latency_ms

    train_time_s = _as_float(metrics.get("train_time_s"))
    if train_time_s is None:
        train_time_s = _as_float(metrics.get("elapsed_time_sec"))
    metrics["train_time_s"] = train_time_s

    params_m = _as_float(metrics.get("params_m"))
    if params_m is None:
        params_m = _params_to_millions(metrics.get("params_total"))
    metrics["params_m"] = params_m

    utility = _as_float(metrics.get("utility"))
    if utility is None and metrics.get("status") == STATUS_SUCCESS and metrics.get("val_dice") is not None:
        utility = float(metrics["val_dice"])
    metrics["utility"] = utility

    status = metrics.get("status")
    if status not in VALID_STATUSES:
        metrics["status"] = STATUS_FAILED

    error_message = metrics.get("error")
    error_type = None
    if error_message:
        error_type = str(error_message).split(":", 1)[0]

    metrics["summary"] = {
        "run_id": metrics.get("run_id"),
        "timestamp": metrics.get("timestamp"),
        "organ": metrics.get("organ"),
        "dataset": metrics.get("dataset"),
        "search_method": metrics.get("search_method"),
        "status": metrics.get("status"),
        "seed": _as_int(metrics.get("seed")),
        "parent_run_id": metrics.get("parent_run_id"),
        "val_dice": _as_float(metrics.get("val_dice")),
        "latency_ms": latency_ms,
        "params_m": params_m,
        "peak_vram_mb": _as_float(metrics.get("peak_vram_mb")),
        "train_time_s": train_time_s,
        "utility": utility,
    }
    metrics["metrics"] = {
        "val_dice": _as_float(metrics.get("val_dice")),
        "val_loss": _as_float(metrics.get("val_loss")),
        "train_loss_last": _as_float(metrics.get("train_loss_last")),
        "best_metric_name": metrics.get("best_metric_name"),
        "best_metric_value": _as_float(metrics.get("best_metric_value")),
        "latency_ms": latency_ms,
        "utility": utility,
    }
    metrics["timing"] = {
        "timestamp": metrics.get("timestamp"),
        "train_time_s": train_time_s,
        "avg_epoch_time_s": _as_float(metrics.get("avg_epoch_time_sec")),
        "latency_ms": latency_ms,
    }
    metrics["resources"] = {
        "params_total": _as_int(metrics.get("params_total")),
        "trainable_params": _as_int(metrics.get("trainable_params")),
        "params_m": params_m,
        "peak_vram_mb": _as_float(metrics.get("peak_vram_mb")),
        "flops_g": _as_float(metrics.get("flops_g")),
        "macs_g": _as_float(metrics.get("macs_g")),
    }
    metrics["artifacts"] = {
        "run_dir": metrics.get("save_dir"),
        "config_path": metrics.get("config_path"),
        "ckpt_best_path": metrics.get("ckpt_best_path"),
        "ckpt_last_path": metrics.get("ckpt_last_path"),
    }
    metrics["context"] = {
        "pruning_ratio": _as_float(metrics.get("pruning_ratio")),
        "quantization": metrics.get("quantization"),
        "best_epoch": _as_int(metrics.get("best_epoch")),
        "train_epochs_completed": _as_int(metrics.get("train_epochs_completed")),
        "failure_stage": metrics.get("failure_stage"),
    }
    metrics["error_info"] = None
    if error_message:
        metrics["error_info"] = {
            "type": error_type,
            "message": error_message,
            "stage": metrics.get("failure_stage"),
        }
    return metrics


def build_history_entry(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": metrics.get("run_id"),
        "timestamp": metrics.get("timestamp"),
        "organ": metrics.get("organ"),
        "dataset": metrics.get("dataset"),
        "search_method": metrics.get("search_method"),
        "status": metrics.get("status"),
        "seed": _as_int(metrics.get("seed")),
        "parent_run_id": metrics.get("parent_run_id"),
        "val_dice": _as_float(metrics.get("val_dice")),
        "latency_ms": _as_float(metrics.get("latency_ms")),
        "params_m": _as_float(metrics.get("params_m")),
        "peak_vram_mb": _as_float(metrics.get("peak_vram_mb")),
        "train_time_s": _as_float(metrics.get("train_time_s")),
        "utility": _as_float(metrics.get("utility")),
        "failure_stage": metrics.get("failure_stage"),
        "error": metrics.get("error"),
        "best_epoch": _as_int(metrics.get("best_epoch")),
        "train_epochs_completed": _as_int(metrics.get("train_epochs_completed")),
        "save_dir": metrics.get("save_dir"),
        "config_path": metrics.get("config_path"),
    }