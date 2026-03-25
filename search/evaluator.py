from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from checkpoint import save_checkpoint
from config import resolve_target_label, save_resolved_config
from dataset import BTCVPatchDataset, load_split
from engine import PolyLRScheduler, build_dataloader, train_one_epoch, validate_patches
from losses import DiceCrossEntropyLoss
from model import build_model
from train import resolve_device, set_seed

from .candidate_schema import Candidate, load_candidate
from .program_schema import ProgramConfig, load_program_config
from .utils.profile_model import profile_model_resources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="Path to candidate JSON/YAML.")
    parser.add_argument("--program", required=True, help="Path to program YAML.")
    parser.add_argument("--run-dir", required=True, help="Trial output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Use dry-run evaluator instead of quick fine-tune.")
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _build_runtime_config(program: ProgramConfig, candidate: Candidate) -> dict[str, Any]:
    checkpoint_payload = _load_checkpoint_payload(candidate.base_model_ckpt)
    config_from_ckpt = deepcopy(checkpoint_payload.get("config") or {})
    if not config_from_ckpt:
        raise ValueError("Candidate base checkpoint does not contain a serialized config.")
    runtime_config = config_from_ckpt
    runtime_config["organ"] = program.organ
    runtime_config["data"]["root"] = program.dataset_root
    runtime_config["data"]["dataset_json"] = str(Path(program.dataset_root) / "dataset.json")
    runtime_config["data"]["plans_file"] = str(Path(program.dataset_root) / "nnUNetPlans.json")
    runtime_config["data"]["split_file"] = str(Path(program.dataset_root) / "splits_final.json")
    runtime_config["data"]["preprocessed_dir"] = str(Path(program.dataset_root) / runtime_config["data"]["data_identifier"])
    with Path(runtime_config["data"]["dataset_json"]).open("r", encoding="utf-8") as handle:
        dataset_json = json.load(handle)
    runtime_config["data"]["labels"] = dataset_json["labels"]
    runtime_config["data"]["channels"] = len(dataset_json.get("channel_names", {"0": "CT"}))
    runtime_config["data"]["target_label"] = resolve_target_label(dataset_json["labels"], program.organ)
    runtime_config["data"]["num_classes"] = 2 if runtime_config["data"]["target_label"] is not None else len(dataset_json["labels"])
    runtime_config["device"] = program.runtime.device
    runtime_config.setdefault("runtime", {})
    runtime_config["runtime"]["search_method"] = "controlled_search_stage2"
    return runtime_config


def _parse_unit_index(unit_id: str) -> int:
    return int(unit_id.split("_")[-1]) - 1


def _default_prune_flags(num_layers: int) -> list[dict[str, bool]]:
    return [{"prune_ln2_mlp": False, "prune_ln1_token_mixer": False} for _ in range(num_layers)]


def apply_candidate_to_config(runtime_config: dict[str, Any], candidate: Candidate) -> dict[str, Any]:
    patched = deepcopy(runtime_config)
    patched.setdefault("search_candidate", {})
    patched.setdefault("quantization", {})
    patched["quantization"]["unit_bits"] = {}
    patched["search_candidate"]["unit_actions"] = [action.to_dict() for action in candidate.unit_actions]
    metaformer_layers = list(patched["model"]["metaformer_layers"])
    prune_flags_list = deepcopy(patched["model"].get("prune_flags_list"))
    if prune_flags_list is None:
        prune_flags_list = [_default_prune_flags(num_layers) for num_layers in metaformer_layers]

    for action in candidate.unit_actions:
        stage_index = _parse_unit_index(action.unit_id)
        if action.action_type == "prune":
            num_layers = int(metaformer_layers[stage_index])
            flags = prune_flags_list[stage_index] if stage_index < len(prune_flags_list) else _default_prune_flags(num_layers)
            while len(flags) < num_layers:
                flags.append({"prune_ln2_mlp": False, "prune_ln1_token_mixer": False})
            num_pruned = int(round(float(action.prune_ratio or 0.0) * num_layers))
            for flag_index in range(num_layers):
                flags[flag_index]["prune_ln2_mlp"] = flag_index >= max(0, num_layers - num_pruned)
            prune_flags_list[stage_index] = flags
        elif action.action_type == "quantize":
            patched["quantization"]["unit_bits"][action.unit_id] = {
                "weight_bits": int(action.weight_bits),
                "act_bits": int(action.act_bits),
                "allocation_reason": action.allocation_reason,
            }
        elif action.action_type == "expand":
            if action.expand_type == "width_mult":
                raise ValueError("width_mult expansion is intentionally blocked in MVP because it would change protected edge blocks")
            metaformer_layers[stage_index] = int(metaformer_layers[stage_index]) + int(action.expand_delta or 0)

    patched["model"]["metaformer_layers"] = metaformer_layers
    patched["model"]["prune_flags_list"] = prune_flags_list
    patched["model"]["quant_bits_config"] = patched["quantization"]["unit_bits"]
    return patched


def _load_model_weights_flexible(checkpoint_path: str | Path, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    return {
        "checkpoint": payload,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _build_loaders(runtime_config: dict[str, Any], program: ProgramConfig):
    train_ids, val_ids = load_split(runtime_config["data"]["split_file"], program.calibration.fold)
    train_dataset = BTCVPatchDataset(
        runtime_config["data"]["preprocessed_dir"],
        train_ids,
        patch_size=runtime_config["data"]["plan_patch_size"],
        samples_per_epoch=max(program.calibration.num_samples, runtime_config["train"]["batch_size"]),
        oversample_foreground_percent=runtime_config["data"]["oversample_foreground_percent"],
        cache_data=runtime_config["data"]["cache_data"],
        validate_mode=False,
        seed=program.calibration.seed,
        target_label=runtime_config["data"].get("target_label"),
    )
    val_dataset = BTCVPatchDataset(
        runtime_config["data"]["preprocessed_dir"],
        val_ids,
        patch_size=runtime_config["data"]["plan_patch_size"],
        samples_per_epoch=max(4, min(program.calibration.num_samples, 16)),
        oversample_foreground_percent=0.0,
        cache_data=runtime_config["data"]["cache_data"],
        validate_mode=True,
        seed=program.calibration.seed + 1,
        target_label=runtime_config["data"].get("target_label"),
    )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=runtime_config["train"]["batch_size"],
        num_workers=runtime_config["data"]["num_workers"],
        pin_memory=runtime_config["data"]["pin_memory"],
        persistent_workers=runtime_config["data"]["persistent_workers"],
        shuffle=True,
        seed=program.calibration.seed,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=runtime_config["train"]["batch_size"],
        num_workers=runtime_config["data"]["num_workers"],
        pin_memory=runtime_config["data"]["pin_memory"],
        persistent_workers=runtime_config["data"]["persistent_workers"],
        shuffle=False,
        seed=program.calibration.seed + 1,
    )
    return train_loader, val_loader


def compute_utility(metrics: dict[str, Any], baseline_metrics: dict[str, Any], objective: Any) -> float:
    val_dice = float(metrics.get("val_dice") or 0.0)
    param_ratio = max(float(metrics.get("params_effective", metrics.get("params_total", 1.0))) / max(float(baseline_metrics.get("params_effective", baseline_metrics.get("params_total", 1.0))), 1e-8), 1e-8)
    flops_ratio = max(float(metrics.get("flops_effective_g", metrics.get("flops_g", 1.0))) / max(float(baseline_metrics.get("flops_effective_g", baseline_metrics.get("flops_g", 1.0))), 1e-8), 1e-8)
    ram_ratio = max(float(metrics.get("peak_vram_effective_mb", metrics.get("peak_vram_mb", 1.0)) or 1.0) / max(float(baseline_metrics.get("peak_vram_effective_mb", baseline_metrics.get("peak_vram_mb", 1.0)) or 1.0), 1e-8), 1e-8)
    return float(
        objective.lambda_dice * val_dice
        - objective.lambda_param * torch.log(torch.tensor(param_ratio)).item()
        - objective.lambda_flops * torch.log(torch.tensor(flops_ratio)).item()
        - objective.lambda_ram * torch.log(torch.tensor(ram_ratio)).item()
    )


def _dry_run_metrics(candidate: Candidate, metrics_payload: dict[str, Any]) -> dict[str, Any]:
    per_unit = metrics_payload["per_unit_metrics"]
    base_score = sum(float(item["S_unit"]) for item in per_unit.values()) / max(len(per_unit), 1)
    base_risk = sum(float(item["R_unit"]) for item in per_unit.values()) / max(len(per_unit), 1)
    val_dice = max(0.0, min(1.0, 0.55 + 0.30 * base_score - 0.12 * base_risk))
    params_total = 1_000_000.0
    flops_g = 10.0
    peak_vram_mb = 1000.0
    for action in candidate.unit_actions:
        score = float(action.score_snapshot.S_unit) if action.score_snapshot is not None else 0.5
        risk = float(action.score_snapshot.R_unit) if action.score_snapshot is not None else 0.5
        if action.action_type == "prune":
            ratio = float(action.prune_ratio or 0.0)
            val_dice -= 0.03 * ratio * (1.0 + risk - score)
            params_total *= 1.0 - 0.20 * ratio
            flops_g *= 1.0 - 0.25 * ratio
            peak_vram_mb *= 1.0 - 0.10 * ratio
        elif action.action_type == "quantize":
            weight_ratio = float(action.weight_bits or 32) / 32.0
            act_ratio = float(action.act_bits or 32) / 32.0
            val_dice -= 0.02 * risk * (1.0 - weight_ratio * act_ratio)
            params_total *= 0.7 + 0.3 * weight_ratio
            flops_g *= 0.7 + 0.3 * ((weight_ratio + act_ratio) / 2.0)
            peak_vram_mb *= 0.7 + 0.3 * act_ratio
        elif action.action_type == "expand":
            delta = float(action.expand_delta or 0.0)
            val_dice += 0.01 * delta
            params_total *= 1.0 + 0.08 * delta
            flops_g *= 1.0 + 0.10 * delta
            peak_vram_mb *= 1.0 + 0.05 * delta
    return {
        "val_dice": max(0.0, min(1.0, val_dice)),
        "val_loss": float(max(0.0, 1.0 - val_dice)),
        "params_total": float(params_total),
        "params_effective": float(params_total),
        "flops_g": float(flops_g),
        "flops_effective_g": float(flops_g),
        "peak_vram_mb": float(peak_vram_mb),
        "peak_vram_effective_mb": float(peak_vram_mb),
        "invalid_trial": False,
        "error_message": None,
        "mode": "dry_run",
    }


def evaluate_candidate(
    candidate_path: str | Path,
    program_path: str | Path,
    run_dir: str | Path,
    dry_run: bool = False,
    baseline_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate = load_candidate(candidate_path)
    program = load_program_config(program_path)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = _load_json(candidate.source_metrics_json)
    resolved_config = apply_candidate_to_config(_build_runtime_config(program, candidate), candidate)
    resolved_config["train"]["epochs"] = int(program.search.quick_epochs)
    resolved_config["validation"]["use_full_volume"] = False
    resolved_config["runtime"]["trial_dir"] = str(run_dir)
    resolved_config_path_json = run_dir / "candidate_resolved_config.json"
    resolved_config_path_yaml = run_dir / "candidate_resolved_config.yaml"
    with resolved_config_path_json.open("w", encoding="utf-8") as handle:
        json.dump(resolved_config, handle, indent=2, ensure_ascii=False)
    save_resolved_config(resolved_config, resolved_config_path_yaml)

    result_metrics: dict[str, Any]
    checkpoint_path = None
    if dry_run:
        result_metrics = _dry_run_metrics(candidate, metrics_payload)
    else:
        set_seed(program.calibration.seed)
        device = resolve_device(resolved_config)
        train_loader, val_loader = _build_loaders(resolved_config, program)
        model = build_model(
            resolved_config["model"],
            resolved_config["data"]["channels"],
            resolved_config["data"]["num_classes"],
        ).to(device)
        load_info = _load_model_weights_flexible(candidate.base_model_ckpt, model, device)
        loss_fn = DiceCrossEntropyLoss(num_classes=resolved_config["data"]["num_classes"])
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=resolved_config["train"]["lr"],
            weight_decay=resolved_config["train"]["weight_decay"],
            momentum=resolved_config["train"]["momentum"],
            nesterov=True,
        )
        scheduler = PolyLRScheduler(optimizer, resolved_config["train"]["lr"], max(resolved_config["train"]["epochs"], 1))
        scaler = torch.cuda.amp.GradScaler(enabled=resolved_config["train"]["amp"] and device.type == "cuda")

        last_train_metrics = {"loss": 0.0}
        val_metrics = {"loss": 0.0, "pseudo_dice": 0.0}
        sample_batch = None
        if resolved_config["train"]["epochs"] > 0:
            for _epoch in range(resolved_config["train"]["epochs"]):
                last_train_metrics = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    loss_fn,
                    scaler,
                    device,
                    grad_clip=resolved_config["train"]["grad_clip"],
                    amp_enabled=resolved_config["train"]["amp"],
                )
                val_metrics = validate_patches(
                    model,
                    val_loader,
                    loss_fn,
                    device,
                    num_classes=resolved_config["data"]["num_classes"],
                    amp_enabled=resolved_config["train"]["amp"],
                )
                scheduler.step()
        for sample_batch in val_loader:
            break
        if sample_batch is None:
            raise ValueError("Validation loader produced no batches during evaluation")
        sample_input = sample_batch["image"].to(device)
        profile = profile_model_resources(model, sample_input, unit_quant_config=resolved_config["quantization"].get("unit_bits") or {}, device=device)
        checkpoint_path = run_dir / "candidate_last.pt"
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, resolved_config["train"]["epochs"], float(val_metrics["pseudo_dice"]), resolved_config["train"]["epochs"], resolved_config)
        result_metrics = {
            "val_dice": float(val_metrics["pseudo_dice"]),
            "val_loss": float(val_metrics["loss"]),
            "train_loss_last": float(last_train_metrics["loss"]),
            "params_total": int(profile["params_total"]),
            "params_effective": float(profile["params_effective"]),
            "flops_g": float(profile["flops_g"]),
            "flops_effective_g": float(profile["flops_effective_g"]),
            "peak_vram_mb": float(profile["peak_vram_mb"]),
            "peak_vram_effective_mb": float(profile["peak_vram_effective_mb"]),
            "missing_keys": load_info["missing_keys"],
            "unexpected_keys": load_info["unexpected_keys"],
            "invalid_trial": False,
            "error_message": None,
            "mode": "quick_eval",
        }

    if baseline_metrics is None:
        baseline_metrics = {
            "params_total": result_metrics.get("params_total", 1.0),
            "params_effective": result_metrics.get("params_effective", result_metrics.get("params_total", 1.0)),
            "flops_g": result_metrics.get("flops_g", 1.0),
            "flops_effective_g": result_metrics.get("flops_effective_g", result_metrics.get("flops_g", 1.0)),
            "peak_vram_mb": result_metrics.get("peak_vram_mb", 1.0),
            "peak_vram_effective_mb": result_metrics.get("peak_vram_effective_mb", result_metrics.get("peak_vram_mb", 1.0)),
        }

    result_metrics["utility"] = compute_utility(result_metrics, baseline_metrics, program.objective)
    result_metrics["invalid_trial"] = bool(result_metrics.get("invalid_trial", False))
    result_metrics["candidate_path"] = str(Path(candidate_path).resolve())
    result_metrics["resolved_config_path"] = str(resolved_config_path_json)
    result_metrics["checkpoint_path"] = str(checkpoint_path) if checkpoint_path is not None else None
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result_metrics, handle, indent=2, ensure_ascii=False)
    return {
        "metrics": result_metrics,
        "metrics_path": str(metrics_path),
        "resolved_config_path": str(resolved_config_path_json),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }


def main() -> None:
    args = parse_args()
    result = evaluate_candidate(args.candidate, args.program, args.run_dir, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()