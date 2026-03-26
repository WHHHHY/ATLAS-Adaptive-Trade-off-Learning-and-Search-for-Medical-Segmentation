from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from checkpoint import load_checkpoint
from config import resolve_target_label
from dataset import BTCVPatchDataset, load_split
from engine import build_dataloader
from model import build_model
from train import resolve_device, set_seed

from .metric_tools import adaptive_activation_map, adaptive_token_features, entropy_summary, linear_cka, rms_kurtosis
from .program_schema import ProgramConfig, load_program_config
from .unit_aggregation import aggregate_unit_metrics


def _default_metrics_output(program_path: str | Path) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "runs" / "search_artifacts" / "metrics" / f"{Path(program_path).stem}_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True, help="Path to the program YAML.")
    parser.add_argument("--output", default=None, help="Optional JSON destination override.")
    parser.add_argument("--device", default=None, help="Optional device override.")
    return parser.parse_args()


def _load_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _derive_runtime_config(program: ProgramConfig, payload: dict[str, Any], device_override: str | None) -> dict[str, Any]:
    config_from_ckpt = deepcopy(payload.get("config") or {})
    if not config_from_ckpt:
        raise ValueError("Checkpoint does not contain a serialized config.")
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
    runtime_config["device"] = device_override or program.runtime.device
    return runtime_config


class ActivationCollector:
    def __init__(self, max_spatial_tokens: int):
        self.max_spatial_tokens = max_spatial_tokens
        self.records: dict[str, dict[str, list[torch.Tensor] | list[list[int]]]] = {}
        self.handles = []

    def _capture(self, name: str, tensor: torch.Tensor, bucket: str) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        record = self.records.setdefault(
            name,
            {
                "input": [],
                "output": [],
                "input_maps": [],
                "output_maps": [],
                "input_shapes": [],
                "output_shapes": [],
            },
        )
        detached = tensor.detach().cpu()
        record[bucket].append(adaptive_token_features(detached, max_spatial_tokens=self.max_spatial_tokens))
        record["input_maps" if bucket == "input" else "output_maps"].append(adaptive_activation_map(detached, max_spatial_tokens=self.max_spatial_tokens))
        record["input_shapes" if bucket == "input" else "output_shapes"].append(list(tensor.shape))

    def register(self, model: torch.nn.Module) -> None:
        modules = {
            "stage1_enc": model.encoder1,
            "stage1_skip": model.encoder1.skip_conv,
            "stage1_dec": model.decoder1,
            "stage2_enc": model.encoder2,
            "stage2_skip": model.encoder2.skip_conv,
            "stage2_dec": model.decoder2,
            "stage3_enc": model.encoder3,
            "stage3_skip": model.encoder3.skip_conv,
            "stage3_dec": model.decoder3,
        }
        for name, module in modules.items():
            self.handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
            self.handles.append(module.register_forward_hook(self._make_post_hook(name)))

    def _make_pre_hook(self, name: str):
        def hook(_module, inputs):
            if inputs:
                self._capture(name, inputs[0], "input")
        return hook

    def _make_post_hook(self, name: str):
        def hook(_module, _inputs, output):
            primary_output = output[0] if isinstance(output, tuple) else output
            self._capture(name, primary_output, "output")
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _summarize_record(record: dict[str, Any]) -> dict[str, Any]:
    inputs = torch.cat(record["input"], dim=0) if record["input"] else torch.zeros((0, 1), dtype=torch.float32)
    outputs = torch.cat(record["output"], dim=0) if record["output"] else torch.zeros((0, 1), dtype=torch.float32)
    count = min(inputs.shape[0], outputs.shape[0])
    cka_value = linear_cka(inputs[:count], outputs[:count]) if count > 0 else 0.0
    output_maps = torch.cat(record["output_maps"], dim=0) if record["output_maps"] else torch.zeros((1, 1, 1), dtype=torch.float32)
    kurtosis = rms_kurtosis(output_maps)
    entropy = entropy_summary(output_maps)
    return {
        "cka": cka_value,
        "kurtosis": float(kurtosis["kurtosis"]),
        "rms_mean": float(kurtosis["rms_mean"]),
        "rms_std": float(kurtosis["rms_std"]),
        "entropy_summary": entropy,
        "input_shapes": record["input_shapes"],
        "output_shapes": record["output_shapes"],
        "num_observations": count,
    }


def collect_activation_metrics(program_path: str | Path, output_override: str | None = None, device_override: str | None = None) -> Path:
    program = load_program_config(program_path)
    runtime_config = _derive_runtime_config(program, _load_checkpoint_payload(program.base_model_ckpt), device_override)
    set_seed(program.calibration.seed)
    device = resolve_device(runtime_config)
    _, val_ids = load_split(runtime_config["data"]["split_file"], program.calibration.fold)
    case_ids = val_ids if program.calibration.split == "val" else load_split(runtime_config["data"]["split_file"], program.calibration.fold)[0]
    dataset = BTCVPatchDataset(
        runtime_config["data"]["preprocessed_dir"],
        case_ids,
        patch_size=runtime_config["data"]["plan_patch_size"],
        samples_per_epoch=program.calibration.num_samples,
        oversample_foreground_percent=0.0,
        cache_data=runtime_config["data"]["cache_data"],
        validate_mode=True,
        seed=program.calibration.seed,
        target_label=runtime_config["data"].get("target_label"),
    )
    loader = build_dataloader(
        dataset,
        batch_size=program.calibration.batch_size,
        num_workers=runtime_config["data"]["num_workers"],
        pin_memory=runtime_config["data"]["pin_memory"],
        persistent_workers=runtime_config["data"]["persistent_workers"],
        shuffle=False,
        seed=program.calibration.seed,
    )
    model = build_model(runtime_config["model"], runtime_config["data"]["channels"], runtime_config["data"]["num_classes"]).to(device)
    load_checkpoint(program.base_model_ckpt, model, device=device)
    model.eval()
    collector = ActivationCollector(program.calibration.max_spatial_tokens)
    collector.register(model)
    sampled_case_ids = []
    with torch.no_grad():
        for batch in loader:
            sampled_case_ids.extend(batch["case_id"])
            _ = model(batch["image"].to(device, non_blocking=True))
    collector.close()

    per_submodule = {name: _summarize_record(record) for name, record in sorted(collector.records.items())}
    aggregated = aggregate_unit_metrics(per_submodule, program.objective, program.search_thresholds)
    output_path = Path(output_override).resolve() if output_override is not None else Path(program.output_path or _default_metrics_output(program_path)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "program": program.to_dict(),
        "calibration_meta": {
            "program_path": str(Path(program_path).resolve()),
            "base_model_ckpt": program.base_model_ckpt,
            "device": str(device),
            "dataset_root": program.dataset_root,
            "organ": program.organ,
            "split": program.calibration.split,
            "fold": program.calibration.fold,
            "num_samples": program.calibration.num_samples,
            "batch_size": program.calibration.batch_size,
            "seed": program.calibration.seed,
            "max_spatial_tokens": program.calibration.max_spatial_tokens,
            "sampled_case_ids": sampled_case_ids[: program.calibration.num_samples],
            "freeze_blocks": program.freeze_blocks,
            "allowed_actions": program.allowed_actions,
            "protected_blocks": ["patch_embed", "seg_head"],
        },
        "per_submodule_metrics": aggregated["per_submodule"],
        "per_unit_metrics": aggregated["per_unit"],
        "aggregation_summary": aggregated["summary"],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return output_path


def main() -> None:
    args = parse_args()
    output_path = collect_activation_metrics(args.program, output_override=args.output, device_override=args.device)
    print(json.dumps({"metrics_path": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()