from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


_PROTECTED_BLOCK_ALIASES = {
    "patch_embedding": "patch_embed",
    "patch_embed": "patch_embed",
    "seg_out": "seg_head",
    "seg_head": "seg_head",
}
_VALID_ALLOWED_ACTIONS = {
    "observe",
    "score",
    "prune",
    "quantize",
    "keep",
    "expand",
}
_VALID_STRATEGIES = {"conservative", "greedy_prune", "greedy_quant", "mixed"}
_VALID_ALLOCATOR_MODES = {"conservative", "balanced", "aggressive"}


@dataclass(slots=True)
class CalibrationConfig:
    num_samples: int = 8
    split: str = "val"
    fold: int = 0
    batch_size: int = 1
    seed: int = 1234
    max_spatial_tokens: int = 64


@dataclass(slots=True)
class ObjectiveConfig:
    similarity: float = 1.0
    risk: float = 1.0
    alpha: float = 0.5
    beta: float = 0.25
    gamma: float = 0.25
    encoder_weight: float = 0.4
    skip_weight: float = 0.2
    decoder_weight: float = 0.4
    lambda_dice: float = 1.0
    lambda_param: float = 0.05
    lambda_flops: float = 0.03
    lambda_ram: float = 0.08
    lambda_latency: float = 0.02


@dataclass(slots=True)
class ThresholdConfig:
    score_percentile: float = 75.0
    risk_percentile: float = 75.0
    min_similarity: float | None = None
    max_risk: float | None = None
    tau_prune: float = 0.75
    tau_quant: float = 0.40
    tau_low: float = 0.30
    tau_high: float = 0.80
    max_prune_ratio: float = 0.5
    eta_expand: float = 0.05


@dataclass(slots=True)
class SearchConfig:
    max_trials: int = 3
    quick_epochs: int = 3
    epsilon_accept: float = 1e-4
    strategy: str = "conservative"
    enable_expand: bool = False
    max_units_per_trial: int = 1


@dataclass(slots=True)
class QuantizationConfig:
    weight_bits: list[int] = field(default_factory=lambda: [4, 8, 16])
    act_bits: list[int] = field(default_factory=lambda: [8, 16, 32])
    allocator_mode: str = "conservative"


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "cuda"
    git_enabled: bool = False
    git_repo_root: str | None = None
    git_remote: str = "origin"
    git_branch: str | None = None
    git_commit_on_accept: bool = False
    git_push_on_accept: bool = False
    git_tag_on_accept: bool = False
    git_rollback_on_reject: bool = True


@dataclass(slots=True)
class ProgramConfig:
    organ: str
    dataset_root: str
    base_model_ckpt: str
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    freeze_blocks: list[str] = field(default_factory=lambda: ["patch_embed", "seg_head"])
    allowed_actions: list[str] = field(default_factory=lambda: ["observe", "score", "keep", "prune", "quantize"])
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    search_thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_mapping(payload: Any, field_name: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return payload


def _normalize_freeze_blocks(blocks: list[Any] | None) -> list[str]:
    normalized = []
    for raw_block in blocks or []:
        block = str(raw_block).strip()
        if not block:
            continue
        normalized.append(_PROTECTED_BLOCK_ALIASES.get(block, block))
    for protected in ("patch_embed", "seg_head"):
        if protected not in normalized:
            normalized.append(protected)
    return normalized


def _validate_allowed_actions(actions: list[Any] | None, freeze_blocks: list[str]) -> list[str]:
    normalized = [str(action).strip() for action in (actions or []) if str(action).strip()]
    if not normalized:
        normalized = ["observe", "score", "keep", "prune", "quantize"]
    invalid = sorted({action for action in normalized if action not in _VALID_ALLOWED_ACTIONS})
    if invalid:
        raise ValueError(f"Unsupported allowed_actions: {invalid}")
    if any(action in {"prune", "quantize", "expand"} for action in normalized):
        for protected in ("patch_embed", "seg_head"):
            if protected not in freeze_blocks:
                raise ValueError(f"{protected} must remain frozen when mutable actions are enabled.")
    return normalized


def _validate_calibration(config: CalibrationConfig) -> CalibrationConfig:
    if config.num_samples <= 0:
        raise ValueError("calibration.num_samples must be > 0")
    if config.split not in {"train", "val"}:
        raise ValueError("calibration.split must be either 'train' or 'val'")
    if config.batch_size <= 0:
        raise ValueError("calibration.batch_size must be > 0")
    if config.max_spatial_tokens <= 0:
        raise ValueError("calibration.max_spatial_tokens must be > 0")
    return config


def _validate_objective(config: ObjectiveConfig) -> ObjectiveConfig:
    if config.alpha < 0.0 or config.beta < 0.0 or config.gamma < 0.0:
        raise ValueError("objective alpha/beta/gamma must be >= 0")
    for field_name in ("lambda_dice", "lambda_param", "lambda_flops", "lambda_ram", "lambda_latency"):
        if getattr(config, field_name) < 0.0:
            raise ValueError(f"objective.{field_name} must be >= 0")
    weight_sum = config.encoder_weight + config.skip_weight + config.decoder_weight
    if weight_sum <= 0.0:
        raise ValueError("objective encoder/skip/decoder weights must sum to > 0")
    return config


def _validate_thresholds(config: ThresholdConfig) -> ThresholdConfig:
    for field_name in ("score_percentile", "risk_percentile", "tau_prune", "tau_quant", "tau_low", "tau_high"):
        value = getattr(config, field_name)
        if value < 0.0 or value > 1.0 and field_name not in {"score_percentile", "risk_percentile"}:
            raise ValueError(f"search_thresholds.{field_name} must be in [0, 1]")
    for field_name in ("score_percentile", "risk_percentile"):
        value = getattr(config, field_name)
        if value < 0.0 or value > 100.0:
            raise ValueError(f"search_thresholds.{field_name} must be in [0, 100]")
    if config.max_prune_ratio < 0.0 or config.max_prune_ratio > 1.0:
        raise ValueError("search_thresholds.max_prune_ratio must be in [0, 1]")
    return config


def _validate_search(config: SearchConfig) -> SearchConfig:
    if config.max_trials <= 0:
        raise ValueError("search.max_trials must be > 0")
    if config.quick_epochs < 0:
        raise ValueError("search.quick_epochs must be >= 0")
    if config.epsilon_accept < 0.0:
        raise ValueError("search.epsilon_accept must be >= 0")
    if config.strategy not in _VALID_STRATEGIES:
        raise ValueError(f"search.strategy must be one of {_VALID_STRATEGIES}")
    if config.max_units_per_trial <= 0:
        raise ValueError("search.max_units_per_trial must be > 0")
    return config


def _validate_quantization(config: QuantizationConfig) -> QuantizationConfig:
    if config.allocator_mode not in _VALID_ALLOCATOR_MODES:
        raise ValueError(f"quantization.allocator_mode must be one of {_VALID_ALLOCATOR_MODES}")
    if not config.weight_bits or not config.act_bits:
        raise ValueError("quantization.weight_bits and quantization.act_bits must be non-empty")
    config.weight_bits = sorted({int(bit) for bit in config.weight_bits})
    config.act_bits = sorted({int(bit) for bit in config.act_bits})
    if any(bit <= 0 for bit in config.weight_bits + config.act_bits):
        raise ValueError("quantization bit candidates must be > 0")
    return config


def _build_program_config(payload: dict[str, Any]) -> ProgramConfig:
    organ = str(payload.get("organ") or "").strip()
    dataset_root = str(payload.get("dataset_root") or "").strip()
    base_model_ckpt = str(payload.get("base_model_ckpt") or "").strip()
    if not organ:
        raise ValueError("organ is required")
    if not dataset_root:
        raise ValueError("dataset_root is required")
    if not base_model_ckpt:
        raise ValueError("base_model_ckpt is required")

    calibration = _validate_calibration(CalibrationConfig(**_as_mapping(payload.get("calibration"), "calibration")))
    objective = _validate_objective(ObjectiveConfig(**_as_mapping(payload.get("objective"), "objective")))
    thresholds = _validate_thresholds(ThresholdConfig(**_as_mapping(payload.get("search_thresholds"), "search_thresholds")))
    search = _validate_search(SearchConfig(**_as_mapping(payload.get("search"), "search")))
    quantization = _validate_quantization(QuantizationConfig(**_as_mapping(payload.get("quantization"), "quantization")))
    runtime_payload = dict(_as_mapping(payload.get("runtime"), "runtime"))
    git_payload = _as_mapping(runtime_payload.pop("git", None), "runtime.git")
    if git_payload:
        if "enabled" in git_payload:
            runtime_payload["git_enabled"] = git_payload["enabled"]
        if "repo_root" in git_payload:
            runtime_payload["git_repo_root"] = git_payload["repo_root"]
        if "remote" in git_payload:
            runtime_payload["git_remote"] = git_payload["remote"]
        if "branch" in git_payload:
            runtime_payload["git_branch"] = git_payload["branch"]
        if "commit_on_accept" in git_payload:
            runtime_payload["git_commit_on_accept"] = git_payload["commit_on_accept"]
        if "push_on_accept" in git_payload:
            runtime_payload["git_push_on_accept"] = git_payload["push_on_accept"]
        if "tag_on_accept" in git_payload:
            runtime_payload["git_tag_on_accept"] = git_payload["tag_on_accept"]
        if "rollback_on_reject" in git_payload:
            runtime_payload["git_rollback_on_reject"] = git_payload["rollback_on_reject"]
    if "git_commit_on_improve" in runtime_payload and "git_commit_on_accept" not in runtime_payload:
        runtime_payload["git_commit_on_accept"] = runtime_payload["git_commit_on_improve"]
    runtime_payload.pop("git_commit_on_improve", None)
    runtime = RuntimeConfig(**runtime_payload)
    freeze_blocks = _normalize_freeze_blocks(payload.get("freeze_blocks"))
    allowed_actions = _validate_allowed_actions(payload.get("allowed_actions"), freeze_blocks)
    return ProgramConfig(
        organ=organ,
        dataset_root=dataset_root,
        base_model_ckpt=base_model_ckpt,
        calibration=calibration,
        freeze_blocks=freeze_blocks,
        allowed_actions=allowed_actions,
        objective=objective,
        search_thresholds=thresholds,
        search=search,
        quantization=quantization,
        runtime=runtime,
        output_path=payload.get("output_path"),
    )


def load_program_config(program_path: str | Path) -> ProgramConfig:
    program_path = Path(program_path).resolve()
    with program_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Program YAML must contain a top-level mapping.")
    program = _build_program_config(payload)
    dataset_root = Path(program.dataset_root).expanduser().resolve()
    checkpoint_path = Path(program.base_model_ckpt).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"base_model_ckpt not found: {checkpoint_path}")
    program.dataset_root = str(dataset_root)
    program.base_model_ckpt = str(checkpoint_path)
    if program.output_path is not None:
        program.output_path = str(Path(program.output_path).expanduser().resolve())
    if program.runtime.git_repo_root is not None:
        program.runtime.git_repo_root = str(Path(program.runtime.git_repo_root).expanduser().resolve())
    return program