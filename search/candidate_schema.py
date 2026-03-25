from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


_VALID_ACTION_TYPES = {"prune", "quantize", "keep", "expand"}
_PROTECTED_BLOCKS = {"patch_embed", "seg_head", "patch_embedding", "seg_out"}
_VALID_EXPAND_TYPES = {"width_mult", "num_subblocks"}


@dataclass(slots=True)
class ScoreSnapshot:
    S_unit: float
    R_unit: float
    utility_from_metrics_json: float | None = None


@dataclass(slots=True)
class UnitAction:
    unit_id: str
    action_type: str
    prune_ratio: float | None = None
    weight_bits: int | None = None
    act_bits: int | None = None
    allocation_mode: str | None = None
    allocation_reason: str | None = None
    expand_type: str | None = None
    expand_delta: float | int | None = None
    proposer_reason: str | None = None
    score_snapshot: ScoreSnapshot | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.score_snapshot is None:
            payload["score_snapshot"] = None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UnitAction":
        snapshot_payload = payload.get("score_snapshot")
        snapshot = ScoreSnapshot(**snapshot_payload) if snapshot_payload is not None else None
        return cls(
            unit_id=str(payload["unit_id"]),
            action_type=str(payload["action_type"]),
            prune_ratio=payload.get("prune_ratio"),
            weight_bits=payload.get("weight_bits"),
            act_bits=payload.get("act_bits"),
            allocation_mode=payload.get("allocation_mode"),
            allocation_reason=payload.get("allocation_reason"),
            expand_type=payload.get("expand_type"),
            expand_delta=payload.get("expand_delta"),
            proposer_reason=payload.get("proposer_reason"),
            score_snapshot=snapshot,
        )

    def validate(self, valid_unit_ids: set[str] | None = None) -> None:
        if self.unit_id in _PROTECTED_BLOCKS:
            raise ValueError(f"Protected block cannot be mutated: {self.unit_id}")
        if valid_unit_ids is not None and self.unit_id not in valid_unit_ids:
            raise ValueError(f"Unknown unit_id for action: {self.unit_id}")
        if self.action_type not in _VALID_ACTION_TYPES:
            raise ValueError(f"Unsupported action_type: {self.action_type}")
        if self.action_type == "prune":
            if self.prune_ratio is None or float(self.prune_ratio) < 0.0 or float(self.prune_ratio) > 1.0:
                raise ValueError("prune action requires prune_ratio in [0, 1]")
        if self.action_type == "quantize":
            if self.weight_bits is None or self.act_bits is None:
                raise ValueError("quantize action requires weight_bits and act_bits")
        if self.action_type == "expand":
            if self.expand_type not in _VALID_EXPAND_TYPES:
                raise ValueError(f"expand_type must be one of {_VALID_EXPAND_TYPES}")
            if self.expand_delta is None or float(self.expand_delta) <= 0.0:
                raise ValueError("expand action requires expand_delta > 0")


@dataclass(slots=True)
class Candidate:
    base_model_ckpt: str
    source_metrics_json: str
    frozen_blocks: list[str]
    unit_actions: list[UnitAction] = field(default_factory=list)
    notes: str | None = None
    proposer_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model_ckpt": self.base_model_ckpt,
            "source_metrics_json": self.source_metrics_json,
            "frozen_blocks": list(self.frozen_blocks),
            "unit_actions": [action.to_dict() for action in self.unit_actions],
            "notes": self.notes,
            "proposer_reason": self.proposer_reason,
            "metadata": self.metadata,
        }

    def validate(self, metrics_payload: dict[str, Any] | None = None) -> None:
        frozen = {block for block in self.frozen_blocks}
        for protected in ("patch_embed", "seg_head"):
            if protected not in frozen:
                raise ValueError(f"Candidate must preserve protected block: {protected}")
        valid_unit_ids = None
        if metrics_payload is not None:
            per_unit_metrics = metrics_payload.get("per_unit_metrics") or {}
            valid_unit_ids = set(per_unit_metrics)
        for action in self.unit_actions:
            action.validate(valid_unit_ids=valid_unit_ids)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Candidate":
        return cls(
            base_model_ckpt=str(payload["base_model_ckpt"]),
            source_metrics_json=str(payload["source_metrics_json"]),
            frozen_blocks=[str(block) for block in payload.get("frozen_blocks") or []],
            unit_actions=[UnitAction.from_dict(item) for item in payload.get("unit_actions") or []],
            notes=payload.get("notes"),
            proposer_reason=payload.get("proposer_reason"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def save(self, destination: str | Path) -> Path:
        destination = Path(destination).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        if destination.suffix.lower() in {".yaml", ".yml"}:
            with destination.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)
        else:
            with destination.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
        return destination


def load_candidate(path: str | Path) -> Candidate:
    path = Path(path).resolve()
    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    else:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Candidate file must contain a top-level mapping")
    return Candidate.from_dict(payload)