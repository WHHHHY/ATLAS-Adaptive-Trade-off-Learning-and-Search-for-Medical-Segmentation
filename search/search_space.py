from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .candidate_schema import Candidate, UnitAction


@dataclass(slots=True)
class SearchSpace:
    per_unit_metrics: dict[str, dict[str, Any]]
    frozen_blocks: tuple[str, ...] = ("patch_embed", "seg_head")

    @classmethod
    def from_metrics_payload(cls, metrics_payload: dict[str, Any], frozen_blocks: list[str] | None = None) -> "SearchSpace":
        per_unit_metrics = metrics_payload.get("per_unit_metrics") or {}
        if not per_unit_metrics:
            raise ValueError("metrics payload must contain per_unit_metrics")
        normalized_frozen = list(frozen_blocks or [])
        for protected in ("patch_embed", "seg_head"):
            if protected not in normalized_frozen:
                normalized_frozen.append(protected)
        return cls(per_unit_metrics=per_unit_metrics, frozen_blocks=tuple(normalized_frozen))

    def list_mutable_units(self) -> list[str]:
        return sorted(self.per_unit_metrics)

    def is_frozen_block(self, name: str) -> bool:
        return name in set(self.frozen_blocks)

    def validate_action(self, unit_action: UnitAction) -> None:
        if self.is_frozen_block(unit_action.unit_id):
            raise ValueError(f"Protected block cannot be mutated: {unit_action.unit_id}")
        if unit_action.unit_id not in self.per_unit_metrics:
            raise ValueError(f"Unknown unit_id: {unit_action.unit_id}")
        if unit_action.action_type == "expand" and unit_action.expand_type == "width_mult":
            raise ValueError("width_mult expand is unsafe in the current repo because it would alter protected edge blocks")
        unit_action.validate(valid_unit_ids=set(self.per_unit_metrics))

    def validate_candidate(self, candidate: Candidate) -> None:
        candidate.validate(metrics_payload={"per_unit_metrics": self.per_unit_metrics})
        for action in candidate.unit_actions:
            self.validate_action(action)