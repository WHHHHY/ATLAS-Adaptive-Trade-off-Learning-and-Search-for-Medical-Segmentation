from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseProposer(ABC):
    name: str

    @abstractmethod
    def propose(
        self,
        per_unit_metrics: dict[str, dict[str, Any]],
        thresholds: Any,
        history_summary: dict[str, Any] | None = None,
        max_units_per_trial: int = 1,
        enable_expand: bool = False,
    ):
        raise NotImplementedError


def get_proposer(strategy: str) -> BaseProposer:
    from .heuristic import HeuristicProposer

    return HeuristicProposer(strategy=strategy)