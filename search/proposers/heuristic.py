from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import BaseProposer


@dataclass(slots=True)
class ProposedAction:
    unit_id: str
    action_type: str
    prune_ratio: float | None = None
    expand_type: str | None = None
    expand_delta: float | int | None = None
    proposer_reason: str | None = None


@dataclass(slots=True)
class Proposal:
    strategy: str
    reason: str
    actions: list[ProposedAction] = field(default_factory=list)


def _compute_prune_ratio(score: float, tau_prune: float, max_prune_ratio: float) -> float:
    if score <= tau_prune:
        return 0.0
    denominator = max(1.0 - tau_prune, 1e-8)
    ratio = max_prune_ratio * (score - tau_prune) / denominator
    return max(0.0, min(max_prune_ratio, ratio))


def propose_actions(
    per_unit_metrics: dict[str, dict[str, Any]],
    thresholds: Any,
    strategy: str = "conservative",
    history_summary: dict[str, Any] | None = None,
    max_units_per_trial: int = 1,
    enable_expand: bool = False,
) -> Proposal:
    history_summary = history_summary or {}
    reject_streaks = history_summary.get("reject_streaks") or {}
    tau_prune = min(0.99, float(thresholds.tau_prune) + 0.03 * int(reject_streaks.get("prune", 0)))
    tau_quant = max(0.0, float(thresholds.tau_quant) - 0.03 * int(reject_streaks.get("quantize", 0)))
    candidates: list[tuple[float, ProposedAction]] = []
    for unit_id, metric in per_unit_metrics.items():
        score = float(metric["S_unit"])
        risk = float(metric["R_unit"])
        utility = float(metric.get("utility", score - risk))
        action = ProposedAction(unit_id=unit_id, action_type="keep", proposer_reason=f"default keep for {unit_id}")
        desirability = utility
        if strategy != "greedy_quant" and score >= tau_prune:
            prune_ratio = _compute_prune_ratio(score, tau_prune, float(thresholds.max_prune_ratio))
            desirability = score - 0.5 * risk
            action = ProposedAction(
                unit_id=unit_id,
                action_type="prune",
                prune_ratio=prune_ratio,
                proposer_reason=f"S_unit={score:.4f} >= tau_prune={tau_prune:.4f}; prune_ratio={prune_ratio:.4f}",
            )
        elif strategy != "greedy_prune" and score < tau_prune and risk < tau_quant:
            desirability = (tau_prune - score) + (tau_quant - risk)
            action = ProposedAction(
                unit_id=unit_id,
                action_type="quantize",
                proposer_reason=f"S_unit={score:.4f} < tau_prune={tau_prune:.4f} and R_unit={risk:.4f} < tau_quant={tau_quant:.4f}",
            )
        elif enable_expand and score < float(thresholds.tau_low) and risk >= float(thresholds.tau_high):
            desirability = 0.05 + risk - score
            action = ProposedAction(
                unit_id=unit_id,
                action_type="expand",
                expand_type="num_subblocks",
                expand_delta=1,
                proposer_reason="expand gate opened by enable_expand and low-score/high-risk signal",
            )
        elif strategy == "greedy_prune":
            desirability = score - risk
        elif strategy == "greedy_quant":
            desirability = -risk + (1.0 - score)
        candidates.append((desirability, action))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for _, action in candidates:
        if action.action_type == "keep" and selected:
            continue
        selected.append(action)
        if len(selected) >= max_units_per_trial:
            break
    if not selected:
        unit_id = sorted(per_unit_metrics)[0]
        selected = [ProposedAction(unit_id=unit_id, action_type="keep", proposer_reason="fallback keep due to insufficient evidence")]
    return Proposal(strategy=strategy, reason=f"heuristic strategy={strategy}; selected {len(selected)} action(s)", actions=selected)


class HeuristicProposer(BaseProposer):
    def __init__(self, strategy: str = "conservative"):
        self.name = "heuristic"
        self.strategy = strategy

    def propose(
        self,
        per_unit_metrics: dict[str, dict[str, Any]],
        thresholds: Any,
        history_summary: dict[str, Any] | None = None,
        max_units_per_trial: int = 1,
        enable_expand: bool = False,
    ) -> Proposal:
        return propose_actions(
            per_unit_metrics,
            thresholds,
            strategy=self.strategy,
            history_summary=history_summary,
            max_units_per_trial=max_units_per_trial,
            enable_expand=enable_expand,
        )