from __future__ import annotations

from typing import Any


def _percentile_rank(values: list[float], target: float) -> float:
    if not values:
        return 0.5
    sorted_values = sorted(values)
    less_equal = sum(1 for value in sorted_values if value <= target)
    return max(0.0, min(1.0, (less_equal - 1) / max(len(sorted_values) - 1, 1)))


def _select_bit(candidates: list[int], target_rank: float) -> int:
    ordered = sorted({int(value) for value in candidates})
    if len(ordered) == 1:
        return ordered[0]
    target_rank = max(0.0, min(1.0, target_rank))
    index = int(round(target_rank * (len(ordered) - 1)))
    return ordered[index]


def allocate_bits(
    unit_id: str,
    per_unit_metrics: dict[str, dict[str, Any]],
    search_thresholds: Any,
    allowed_weight_bits: list[int],
    allowed_act_bits: list[int],
    history_summary: dict[str, Any] | None = None,
    allocation_mode: str = "conservative",
) -> tuple[int, int, str]:
    if unit_id not in per_unit_metrics:
        raise ValueError(f"Unknown unit_id for allocation: {unit_id}")
    history_summary = history_summary or {}
    metric = per_unit_metrics[unit_id]
    risk_values = [float(payload["R_unit"]) for payload in per_unit_metrics.values()]
    similarity_values = [float(payload["S_unit"]) for payload in per_unit_metrics.values()]
    risk_rank = _percentile_rank(risk_values, float(metric["R_unit"]))
    similarity_rank = _percentile_rank(similarity_values, float(metric["S_unit"]))
    mode_shift = {"conservative": 0.2, "balanced": 0.0, "aggressive": -0.2}[allocation_mode]
    reject_streak = int((history_summary.get("reject_streaks") or {}).get("quantize", 0))
    low_bit_rejects = history_summary.get("low_bit_rejects") or {"weight": {}, "act": {}}

    weight_penalty = 0.04 * sum(count for bit, count in (low_bit_rejects.get("weight") or {}).items() if int(bit) <= min(allowed_weight_bits))
    act_penalty = 0.04 * sum(count for bit, count in (low_bit_rejects.get("act") or {}).items() if int(bit) <= min(allowed_act_bits))
    threshold_bias = 0.0
    if hasattr(search_thresholds, "tau_quant"):
        threshold_bias = max(0.0, float(metric["R_unit"]) - float(search_thresholds.tau_quant)) * 0.25

    weight_target = risk_rank + 0.20 * (1.0 - similarity_rank) + mode_shift + 0.06 * reject_streak + weight_penalty + threshold_bias
    act_target = risk_rank + 0.25 * (1.0 - similarity_rank) + mode_shift + 0.08 * reject_streak + act_penalty + threshold_bias
    selected_weight_bits = _select_bit(allowed_weight_bits, weight_target)
    selected_act_bits = _select_bit(allowed_act_bits, act_target)
    reason = (
        f"unit={unit_id}; risk_rank={risk_rank:.3f}; similarity_rank={similarity_rank:.3f}; "
        f"mode={allocation_mode}; reject_streak={reject_streak}; "
        f"weight_target={weight_target:.3f}->bits={selected_weight_bits}; "
        f"act_target={act_target:.3f}->bits={selected_act_bits}"
    )
    return selected_weight_bits, selected_act_bits, reason