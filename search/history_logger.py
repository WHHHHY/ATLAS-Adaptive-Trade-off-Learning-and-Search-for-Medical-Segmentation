from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_history(history_path: str | Path) -> list[dict[str, Any]]:
    history_path = Path(history_path)
    if not history_path.exists():
        return []
    entries = []
    with history_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def append_history_entry(history_path: str | Path, entry: dict[str, Any]) -> None:
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def summarize_history(entries: list[dict[str, Any]]) -> dict[str, Any]:
    reject_streaks = {"prune": 0, "quantize": 0, "keep": 0, "expand": 0}
    low_bit_rejects: dict[str, dict[int, int]] = {"weight": {}, "act": {}}
    for entry in reversed(entries):
        verdict = entry.get("accept_or_reject")
        if verdict != "reject":
            break
        for action in entry.get("unit_actions") or []:
            action_type = action.get("action_type")
            if action_type in reject_streaks:
                reject_streaks[action_type] += 1
            if action_type == "quantize":
                weight_bits = action.get("weight_bits")
                act_bits = action.get("act_bits")
                if weight_bits is not None:
                    low_bit_rejects["weight"][int(weight_bits)] = low_bit_rejects["weight"].get(int(weight_bits), 0) + 1
                if act_bits is not None:
                    low_bit_rejects["act"][int(act_bits)] = low_bit_rejects["act"].get(int(act_bits), 0) + 1
    return {
        "num_entries": len(entries),
        "reject_streaks": reject_streaks,
        "low_bit_rejects": low_bit_rejects,
        "last_entry": entries[-1] if entries else None,
    }


def build_history_entry(
    trial_id: str,
    source_metrics_json: str,
    candidate_path: str,
    proposal_reason: str,
    unit_actions: list[dict[str, Any]],
    resolved_config_path: str,
    metrics: dict[str, Any],
    utility: float | None,
    accept_or_reject: str,
    invalid_trial: bool,
    error_message: str | None,
    best_utility_before: float | None,
    best_utility_after: float | None,
    git_head_before: str | None = None,
    git_head_after: str | None = None,
) -> dict[str, Any]:
    allocation_reasons = [action.get("allocation_reason") for action in unit_actions if action.get("allocation_reason")]
    return {
        "trial_id": trial_id,
        "source_metrics_json": source_metrics_json,
        "candidate_path": candidate_path,
        "proposal_reason": proposal_reason,
        "unit_actions": unit_actions,
        "resolved_config_path": resolved_config_path,
        "metrics": metrics,
        "utility": utility,
        "accept_or_reject": accept_or_reject,
        "invalid_trial": invalid_trial,
        "error_message": error_message,
        "best_utility_before": best_utility_before,
        "best_utility_after": best_utility_after,
        "allocation_reason": allocation_reasons,
        "git_head_before": git_head_before,
        "git_head_after": git_head_after,
    }