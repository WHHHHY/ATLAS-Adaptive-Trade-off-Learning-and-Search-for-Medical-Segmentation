from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .bit_allocator import allocate_bits
from .candidate_schema import Candidate, ScoreSnapshot, UnitAction
from .history_logger import load_history, summarize_history
from .program_schema import load_program_config
from .proposers import get_proposer
from .search_space import SearchSpace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="Path to the stage-1 metrics JSON.")
    parser.add_argument("--program", required=True, help="Path to the program YAML.")
    parser.add_argument("--output", required=True, help="Destination candidate JSON/YAML.")
    parser.add_argument("--history", default=None, help="Optional history.jsonl path.")
    parser.add_argument("--base-ckpt", default=None, help="Optional best-state checkpoint override.")
    return parser.parse_args()


def _load_metrics(metrics_path: str | Path) -> dict[str, Any]:
    with Path(metrics_path).resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_candidate(
    metrics_path: str | Path,
    program_path: str | Path,
    output_path: str | Path | None = None,
    history_path: str | Path | None = None,
    base_ckpt_override: str | None = None,
) -> tuple[Candidate, Path | None]:
    metrics_path = Path(metrics_path).resolve()
    metrics_payload = _load_metrics(metrics_path)
    program = load_program_config(program_path)
    history_entries = load_history(history_path) if history_path is not None else []
    history_summary = summarize_history(history_entries)
    search_space = SearchSpace.from_metrics_payload(metrics_payload, frozen_blocks=program.freeze_blocks)
    proposer = get_proposer(program.search.strategy)
    proposal = proposer.propose(
        metrics_payload["per_unit_metrics"],
        program.search_thresholds,
        history_summary=history_summary,
        max_units_per_trial=program.search.max_units_per_trial,
        enable_expand=program.search.enable_expand,
    )

    unit_actions = []
    for proposed_action in proposal.actions:
        unit_metric = metrics_payload["per_unit_metrics"][proposed_action.unit_id]
        weight_bits = None
        act_bits = None
        allocation_reason = None
        allocation_mode = None
        if proposed_action.action_type == "quantize":
            allocation_mode = program.quantization.allocator_mode
            weight_bits, act_bits, allocation_reason = allocate_bits(
                proposed_action.unit_id,
                metrics_payload["per_unit_metrics"],
                program.search_thresholds,
                allowed_weight_bits=program.quantization.weight_bits,
                allowed_act_bits=program.quantization.act_bits,
                history_summary=history_summary,
                allocation_mode=allocation_mode,
            )
        unit_action = UnitAction(
            unit_id=proposed_action.unit_id,
            action_type=proposed_action.action_type,
            prune_ratio=proposed_action.prune_ratio,
            weight_bits=weight_bits,
            act_bits=act_bits,
            allocation_mode=allocation_mode,
            allocation_reason=allocation_reason,
            expand_type=proposed_action.expand_type,
            expand_delta=proposed_action.expand_delta,
            proposer_reason=proposed_action.proposer_reason,
            score_snapshot=ScoreSnapshot(
                S_unit=float(unit_metric["S_unit"]),
                R_unit=float(unit_metric["R_unit"]),
                utility_from_metrics_json=float(unit_metric.get("utility")) if unit_metric.get("utility") is not None else None,
            ),
        )
        search_space.validate_action(unit_action)
        unit_actions.append(unit_action)

    candidate = Candidate(
        base_model_ckpt=base_ckpt_override or program.base_model_ckpt,
        source_metrics_json=str(metrics_path),
        frozen_blocks=list(search_space.frozen_blocks),
        unit_actions=unit_actions,
        notes="Heuristic stage-2 candidate.",
        proposer_reason=proposal.reason,
        metadata={
            "strategy": program.search.strategy,
            "history_summary": history_summary,
        },
    )
    search_space.validate_candidate(candidate)
    saved_path = candidate.save(output_path) if output_path is not None else None
    return candidate, saved_path


def main() -> None:
    args = parse_args()
    candidate, saved_path = build_candidate(
        metrics_path=args.metrics,
        program_path=args.program,
        output_path=args.output,
        history_path=args.history,
        base_ckpt_override=args.base_ckpt,
    )
    payload = {
        "candidate_path": str(saved_path) if saved_path is not None else None,
        "num_actions": len(candidate.unit_actions),
        "proposer_reason": candidate.proposer_reason,
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()