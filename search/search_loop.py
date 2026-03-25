from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .candidate_builder import build_candidate
from .candidate_schema import Candidate
from .evaluator import evaluate_candidate
from .git_runtime import GitRuntimeConfig, GitSearchRuntime
from .history_logger import append_history_entry, build_history_entry, load_history
from .program_schema import load_program_config
from .search_space import SearchSpace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True, help="Path to the program YAML.")
    parser.add_argument("--metrics", required=True, help="Path to the stage-1 metrics JSON.")
    parser.add_argument("--output-root", default=None, help="Optional search run root.")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional max trial override.")
    parser.add_argument("--dry-run", action="store_true", help="Use dry-run evaluator.")
    parser.add_argument("--git-commit-on-accept", action="store_true", help="Commit the accepted snapshot when utility improves.")
    parser.add_argument("--git-commit-on-improve", action="store_true", help="Backward-compatible alias for --git-commit-on-accept.")
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_run_root(metrics_path: str | Path) -> Path:
    del metrics_path
    project_root = Path(__file__).resolve().parent.parent
    run_id = datetime.now().strftime("search-%Y%m%d-%H%M%S")
    return project_root / "runs" / run_id


def _save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_search_loop(
    program_path: str | Path,
    metrics_path: str | Path,
    output_root: str | Path | None = None,
    max_trials: int | None = None,
    dry_run: bool = False,
    git_commit_on_accept: bool = False,
) -> Path:
    program = load_program_config(program_path)
    metrics_payload = _load_json(metrics_path)
    search_space = SearchSpace.from_metrics_payload(metrics_payload, frozen_blocks=program.freeze_blocks)
    run_root = Path(output_root).resolve() if output_root is not None else _default_run_root(metrics_path)
    run_root.mkdir(parents=True, exist_ok=True)
    history_path = run_root / "history.jsonl"
    best_state_path = run_root / "best_state.json"
    best_candidate_path = run_root / "best_candidate.json"
    best_metrics_path = run_root / "best_metrics.json"
    git_runtime = GitSearchRuntime(
        GitRuntimeConfig(
            repo_root=program.runtime.git_repo_root,
            remote=program.runtime.git_remote,
            branch=program.runtime.git_branch,
            commit_on_accept=bool(program.runtime.git_commit_on_accept or git_commit_on_accept),
            push_on_accept=bool(program.runtime.git_push_on_accept),
            tag_on_accept=bool(program.runtime.git_tag_on_accept),
            rollback_on_reject=bool(program.runtime.git_rollback_on_reject),
        ),
        repo_hint=Path(program_path).resolve().parent.parent,
    )
    git_runtime.ensure_ready()

    baseline_trial_dir = run_root / "trial_000_baseline"
    baseline_candidate = Candidate(
        base_model_ckpt=program.base_model_ckpt,
        source_metrics_json=str(Path(metrics_path).resolve()),
        frozen_blocks=list(search_space.frozen_blocks),
        unit_actions=[],
        notes="Mother baseline",
        proposer_reason="Baseline evaluation before search",
        metadata={"trial": 0, "kind": "baseline"},
    )
    baseline_candidate_file = baseline_candidate.save(baseline_trial_dir / "candidate.json")
    baseline_eval = evaluate_candidate(baseline_candidate_file, program_path, baseline_trial_dir, dry_run=dry_run, baseline_metrics=None)
    best_utility = float(baseline_eval["metrics"]["utility"])
    best_state = {
        "trial_id": "trial_000_baseline",
        "candidate_path": str(baseline_candidate_file),
        "metrics_path": baseline_eval["metrics_path"],
        "checkpoint_path": baseline_eval.get("checkpoint_path") or program.base_model_ckpt,
        "utility": best_utility,
        "source_metrics_json": str(Path(metrics_path).resolve()),
    }
    _save_json(best_state_path, best_state)
    _save_json(best_candidate_path, baseline_candidate.to_dict())
    _save_json(best_metrics_path, baseline_eval["metrics"])

    append_history_entry(
        history_path,
        build_history_entry(
            trial_id="trial_000_baseline",
            source_metrics_json=str(Path(metrics_path).resolve()),
            candidate_path=str(baseline_candidate_file),
            proposal_reason=baseline_candidate.proposer_reason or "baseline",
            unit_actions=[action.to_dict() for action in baseline_candidate.unit_actions],
            resolved_config_path=baseline_eval["resolved_config_path"],
            metrics=baseline_eval["metrics"],
            utility=baseline_eval["metrics"]["utility"],
            accept_or_reject="accept",
            invalid_trial=bool(baseline_eval["metrics"].get("invalid_trial", False)),
            error_message=baseline_eval["metrics"].get("error_message"),
            best_utility_before=None,
            best_utility_after=best_utility,
            git_head_before=git_runtime.accepted_ref,
            git_head_after=git_runtime.accepted_ref,
        ),
    )

    trial_budget = int(max_trials if max_trials is not None else program.search.max_trials)
    for trial_index in range(1, trial_budget + 1):
        trial_id = f"trial_{trial_index:03d}"
        trial_dir = run_root / trial_id
        history_entries = load_history(history_path)
        candidate, candidate_file = build_candidate(
            metrics_path=metrics_path,
            program_path=program_path,
            output_path=trial_dir / "candidate.json",
            history_path=history_path,
            base_ckpt_override=best_state["checkpoint_path"],
        )
        search_space.validate_candidate(candidate)
        eval_result = evaluate_candidate(
            candidate_file,
            program_path,
            trial_dir,
            dry_run=dry_run,
            baseline_metrics=baseline_eval["metrics"],
        )
        utility = float(eval_result["metrics"]["utility"])
        invalid_trial = bool(eval_result["metrics"].get("invalid_trial", False))
        accept = (not invalid_trial) and utility > best_utility + float(program.search.epsilon_accept)
        verdict = "accept" if accept else "reject"
        best_utility_before = best_utility
        git_head_before = git_runtime.accepted_ref
        git_head_after = git_runtime.accepted_ref
        if accept:
            best_utility = utility
            best_state = {
                "trial_id": trial_id,
                "candidate_path": str(candidate_file),
                "metrics_path": eval_result["metrics_path"],
                "checkpoint_path": eval_result.get("checkpoint_path") or best_state["checkpoint_path"],
                "utility": utility,
                "source_metrics_json": str(Path(metrics_path).resolve()),
            }
            _save_json(best_state_path, best_state)
            _save_json(best_candidate_path, candidate.to_dict())
            _save_json(best_metrics_path, eval_result["metrics"])
            git_head_after = git_runtime.accept(
                trial_id=trial_id,
                program_path=program_path,
                metrics_path=metrics_path,
                run_root=run_root,
                candidate_payload=candidate.to_dict(),
                metrics_payload=eval_result["metrics"],
                best_state=best_state,
            )
            if git_head_after is not None:
                best_state["accepted_git_commit"] = git_head_after
                _save_json(best_state_path, best_state)
        else:
            git_runtime.reject(trial_dir)
        append_history_entry(
            history_path,
            build_history_entry(
                trial_id=trial_id,
                source_metrics_json=str(Path(metrics_path).resolve()),
                candidate_path=str(candidate_file),
                proposal_reason=candidate.proposer_reason or "heuristic proposal",
                unit_actions=[action.to_dict() for action in candidate.unit_actions],
                resolved_config_path=eval_result["resolved_config_path"],
                metrics=eval_result["metrics"],
                utility=utility,
                accept_or_reject=verdict,
                invalid_trial=invalid_trial,
                error_message=eval_result["metrics"].get("error_message"),
                best_utility_before=best_utility_before,
                best_utility_after=best_utility,
                git_head_before=git_head_before,
                git_head_after=git_head_after,
            ),
        )
    return run_root


def main() -> None:
    args = parse_args()
    run_root = run_search_loop(
        program_path=args.program,
        metrics_path=args.metrics,
        output_root=args.output_root,
        max_trials=args.max_trials,
        dry_run=args.dry_run,
        git_commit_on_accept=bool(args.git_commit_on_accept or args.git_commit_on_improve),
    )
    print(json.dumps({"run_root": str(run_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()