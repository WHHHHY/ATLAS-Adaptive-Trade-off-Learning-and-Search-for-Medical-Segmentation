from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


@dataclass(slots=True)
class GitRuntimeConfig:
    enabled: bool = False
    repo_root: str | None = None
    remote: str = "origin"
    branch: str | None = None
    commit_on_accept: bool = False
    push_on_accept: bool = False
    tag_on_accept: bool = False
    rollback_on_reject: bool = True
    accepted_state_dir: str = "search/results/accepted"

    @property
    def active(self) -> bool:
        return self.enabled and any((self.commit_on_accept, self.push_on_accept, self.tag_on_accept, self.rollback_on_reject))


class GitSearchRuntime:
    def __init__(self, config: GitRuntimeConfig, repo_hint: str | Path | None = None) -> None:
        self.config = config
        self.repo_root = Path(config.repo_root).resolve() if config.repo_root else Path(repo_hint).resolve() if repo_hint is not None else None
        self.accepted_ref: str | None = None

    def ensure_ready(self) -> None:
        if not self.config.active:
            return
        if self.repo_root is None:
            raise RuntimeError("Git runtime requires git_repo_root or repo_hint.")
        self.repo_root = Path(
            _git(self.repo_root, "rev-parse", "--show-toplevel").stdout.strip()
        ).resolve()
        status = _git(self.repo_root, "status", "--porcelain").stdout.strip()
        if status:
            raise RuntimeError("Git-driven search requires a clean worktree before starting.")
        if self.config.push_on_accept:
            remotes = _git(self.repo_root, "remote").stdout.splitlines()
            if self.config.remote not in remotes:
                raise RuntimeError(f"Git remote not found: {self.config.remote}")
        self.accepted_ref = self.head_commit()

    def head_commit(self) -> str:
        if self.repo_root is None:
            raise RuntimeError("Git runtime has no repository root.")
        return _git(self.repo_root, "rev-parse", "HEAD").stdout.strip()

    def _accepted_state_path(self) -> Path:
        if self.repo_root is None:
            raise RuntimeError("Git runtime has no repository root.")
        return self.repo_root / self.config.accepted_state_dir

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def accept(
        self,
        trial_id: str,
        program_path: str | Path,
        metrics_path: str | Path,
        run_root: str | Path,
        candidate_payload: dict[str, Any],
        metrics_payload: dict[str, Any],
        best_state: dict[str, Any],
    ) -> str | None:
        if not self.config.active:
            return self.accepted_ref
        state_dir = self._accepted_state_path()
        manifest = {
            "trial_id": trial_id,
            "program_path": str(Path(program_path).resolve()),
            "metrics_path": str(Path(metrics_path).resolve()),
            "run_root": str(Path(run_root).resolve()),
            "accepted_ref_before": self.accepted_ref,
        }
        self._write_json(state_dir / "accepted_candidate.json", candidate_payload)
        self._write_json(state_dir / "accepted_metrics.json", metrics_payload)
        self._write_json(state_dir / "accepted_state.json", best_state)
        self._write_json(state_dir / "manifest.json", manifest)
        if not any((self.config.commit_on_accept, self.config.push_on_accept, self.config.tag_on_accept)):
            return self.accepted_ref
        _git(self.repo_root, "add", self.config.accepted_state_dir)
        has_changes = bool(_git(self.repo_root, "status", "--porcelain", "--", self.config.accepted_state_dir).stdout.strip())
        if has_changes and self.config.commit_on_accept:
            _git(self.repo_root, "commit", "-m", f"Accept {trial_id} utility improvement")
        self.accepted_ref = self.head_commit()
        if self.config.tag_on_accept:
            _git(self.repo_root, "tag", f"accept-{trial_id}", self.accepted_ref)
        if self.config.push_on_accept:
            push_target = self.config.branch or _git(self.repo_root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
            _git(self.repo_root, "push", self.config.remote, f"HEAD:{push_target}")
            if self.config.tag_on_accept:
                _git(self.repo_root, "push", self.config.remote, f"accept-{trial_id}")
        return self.accepted_ref

    def reject(self, trial_dir: str | Path) -> None:
        trial_dir = Path(trial_dir)
        if trial_dir.exists():
            shutil.rmtree(trial_dir, ignore_errors=True)
        if not self.config.active or not self.config.rollback_on_reject or self.accepted_ref is None:
            return
        tracked_changes = _git(self.repo_root, "status", "--porcelain").stdout.strip()
        if tracked_changes:
            _git(self.repo_root, "reset", "--hard", self.accepted_ref)