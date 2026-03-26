from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from search.history_logger import load_history
from search.search_loop import run_search_loop


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BTCV_PROGRAM_PATH = PROJECT_ROOT / "search/programs/btcv_pancreas.yaml"
EXAMPLE_PROGRAM_PATH = PROJECT_ROOT / "search/programs/example_organ.yaml"
METRICS_PATH = PROJECT_ROOT / "tests/fixtures/search/example_organ_metrics.json"


class GitSearchRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="git-search-runtime-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _init_git_repo(self) -> Path:
        repo_root = self.temp_dir / "repo"
        remote_root = self.temp_dir / "remote.git"
        repo_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "--bare", str(remote_root)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["git", "-C", str(repo_root), "init"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["git", "-C", str(repo_root), "config", "user.name", "Test User"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["git", "-C", str(repo_root), "config", "user.email", "test@example.com"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["git", "-C", str(repo_root), "checkout", "-b", "search/btcv-pancreas"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["git", "-C", str(repo_root), "remote", "add", "origin", str(remote_root)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        accepted_dir = repo_root / "search/results/accepted"
        accepted_dir.mkdir(parents=True, exist_ok=True)
        (accepted_dir / ".gitkeep").write_text("tracked accepted state lives here\n", encoding="utf-8")
        (repo_root / "README.md").write_text("baseline\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo_root), "add", "README.md", "search/results/accepted/.gitkeep"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["git", "-C", str(repo_root), "commit", "-m", "baseline"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return repo_root

    def _write_program(self, template_path: Path, repo_root: Path, *, git_enabled: bool, push_on_accept: bool) -> Path:
        program_path = self.temp_dir / f"{template_path.stem}_{'git' if git_enabled else 'plain'}.yaml"
        content = template_path.read_text(encoding="utf-8")
        content = content.replace("repo_root: /root/autodl-tmp/ATLAS", f"repo_root: {repo_root}")
        content = content.replace("enabled: true", f"enabled: {'true' if git_enabled else 'false'}")
        content = content.replace("enabled: false", f"enabled: {'true' if git_enabled else 'false'}")
        content = content.replace("push_on_accept: true", f"push_on_accept: {'true' if push_on_accept else 'false'}")
        content = content.replace("git_branch: search/btcv-pancreas", "git_branch: search/btcv-pancreas")
        content = content.replace("output_path: /root/autodl-tmp/ATLAS/runs/search_artifacts/metrics/btcv_pancreas_metrics.json", f"output_path: {self.temp_dir / 'metrics.json'}")
        content = content.replace("output_path: /root/autodl-tmp/ATLAS/runs/search_artifacts/metrics/example_organ_metrics.json", f"output_path: {self.temp_dir / 'metrics.json'}")
        program_path.write_text(content, encoding="utf-8")
        return program_path

    def _run_with_mocked_eval(self, program_path: Path, metrics_sequence: list[dict[str, object]], output_name: str) -> Path:
        with mock.patch("search.search_loop.evaluate_candidate", side_effect=metrics_sequence):
            return run_search_loop(
                program_path=program_path,
                metrics_path=METRICS_PATH,
                output_root=self.temp_dir / output_name,
                max_trials=1,
                dry_run=True,
                git_commit_on_accept=False,
            )

    def test_accept_commits_and_pushes_git(self) -> None:
        repo_root = self._init_git_repo()
        program_path = self._write_program(BTCV_PROGRAM_PATH, repo_root, git_enabled=True, push_on_accept=True)
        eval_sequence = [
            {"metrics": {"utility": 0.5, "val_dice": 0.5, "invalid_trial": False, "error_message": None}, "metrics_path": str(self.temp_dir / "b.json"), "resolved_config_path": str(self.temp_dir / "b_cfg.json"), "checkpoint_path": None},
            {"metrics": {"utility": 0.8, "val_dice": 0.8, "invalid_trial": False, "error_message": None}, "metrics_path": str(self.temp_dir / "t.json"), "resolved_config_path": str(self.temp_dir / "t_cfg.json"), "checkpoint_path": None},
        ]
        push_calls: list[list[str]] = []

        original_run = subprocess.run

        def fake_run(cmd, *args, **kwargs):
            if len(cmd) >= 4 and cmd[0] == "git" and cmd[3] == "push":
                push_calls.append(cmd)
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return original_run(cmd, *args, **kwargs)

        with mock.patch("subprocess.run", side_effect=fake_run):
            self._run_with_mocked_eval(program_path, eval_sequence, "accept_push")
        self.assertTrue(push_calls)

    def test_reject_rolls_back_to_last_accepted_state(self) -> None:
        repo_root = self._init_git_repo()
        program_path = self._write_program(BTCV_PROGRAM_PATH, repo_root, git_enabled=True, push_on_accept=False)
        eval_sequence = [
            {"metrics": {"utility": 0.5, "val_dice": 0.5, "invalid_trial": False, "error_message": None}, "metrics_path": str(self.temp_dir / "b.json"), "resolved_config_path": str(self.temp_dir / "b_cfg.json"), "checkpoint_path": None},
            {"metrics": {"utility": 0.4, "val_dice": 0.4, "invalid_trial": False, "error_message": None}, "metrics_path": str(self.temp_dir / "t.json"), "resolved_config_path": str(self.temp_dir / "t_cfg.json"), "checkpoint_path": None},
        ]
        trial_root = self._run_with_mocked_eval(program_path, eval_sequence, "reject_case")
        history = load_history(trial_root / "history.jsonl")
        self.assertEqual(history[-1]["accept_or_reject"], "reject")
        self.assertFalse((trial_root / "trial_001").exists())
        head = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip()
        self.assertEqual(history[-1]["rollback_target"], head)

    def test_history_contains_commit_and_rollback_fields(self) -> None:
        repo_root = self._init_git_repo()
        program_path = self._write_program(BTCV_PROGRAM_PATH, repo_root, git_enabled=True, push_on_accept=False)
        eval_sequence = [
            {"metrics": {"utility": 0.5, "val_dice": 0.5, "invalid_trial": False, "error_message": None}, "metrics_path": str(self.temp_dir / "b.json"), "resolved_config_path": str(self.temp_dir / "b_cfg.json"), "checkpoint_path": None},
            {"metrics": {"utility": 0.8, "val_dice": 0.8, "invalid_trial": False, "error_message": None}, "metrics_path": str(self.temp_dir / "t.json"), "resolved_config_path": str(self.temp_dir / "t_cfg.json"), "checkpoint_path": None},
        ]
        run_root = self._run_with_mocked_eval(program_path, eval_sequence, "history_case")
        history = load_history(run_root / "history.jsonl")
        self.assertIn("commit_hash", history[-1])
        self.assertIn("rollback_target", history[-1])

    def test_outputs_stay_under_runs_dir(self) -> None:
        run_root = run_search_loop(
            program_path=EXAMPLE_PROGRAM_PATH,
            metrics_path=METRICS_PATH,
            output_root=self.temp_dir / "runs" / "search-test",
            max_trials=1,
            dry_run=True,
            git_commit_on_accept=False,
        )
        self.assertTrue(str(run_root).startswith(str(self.temp_dir / "runs")))
        self.assertTrue((run_root / "best" / "best_state.json").exists())

    def test_btcv_pancreas_program_smoke(self) -> None:
        repo_root = self._init_git_repo()
        program_path = self._write_program(BTCV_PROGRAM_PATH, repo_root, git_enabled=False, push_on_accept=False)
        run_root = run_search_loop(
            program_path=program_path,
            metrics_path=METRICS_PATH,
            output_root=self.temp_dir / "runs" / "btcv-smoke",
            max_trials=1,
            dry_run=True,
            git_commit_on_accept=False,
        )
        self.assertTrue((run_root / "history.jsonl").exists())

    def test_search_loop_can_finish_one_small_search(self) -> None:
        run_root = run_search_loop(
            program_path=EXAMPLE_PROGRAM_PATH,
            metrics_path=METRICS_PATH,
            output_root=self.temp_dir / "runs" / "small-search",
            max_trials=1,
            dry_run=True,
            git_commit_on_accept=False,
        )
        history = load_history(run_root / "history.jsonl")
        self.assertGreaterEqual(len(history), 2)
