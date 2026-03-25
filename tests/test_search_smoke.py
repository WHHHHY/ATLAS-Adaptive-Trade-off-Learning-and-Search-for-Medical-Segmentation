from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from search.bit_allocator import allocate_bits
from search.candidate_builder import build_candidate
from search.evaluator import evaluate_candidate
from search.program_schema import load_program_config
from search.search_loop import run_search_loop
from search.search_space import SearchSpace


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROGRAM_PATH = PROJECT_ROOT / "search/programs/example_organ.yaml"
METRICS_PATH = PROJECT_ROOT / "tests/fixtures/search/example_organ_metrics.json"


class SearchSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="search-smoke-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_candidate_builder_and_search_loop(self) -> None:
        program = load_program_config(PROGRAM_PATH)
        with METRICS_PATH.open("r", encoding="utf-8") as handle:
            metrics_payload = json.load(handle)
        self.assertIn("per_unit_metrics", metrics_payload)

        candidate, candidate_path = build_candidate(
            metrics_path=METRICS_PATH,
            program_path=PROGRAM_PATH,
            output_path=self.temp_dir / "candidate.json",
        )
        self.assertTrue(candidate_path.exists())
        search_space = SearchSpace.from_metrics_payload(metrics_payload, frozen_blocks=program.freeze_blocks)
        search_space.validate_candidate(candidate)
        self.assertIn("patch_embed", candidate.frozen_blocks)
        self.assertIn("seg_head", candidate.frozen_blocks)

        per_unit_metrics = metrics_payload["per_unit_metrics"]
        ordered_units = sorted(per_unit_metrics)
        low_unit = min(ordered_units, key=lambda unit_id: per_unit_metrics[unit_id]["R_unit"])
        high_unit = max(ordered_units, key=lambda unit_id: per_unit_metrics[unit_id]["R_unit"])
        low_alloc = allocate_bits(
            low_unit,
            per_unit_metrics,
            program.search_thresholds,
            program.quantization.weight_bits,
            program.quantization.act_bits,
            history_summary=None,
            allocation_mode="conservative",
        )
        high_alloc = allocate_bits(
            high_unit,
            per_unit_metrics,
            program.search_thresholds,
            program.quantization.weight_bits,
            program.quantization.act_bits,
            history_summary=None,
            allocation_mode="conservative",
        )
        self.assertTrue(low_alloc[0] <= high_alloc[0] or low_alloc[1] <= high_alloc[1])
        self.assertNotEqual(low_alloc[2], high_alloc[2])

        eval_result = evaluate_candidate(candidate_path, PROGRAM_PATH, self.temp_dir / "eval", dry_run=True)
        self.assertTrue(Path(eval_result["metrics_path"]).exists())
        self.assertIn("utility", eval_result["metrics"])

        run_root = run_search_loop(
            program_path=PROGRAM_PATH,
            metrics_path=METRICS_PATH,
            output_root=self.temp_dir / "loop",
            max_trials=1,
            dry_run=True,
            git_commit_on_improve=False,
        )
        self.assertTrue((run_root / "history.jsonl").exists())
        self.assertTrue((run_root / "best_state.json").exists())


if __name__ == "__main__":
    unittest.main()