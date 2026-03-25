from __future__ import annotations

import unittest
from pathlib import Path

import torch

from search.metric_tools import adaptive_token_features, entropy_summary, linear_cka, percentile_threshold, rms_kurtosis
from search.program_schema import load_program_config
from search.unit_aggregation import aggregate_unit_metrics


class MetricToolsSmokeTest(unittest.TestCase):
    def test_linear_cka_identity(self) -> None:
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertAlmostEqual(linear_cka(features, features), 1.0, places=5)

    def test_entropy_summary_ordering(self) -> None:
        logits = torch.tensor([[[2.0, -2.0], [0.0, 0.0]]]).transpose(1, 2)
        summary = entropy_summary(logits)
        self.assertGreaterEqual(summary["E_max"], summary["E_min"])

    def test_rms_kurtosis_non_negative(self) -> None:
        activations = torch.arange(1, 33, dtype=torch.float32).reshape(2, 4, 4)
        summary = rms_kurtosis(activations)
        self.assertGreaterEqual(summary["kurtosis"], 0.0)

    def test_percentile_threshold(self) -> None:
        self.assertAlmostEqual(percentile_threshold([1.0, 2.0, 3.0, 4.0], 50.0), 2.5, places=5)

    def test_adaptive_token_features_shape(self) -> None:
        tensor = torch.randn(2, 3, 4, 4, 4)
        features = adaptive_token_features(tensor, max_spatial_tokens=8)
        self.assertEqual(features.shape[1], 3)

    def test_aggregate_unit_metrics(self) -> None:
        per_submodule = {
            key: {
                "cka": 0.8,
                "kurtosis": 1.1,
                "entropy_summary": {"E_min": 0.1, "E_max": 0.2, "E_mean": 0.15, "E_std": 0.02},
            }
            for key in (
                "stage1_enc",
                "stage1_skip",
                "stage1_dec",
                "stage2_enc",
                "stage2_skip",
                "stage2_dec",
                "stage3_enc",
                "stage3_skip",
                "stage3_dec",
            )
        }
        program = load_program_config(Path(__file__).resolve().parent.parent / "search/programs/example_organ.yaml")
        aggregated = aggregate_unit_metrics(per_submodule, program.objective, program.search_thresholds)
        self.assertIn("unit_1", aggregated["per_unit"])


if __name__ == "__main__":
    unittest.main()