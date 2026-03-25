from __future__ import annotations

from typing import Any

import torch

from .metric_tools import minmax_normalize, percentile_threshold
from .program_schema import ObjectiveConfig, ThresholdConfig


def compute_submodule_risk(
    normalized_kurtosis: float,
    normalized_entropy_std: float,
    normalized_entropy_span: float,
    objective: ObjectiveConfig,
) -> float:
    return float(
        objective.alpha * normalized_kurtosis
        + objective.beta * normalized_entropy_std
        + objective.gamma * normalized_entropy_span
    )


def aggregate_unit_metrics(
    per_submodule: dict[str, dict[str, Any]],
    objective: ObjectiveConfig,
    thresholds: ThresholdConfig,
) -> dict[str, Any]:
    ordered_keys = sorted(per_submodule)
    kurtosis_values = []
    entropy_std_values = []
    entropy_span_values = []
    for key in ordered_keys:
        metric = per_submodule[key]
        entropy = metric["entropy_summary"]
        kurtosis_values.append(float(metric["kurtosis"]))
        entropy_std_values.append(float(entropy["E_std"]))
        entropy_span_values.append(float(entropy["E_max"] - entropy["E_min"]))

    normalized_kurtosis = minmax_normalize(kurtosis_values)
    normalized_entropy_std = minmax_normalize(entropy_std_values)
    normalized_entropy_span = minmax_normalize(entropy_span_values)

    per_submodule_result = {key: dict(value) for key, value in per_submodule.items()}
    for index, key in enumerate(ordered_keys):
        risk = compute_submodule_risk(
            float(normalized_kurtosis[index].item()),
            float(normalized_entropy_std[index].item()),
            float(normalized_entropy_span[index].item()),
            objective,
        )
        per_submodule_result[key]["normalized_components"] = {
            "kurtosis": float(normalized_kurtosis[index].item()),
            "entropy_std": float(normalized_entropy_std[index].item()),
            "entropy_span": float(normalized_entropy_span[index].item()),
        }
        per_submodule_result[key]["risk"] = risk

    unit_ids = sorted({key.split("_")[0].replace("stage", "unit_") for key in ordered_keys})
    unit_weights = {
        "enc": objective.encoder_weight,
        "skip": objective.skip_weight,
        "dec": objective.decoder_weight,
    }
    weight_sum = sum(unit_weights.values())
    per_unit = {}
    score_values = []
    risk_values = []
    for unit_id in unit_ids:
        stage_id = unit_id.replace("unit_", "stage")
        members = {
            "encoder": f"{stage_id}_enc",
            "skip": f"{stage_id}_skip",
            "decoder": f"{stage_id}_dec",
        }
        similarity = sum(unit_weights[kind] * float(per_submodule_result[members[label]]["cka"]) for kind, label in (("enc", "encoder"), ("skip", "skip"), ("dec", "decoder"))) / weight_sum
        risk = sum(unit_weights[kind] * float(per_submodule_result[members[label]]["risk"]) for kind, label in (("enc", "encoder"), ("skip", "skip"), ("dec", "decoder"))) / weight_sum
        utility = objective.similarity * similarity - objective.risk * risk
        per_unit[unit_id] = {
            "weights": dict(unit_weights),
            "members": members,
            "S_unit": similarity,
            "R_unit": risk,
            "utility": utility,
        }
        score_values.append(similarity)
        risk_values.append(risk)

    score_threshold = percentile_threshold(score_values, thresholds.score_percentile)
    risk_threshold = percentile_threshold(risk_values, thresholds.risk_percentile)
    for unit_payload in per_unit.values():
        unit_payload["thresholds"] = {
            "score_percentile": thresholds.score_percentile,
            "risk_percentile": thresholds.risk_percentile,
            "score_threshold": score_threshold,
            "risk_threshold": risk_threshold,
            "selected_by_percentile": unit_payload["S_unit"] >= score_threshold and unit_payload["R_unit"] <= risk_threshold,
            "passes_similarity_limit": thresholds.min_similarity is None or unit_payload["S_unit"] >= thresholds.min_similarity,
            "passes_risk_limit": thresholds.max_risk is None or unit_payload["R_unit"] <= thresholds.max_risk,
        }

    return {
        "per_submodule": per_submodule_result,
        "per_unit": per_unit,
        "summary": {
            "score_threshold": score_threshold,
            "risk_threshold": risk_threshold,
            "mean_unit_risk": float(torch.as_tensor(risk_values, dtype=torch.float32).mean().item()) if risk_values else 0.0,
        },
    }