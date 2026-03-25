from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def dice_from_labels(prediction: np.ndarray, target: np.ndarray, num_classes: int) -> list[float]:
    scores = []
    for class_index in range(1, num_classes):
        pred_mask = prediction == class_index
        target_mask = target == class_index
        intersection = float(np.logical_and(pred_mask, target_mask).sum())
        denom = float(pred_mask.sum() + target_mask.sum())
        if denom == 0.0:
            scores.append(1.0)
        else:
            scores.append((2.0 * intersection) / denom)
    return scores


def logits_to_labels(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=1)


def summarize_case_metrics(case_metrics: list[dict]) -> dict:
    if not case_metrics:
        return {"foreground_mean": {"Dice": 0.0}, "cases": []}

    dice_matrix = np.asarray([entry["dice_per_class"] for entry in case_metrics], dtype=np.float64)
    mean_per_class = dice_matrix.mean(axis=0).tolist()
    foreground_mean = float(dice_matrix.mean())
    return {
        "foreground_mean": {"Dice": foreground_mean},
        "mean_per_class": mean_per_class,
        "cases": case_metrics,
    }


def save_summary(summary: dict, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
