from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F


def _to_float_tensor(values: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().float()
    return torch.as_tensor(list(values), dtype=torch.float32)


def minmax_normalize(values: Iterable[float] | torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    tensor = _to_float_tensor(values)
    if tensor.numel() == 0:
        return tensor
    min_value = tensor.min()
    max_value = tensor.max()
    if float(max_value - min_value) <= eps:
        return torch.zeros_like(tensor)
    return (tensor - min_value) / (max_value - min_value + eps)


def percentile_threshold(values: Iterable[float] | torch.Tensor, percentile: float) -> float:
    tensor = _to_float_tensor(values).flatten()
    if tensor.numel() == 0:
        return 0.0
    percentile = min(100.0, max(0.0, float(percentile)))
    rank = (percentile / 100.0) * max(tensor.numel() - 1, 0)
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    sorted_tensor, _ = torch.sort(tensor)
    if lower_index == upper_index:
        return float(sorted_tensor[lower_index].item())
    weight = rank - lower_index
    lower_value = sorted_tensor[lower_index]
    upper_value = sorted_tensor[upper_index]
    return float((lower_value + (upper_value - lower_value) * weight).item())


def adaptive_activation_map(tensor: torch.Tensor, max_spatial_tokens: int = 64) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError("Activation tensor must have at least batch and channel dimensions.")
    if tensor.ndim == 2:
        return tensor.detach().float()
    tensor = tensor.detach().float()
    spatial_dims = tensor.shape[2:]
    if not spatial_dims:
        return tensor
    target_edge = max(1, round(max_spatial_tokens ** (1.0 / len(spatial_dims))))
    if len(spatial_dims) == 3:
        return F.adaptive_avg_pool3d(tensor, output_size=(target_edge, target_edge, target_edge))
    if len(spatial_dims) == 2:
        return F.adaptive_avg_pool2d(tensor, output_size=(target_edge, target_edge))
    if len(spatial_dims) == 1:
        return F.adaptive_avg_pool1d(tensor, output_size=target_edge)
    raise ValueError("Unsupported activation dimensionality.")


def adaptive_token_features(tensor: torch.Tensor, max_spatial_tokens: int = 64) -> torch.Tensor:
    pooled = adaptive_activation_map(tensor, max_spatial_tokens=max_spatial_tokens)
    if pooled.ndim == 2:
        return pooled
    batch, channels = pooled.shape[:2]
    pooled = pooled.reshape(batch, channels, -1)
    return pooled.permute(0, 2, 1).reshape(-1, channels)


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("linear_cka expects 2D tensors")
    if x.shape[0] != y.shape[0]:
        raise ValueError("linear_cka requires the same number of observations")
    x = x.detach().float()
    y = y.detach().float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = x.transpose(0, 1) @ y
    x_gram = x.transpose(0, 1) @ x
    y_gram = y.transpose(0, 1) @ y
    numerator = torch.sum(cross * cross)
    denominator = torch.linalg.norm(x_gram, ord="fro") * torch.linalg.norm(y_gram, ord="fro")
    if float(denominator) <= eps:
        return 0.0
    return float((numerator / (denominator + eps)).item())


def rms_kurtosis(activations: torch.Tensor, eps: float = 1e-8) -> dict[str, float | list[float]]:
    if activations.ndim < 2:
        raise ValueError("rms_kurtosis expects at least batch and channel dimensions")
    activations = activations.detach().float()
    flattened = activations.reshape(activations.shape[0], activations.shape[1], -1)
    rms_per_sample_channel = torch.sqrt(torch.mean(flattened.square(), dim=2))
    numerator = torch.mean(rms_per_sample_channel.pow(4), dim=0).mean()
    denominator = torch.mean(rms_per_sample_channel.pow(2), dim=0).mean().pow(2)
    kurtosis = 0.0 if float(denominator) <= eps else float((numerator / (denominator + eps)).item())
    rms_channel = rms_per_sample_channel.mean(dim=0)
    return {
        "kurtosis": kurtosis,
        "rms_mean": float(rms_channel.mean().item()),
        "rms_std": float(rms_channel.std(unbiased=False).item()) if rms_channel.numel() > 1 else 0.0,
        "rms_per_channel": [float(value) for value in rms_channel.tolist()],
    }


def entropy_summary(logits: torch.Tensor, eps: float = 1e-8) -> dict[str, float]:
    if logits.ndim < 2:
        raise ValueError("entropy_summary expects logits with batch and class dimensions")
    logits = logits.detach().float()
    probabilities = torch.softmax(logits, dim=1)
    entropy = -(probabilities * torch.log(probabilities + eps)).sum(dim=1)
    flattened = entropy.reshape(-1)
    if flattened.numel() == 0:
        return {"E_min": 0.0, "E_max": 0.0, "E_mean": 0.0, "E_std": 0.0}
    return {
        "E_min": float(flattened.min().item()),
        "E_max": float(flattened.max().item()),
        "E_mean": float(flattened.mean().item()),
        "E_std": float(flattened.std(unbiased=False).item()) if flattened.numel() > 1 else 0.0,
    }