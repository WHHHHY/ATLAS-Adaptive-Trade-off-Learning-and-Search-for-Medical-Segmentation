from __future__ import annotations

import re
import time
from typing import Any

import torch
import torch.nn as nn


_ENCODER_PATTERN = re.compile(r"^encoder(\d+)")
_DECODER_PATTERN = re.compile(r"^decoder(\d+)")


def _module_unit_id(name: str) -> str | None:
    match = _ENCODER_PATTERN.match(name)
    if match:
        return f"unit_{int(match.group(1))}"
    match = _DECODER_PATTERN.match(name)
    if match:
        return f"unit_{int(match.group(1))}"
    return None


def _conv3d_flops(module: nn.Conv3d, output: torch.Tensor) -> float:
    out_elements = output.numel()
    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2] * (module.in_channels / module.groups)
    return float(out_elements * kernel_ops * 2.0)


def _linear_flops(module: nn.Linear, output: torch.Tensor) -> float:
    out_elements = output.numel()
    return float(out_elements * module.in_features * 2.0)


def _layernorm_flops(module: nn.LayerNorm, output: torch.Tensor) -> float:
    del module
    return float(output.numel() * 5.0)


def _collect_parameter_counts(model: nn.Module) -> tuple[int, dict[str, int]]:
    total = 0
    unit_params: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        unit_id = _module_unit_id(name)
        if unit_id is not None:
            unit_params[unit_id] = unit_params.get(unit_id, 0) + count
    return total, unit_params


def profile_model_resources(
    model: nn.Module,
    sample_input: torch.Tensor,
    unit_quant_config: dict[str, dict[str, int]] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    unit_quant_config = unit_quant_config or {}
    named_modules = dict(model.named_modules())
    flops_total = 0.0
    flops_by_unit: dict[str, float] = {}
    handles = []

    def register_hook(module_name: str, module: nn.Module) -> None:
        def hook(_module, _inputs, output):
            nonlocal flops_total
            if isinstance(output, tuple):
                output_tensor = output[0]
            else:
                output_tensor = output
            if not isinstance(output_tensor, torch.Tensor):
                return
            if isinstance(module, nn.Conv3d):
                flops = _conv3d_flops(module, output_tensor)
            elif isinstance(module, nn.Linear):
                flops = _linear_flops(module, output_tensor)
            elif isinstance(module, nn.LayerNorm):
                flops = _layernorm_flops(module, output_tensor)
            else:
                return
            flops_total += flops
            unit_id = _module_unit_id(module_name)
            if unit_id is not None:
                flops_by_unit[unit_id] = flops_by_unit.get(unit_id, 0.0) + flops

        handles.append(module.register_forward_hook(hook))

    for module_name, module in named_modules.items():
        if isinstance(module, (nn.Conv3d, nn.Linear, nn.LayerNorm)):
            register_hook(module_name, module)

    model.eval()
    with torch.no_grad():
        if device is not None and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        latency_start = time.perf_counter()
        _ = model(sample_input)
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - latency_start) * 1000.0
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device is not None and device.type == "cuda" else 0.0

    for handle in handles:
        handle.remove()

    params_total, params_by_unit = _collect_parameter_counts(model)
    effective_params = float(params_total)
    effective_flops = float(flops_total)
    effective_ram = float(peak_vram_mb)
    effective_latency_ms = float(latency_ms)
    for unit_id, quant_cfg in unit_quant_config.items():
        weight_bits = int(quant_cfg.get("weight_bits", 32))
        act_bits = int(quant_cfg.get("act_bits", 32))
        weight_ratio = weight_bits / 32.0
        act_ratio = act_bits / 32.0
        unit_params = float(params_by_unit.get(unit_id, 0))
        unit_flops = float(flops_by_unit.get(unit_id, 0.0))
        effective_params -= unit_params
        effective_params += unit_params * weight_ratio
        effective_flops -= unit_flops
        effective_flops += unit_flops * ((weight_ratio + act_ratio) / 2.0)
        effective_ram *= 1.0 - 0.05 * (1.0 - act_ratio)
        effective_latency_ms *= 1.0 - 0.03 * (1.0 - ((weight_ratio + act_ratio) / 2.0))

    return {
        "params_total": int(params_total),
        "params_effective": float(effective_params),
        "flops_g": float(flops_total / 1_000_000_000.0),
        "flops_effective_g": float(effective_flops / 1_000_000_000.0),
        "peak_vram_mb": float(peak_vram_mb),
        "peak_vram_effective_mb": float(max(effective_ram, 0.0)),
        "latency_ms": float(max(latency_ms, 1e-6)),
        "latency_effective_ms": float(max(effective_latency_ms, 1e-6)),
        "params_by_unit": params_by_unit,
        "flops_by_unit": flops_by_unit,
    }