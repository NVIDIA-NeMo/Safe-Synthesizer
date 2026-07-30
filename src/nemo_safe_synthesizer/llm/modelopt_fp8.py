# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autograd-safe loading for ModelOpt FP8 training checkpoints."""

from __future__ import annotations

import json
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn import functional as F
from transformers.quantizers.auto import register_quantization_config, register_quantizer
from transformers.quantizers.base import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from ..errors import ParameterError

if TYPE_CHECKING:
    from .utils import ModelRef

MODELOPT_FP8_TRAINING_METHOD = "nss_modelopt_fp8"
MODELOPT_QUANT_CONFIG_FILENAME = "hf_quant_config.json"


class ModelOptFP8Linear(nn.Linear):
    """Linear layer with ModelOpt FP8 storage and BF16 autograd compute.

    ModelOpt's unified FP8 format stores a frozen E4M3 weight plus per-tensor
    weight and input scales. Training dequantizes the current layer only, so
    gradients can cross the frozen base while the resident weight stays FP8.
    """

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False) -> None:
        super().__init__(in_features, out_features, bias=bias)
        device = self.weight.device
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device=device),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(torch.empty(1, dtype=torch.float32, device=device), requires_grad=False)
        self.input_scale = nn.Parameter(torch.empty(1, dtype=torch.float32, device=device), requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base with a differentiable dequantization step."""
        scale = self.weight_scale.to(device=input.device, dtype=input.dtype)
        weight = self.weight.to(device=input.device, dtype=input.dtype) * scale
        bias = None if self.bias is None else self.bias.to(device=input.device, dtype=input.dtype)
        return F.linear(input, weight, bias)


@register_quantization_config(MODELOPT_FP8_TRAINING_METHOD)
class ModelOptFP8TrainingConfig(QuantizationConfigMixin):
    """Transformers loader configuration for a ModelOpt unified FP8 checkpoint."""

    def __init__(
        self,
        *,
        exclude_modules: list[str] | None = None,
        group_size: int = 16,
        producer_version: str | None = None,
        **_kwargs: object,
    ) -> None:
        self.quant_method = MODELOPT_FP8_TRAINING_METHOD  # ty: ignore[invalid-assignment] -- Transformers supports registered string quantization methods at runtime
        self.exclude_modules = exclude_modules or []
        self.group_size = group_size
        self.producer_version = producer_version


def _is_excluded(module_name: str, exclude_modules: list[str]) -> bool:
    return any(module_name == pattern or fnmatch(module_name, pattern) for pattern in exclude_modules)


@register_quantizer(MODELOPT_FP8_TRAINING_METHOD)
class ModelOptFP8TrainingHfQuantizer(HfQuantizer):
    """Load ModelOpt FP8 storage into differentiable dequantizing linears."""

    requires_calibration = False
    quantization_config: ModelOptFP8TrainingConfig

    def validate_environment(self, *args, **kwargs) -> None:
        if not self.pre_quantized:
            raise ValueError("ModelOpt FP8 training requires a prequantized checkpoint")
        device_map = kwargs.get("device_map")
        if isinstance(device_map, dict):
            placements = device_map.values()
            if any(
                (isinstance(placement, torch.device) and placement.type == "cpu")
                or (isinstance(placement, str) and placement.casefold() in {"cpu", "disk"})
                for placement in placements
            ):
                raise RuntimeError("ModelOpt FP8 training does not support CPU or disk offload")
        if not torch.cuda.is_available():
            raise RuntimeError("ModelOpt FP8 training requires a CUDA GPU with compute capability 8.9 or newer")
        capability = torch.cuda.get_device_capability()
        if capability < (8, 9):
            actual = ".".join(str(part) for part in capability)
            raise RuntimeError(f"ModelOpt FP8 training requires compute capability 8.9 or newer; detected {actual}")

    def update_dtype(self, dtype: torch.dtype) -> torch.dtype:
        return torch.bfloat16

    def _normalize_exclude_modules(self, model) -> list[str]:
        """Map export-side module names onto the native Transformers graph."""
        from transformers.conversion_mapping import get_model_conversion_mapping

        normalized: list[str] = []
        renamings = get_model_conversion_mapping(model)
        for module_name in self.quantization_config.exclude_modules:
            renamed = module_name
            for renaming in renamings:
                renamed, _ = renaming.rename_source_key(renamed)
            normalized.append(renamed)
        return normalized

    def _process_model_before_weight_loading(self, model, **kwargs):
        replacements: list[tuple[str, nn.Linear]] = []
        excluded = self._normalize_exclude_modules(model)
        for module_name, module in model.named_modules():
            if type(module) is nn.Linear and not _is_excluded(module_name, excluded):
                replacements.append((module_name, module))

        for module_name, module in replacements:
            with torch.device(module.weight.device):
                replacement = ModelOptFP8Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                )
            model.set_submodule(module_name, replacement)
        return model

    def _process_model_after_weight_loading(self, model, **kwargs):
        if not any(isinstance(module, ModelOptFP8Linear) for module in model.modules()):
            raise ValueError("The ModelOpt FP8 checkpoint did not replace any linear modules")
        return model

    def is_serializable(self) -> bool:
        return True

    @property
    def is_trainable(self) -> bool:
        return True


def _resolve_quant_config_path(model_ref: ModelRef) -> Path:
    """Resolve the ModelOpt sidecar locally or through Hugging Face Hub."""
    if model_ref.local_path is not None:
        candidate = model_ref.local_path / MODELOPT_QUANT_CONFIG_FILENAME
        if candidate.is_file():
            return candidate

    if model_ref.repo_id is not None:
        try:
            downloaded = hf_hub_download(
                repo_id=model_ref.repo_id,
                filename=MODELOPT_QUANT_CONFIG_FILENAME,
                revision=model_ref.revision,
                cache_dir=str(model_ref.cache_root) if model_ref.cache_root is not None else None,
            )
        except OSError as exc:
            raise ParameterError(
                f"Could not load {MODELOPT_QUANT_CONFIG_FILENAME} for the ModelOpt FP8 training checkpoint"
            ) from exc
        return Path(downloaded)

    raise ParameterError(f"ModelOpt FP8 training requires a checkpoint containing {MODELOPT_QUANT_CONFIG_FILENAME}")


def _read_quant_config(quant_config_path: Path) -> dict:
    """Read a ModelOpt quantization sidecar as a JSON object."""
    try:
        data = json.loads(quant_config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ParameterError(f"Invalid ModelOpt quantization sidecar: {quant_config_path}") from exc
    if not isinstance(data, dict):
        raise ParameterError(f"Invalid ModelOpt quantization sidecar: {quant_config_path}")
    return data


def load_modelopt_fp8_training_config(model_ref: ModelRef) -> ModelOptFP8TrainingConfig:
    """Load and validate the ModelOpt sidecar for a direct FP8 training base."""
    data = _read_quant_config(_resolve_quant_config_path(model_ref))

    producer = data.get("producer")
    quantization = data.get("quantization")
    if not isinstance(producer, dict) or producer.get("name", "").casefold() != "modelopt":
        raise ParameterError("The FP8 training checkpoint was not produced by NVIDIA ModelOpt")
    if not isinstance(quantization, dict) or quantization.get("quant_algo", "").casefold() != "fp8":
        raise ParameterError("The ModelOpt training checkpoint does not declare FP8 weights")

    group_size = quantization.get("group_size", 16)
    exclude_modules = quantization.get("exclude_modules", [])
    if not isinstance(group_size, int) or group_size != 16:
        raise ParameterError(f"Unsupported ModelOpt FP8 group size: {group_size!r}; expected 16")
    if not isinstance(exclude_modules, list) or not all(isinstance(name, str) for name in exclude_modules):
        raise ParameterError("ModelOpt FP8 exclude_modules must be a list of module names")

    return ModelOptFP8TrainingConfig(
        exclude_modules=exclude_modules,
        group_size=group_size,
        producer_version=producer.get("version"),
    )
