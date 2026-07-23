# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact model-family policies shared by loading, training, and generation."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from ..errors import ParameterError

NEMOTRON3_NANO_4B_BF16 = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"


@dataclass(frozen=True)
class ModelPolicy:
    """Runtime and training policy for an exact supported model family."""

    canonical_ids: frozenset[str]
    uses_rope: bool = True
    automatic_lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    vllm_kwargs: tuple[tuple[str, object], ...] = ()
    force_native_transformers: bool = False

    def matches(self, repo_id: str | None) -> bool:
        """Return whether ``repo_id`` is one of this policy's exact identifiers."""
        if repo_id is None:
            return False
        return repo_id.casefold() in {model_id.casefold() for model_id in self.canonical_ids}

    def engine_kwargs(self) -> dict[str, object]:
        """Return a mutable copy of the model-specific vLLM arguments."""
        return dict(self.vllm_kwargs)


NEMOTRON3_NANO_POLICY = ModelPolicy(
    canonical_ids=frozenset({NEMOTRON3_NANO_4B_BF16}),
    uses_rope=False,
    automatic_lora_targets=(
        "in_proj",
        "up_proj",
        "down_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ),
    vllm_kwargs=(
        ("mamba_ssm_cache_dtype", "float32"),
        ("max_num_seqs", 8),
    ),
    force_native_transformers=True,
)

_POLICIES = (NEMOTRON3_NANO_POLICY,)

_NEMOTRON3_NANO_CONFIG_SIGNATURE = {
    "architectures": ["NemotronHForCausalLM"],
    "hidden_size": 3136,
    "mamba_head_dim": 80,
    "mamba_num_heads": 96,
    "model_type": "nemotron_h",
    "num_hidden_layers": 42,
    "ssm_state_size": 128,
    "torch_dtype": "bfloat16",
    "vocab_size": 131072,
}


def model_policy_for(repo_id: str | None) -> ModelPolicy | None:
    """Resolve an exact model policy without substring matching."""
    return next((policy for policy in _POLICIES if policy.matches(repo_id)), None)


def model_policy_for_local_path(model_path: Path | None) -> ModelPolicy | None:
    """Resolve a policy from an exact local checkpoint configuration fingerprint."""
    if model_path is None:
        return None
    config_path = model_path / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if config.get("quantization_config") is not None:
        return None
    if all(config.get(key) == value for key, value in _NEMOTRON3_NANO_CONFIG_SIGNATURE.items()):
        return NEMOTRON3_NANO_POLICY
    return None


def model_policy_for_reference(repo_id: str | None, local_path: Path | None) -> ModelPolicy | None:
    """Resolve an exact Hub/cache policy or a fingerprinted local checkpoint policy."""
    return model_policy_for(repo_id) or model_policy_for_local_path(local_path)


def configure_local_training_kernels(repo_id: str | None, local_path: Path | None = None) -> None:
    """Register compiled Nemotron kernels before native Transformers model loading.

    Transformers otherwise prefers Kernel Hub whenever its ``kernels`` package
    is installed, even when compatible local modules are available.
    """
    if model_policy_for_reference(repo_id, local_path) is not NEMOTRON3_NANO_POLICY:
        return

    try:
        import causal_conv1d
        from transformers.integrations import hub_kernels
    except ImportError as exc:
        raise ParameterError(
            "Nemotron 3 Nano BF16 training requires the compiled causal-conv1d and mamba-ssm packages; "
            "run `mise run bootstrap-nemotron-kernels`"
        ) from exc

    try:
        package_root = importlib.metadata.distribution("mamba-ssm").locate_file("mamba_ssm")
        mamba_ssm = ModuleType("mamba_ssm")
        mamba_ssm.__path__ = [str(package_root)]
        mamba_ssm.__package__ = "mamba_ssm"
        sys.modules["mamba_ssm"] = mamba_ssm
        selective_state_update = importlib.import_module("mamba_ssm.ops.triton.selective_state_update")
        ssd_combined = importlib.import_module("mamba_ssm.ops.triton.ssd_combined")
        setattr(mamba_ssm, "selective_state_update", selective_state_update.selective_state_update)
        setattr(mamba_ssm, "mamba_chunk_scan_combined", ssd_combined.mamba_chunk_scan_combined)
        setattr(mamba_ssm, "mamba_split_conv1d_scan_combined", ssd_combined.mamba_split_conv1d_scan_combined)
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise ParameterError(
            "Nemotron 3 Nano BF16 training could not import the compiled mamba-ssm runtime; "
            "run `mise run bootstrap-nemotron-kernels`"
        ) from exc

    # Transformers 5.12 exposes no public local-kernel registration API.
    # Populate its loader cache so native Nemotron-H resolves the installed
    # modules without a Kernel Hub network request.
    hub_kernels._KERNEL_MODULE_MAPPING.update(  # noqa: SLF001
        {"causal-conv1d": causal_conv1d, "mamba-ssm": mamba_ssm}
    )


def validate_lora_targets(model: Any, target_suffixes: list[str]) -> dict[str, int]:
    """Return per-suffix module counts or reject a partially unmatched target set."""
    counts = {
        suffix: sum(1 for name, _module in model.named_modules() if name == suffix or name.endswith(f".{suffix}"))
        for suffix in target_suffixes
    }
    missing = [suffix for suffix, count in counts.items() if count == 0]
    if missing:
        raise ParameterError(f"LoRA target modules did not match the loaded model: {', '.join(missing)}")
    return counts
