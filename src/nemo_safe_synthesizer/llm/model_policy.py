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

from ..errors import ParameterError

NEMOTRON3_NANO_4B_BF16 = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
NEMOTRON3_NANO_4B_FP8 = "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8"


@dataclass(frozen=True)
class ModelPolicy:
    """Runtime and training policy for an exact supported model family."""

    canonical_ids: frozenset[str]
    uses_rope: bool = True
    automatic_lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    vllm_kwargs: tuple[tuple[str, object], ...] = ()
    force_native_transformers: bool = False
    minimum_training_compute_capability: tuple[int, int] | None = None
    minimum_generation_compute_capability: tuple[int, int] | None = None

    def matches(self, repo_id: str | None) -> bool:
        """Return whether ``repo_id`` is one of this policy's exact identifiers."""
        if repo_id is None:
            return False
        return repo_id.casefold() in {model_id.casefold() for model_id in self.canonical_ids}

    def engine_kwargs(self) -> dict[str, object]:
        """Return a mutable copy of the model-specific vLLM arguments."""
        return dict(self.vllm_kwargs)


_NEMOTRON3_NANO_LORA_TARGETS = (
    "in_proj",
    "up_proj",
    "down_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)

NEMOTRON3_NANO_POLICY = ModelPolicy(
    canonical_ids=frozenset({NEMOTRON3_NANO_4B_BF16}),
    uses_rope=False,
    automatic_lora_targets=_NEMOTRON3_NANO_LORA_TARGETS,
    vllm_kwargs=(
        ("mamba_ssm_cache_dtype", "float32"),
        ("max_num_seqs", 8),
    ),
    force_native_transformers=True,
)

NEMOTRON3_NANO_FP8_POLICY = ModelPolicy(
    canonical_ids=frozenset({NEMOTRON3_NANO_4B_FP8}),
    uses_rope=False,
    automatic_lora_targets=_NEMOTRON3_NANO_LORA_TARGETS,
    vllm_kwargs=(
        ("mamba_ssm_cache_dtype", "float32"),
        ("max_num_seqs", 8),
        ("kv_cache_dtype", "fp8"),
    ),
    force_native_transformers=True,
    minimum_training_compute_capability=(8, 9),
    minimum_generation_compute_capability=(8, 9),
)

_POLICIES = (NEMOTRON3_NANO_POLICY, NEMOTRON3_NANO_FP8_POLICY)

NEMOTRON3_NANO_LAYER_BLOCK_TYPES = (
    "mamba",
    "mlp",
    "mamba",
    "mlp",
    "mamba",
    "mlp",
    "mamba",
    "mamba",
    "mlp",
    "mamba",
    "mlp",
    "mamba",
    "attention",
    "mlp",
    "mamba",
    "mlp",
    "mamba",
    "attention",
    "mlp",
    "mamba",
    "mlp",
    "mamba",
    "mlp",
    "mamba",
    "attention",
    "mlp",
    "mamba",
    "mlp",
    "mamba",
    "mlp",
    "mamba",
    "mamba",
    "attention",
    "mlp",
    "mamba",
    "mamba",
    "mamba",
    "mlp",
    "mamba",
    "mlp",
    "mamba",
    "mlp",
)

NEMOTRON3_NANO_HYBRID_OVERRIDE_PATTERN = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"

_NEMOTRON3_NANO_CONFIG_SIGNATURE = {
    "architectures": ["NemotronHForCausalLM"],
    "hidden_size": 3136,
    "mamba_head_dim": 80,
    "mamba_num_heads": 96,
    "model_type": "nemotron_h",
    "ssm_state_size": 128,
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
    dtype = config.get("dtype", config.get("torch_dtype"))
    layer_types = config.get("layers_block_type")
    if isinstance(layer_types, list):
        layer_layout_matches = tuple(layer_types) == NEMOTRON3_NANO_LAYER_BLOCK_TYPES
    else:
        layer_layout_matches = (
            config.get("num_hidden_layers") == len(NEMOTRON3_NANO_LAYER_BLOCK_TYPES)
            and config.get("hybrid_override_pattern") == NEMOTRON3_NANO_HYBRID_OVERRIDE_PATTERN
        )
    if (
        dtype == "bfloat16"
        and layer_layout_matches
        and all(config.get(key) == value for key, value in _NEMOTRON3_NANO_CONFIG_SIGNATURE.items())
    ):
        quant_config_path = model_path / "hf_quant_config.json"
        if quant_config_path.exists():
            try:
                quant_config = json.loads(quant_config_path.read_text())
            except (OSError, json.JSONDecodeError):
                return None
            producer = quant_config.get("producer", {})
            quantization = quant_config.get("quantization", {})
            if (
                not isinstance(producer, dict)
                or producer.get("name", "").casefold() != "modelopt"
                or not isinstance(quantization, dict)
                or quantization.get("quant_algo", "").casefold() != "fp8"
            ):
                return None
            return NEMOTRON3_NANO_FP8_POLICY
        return NEMOTRON3_NANO_POLICY
    return None


def model_policy_for_reference(repo_id: str | None, local_path: Path | None) -> ModelPolicy | None:
    """Resolve an exact Hub/cache policy or a fingerprinted local checkpoint policy."""
    return model_policy_for(repo_id) or model_policy_for_local_path(local_path)


def validate_generation_model_pair(
    training_repo_id: str | None,
    generation_repo_id: str | None,
) -> None:
    """Reject a generation override outside an explicitly compatible repository pair."""
    if training_repo_id is None or generation_repo_id is None:
        raise ParameterError("The requested generation base is not compatible with the adapter's training model")

    pair = (training_repo_id.casefold(), generation_repo_id.casefold())
    if pair[0] == pair[1]:
        return
    if pair == (NEMOTRON3_NANO_4B_BF16.casefold(), NEMOTRON3_NANO_4B_FP8.casefold()):
        return
    raise ParameterError("The requested generation base is not compatible with the adapter's training model")


def _load_local_mamba_runtime() -> ModuleType:
    """Load local Mamba kernels without leaking a temporary package module."""
    had_previous_mamba_ssm = "mamba_ssm" in sys.modules
    previous_mamba_ssm = sys.modules.get("mamba_ssm")
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
    except Exception as exc:
        raise ParameterError(
            "Nemotron 3 Nano training could not import the compiled mamba-ssm runtime; "
            "run `mise run bootstrap-nemotron-kernels`"
        ) from exc
    finally:
        if had_previous_mamba_ssm:
            assert previous_mamba_ssm is not None
            sys.modules["mamba_ssm"] = previous_mamba_ssm
        else:
            sys.modules.pop("mamba_ssm", None)
    return mamba_ssm


def configure_local_training_kernels(repo_id: str | None, local_path: Path | None = None) -> None:
    """Register compiled Nemotron kernels before native Transformers model loading.

    Transformers otherwise prefers Kernel Hub whenever its ``kernels`` package
    is installed, even when compatible local modules are available.
    """
    if model_policy_for_reference(repo_id, local_path) not in (
        NEMOTRON3_NANO_POLICY,
        NEMOTRON3_NANO_FP8_POLICY,
    ):
        return

    try:
        from transformers.integrations import hub_kernels

        causal_conv1d = importlib.import_module("causal_conv1d")
    except ImportError as exc:
        raise ParameterError(
            "Nemotron 3 Nano training requires the compiled causal-conv1d and mamba-ssm packages; "
            "run `mise run bootstrap-nemotron-kernels`"
        ) from exc
    mamba_ssm = _load_local_mamba_runtime()

    # Transformers 5.12 exposes no public local-kernel registration API.
    # Populate its loader cache so native Nemotron-H resolves the installed
    # modules without a Kernel Hub network request.
    hub_kernels._KERNEL_MODULE_MAPPING.update(  # noqa: SLF001
        {"causal-conv1d": causal_conv1d, "mamba-ssm": mamba_ssm}
    )
