# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-stage checks: GPU, VRAM, tokens, log settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from typing_extensions import override

from ...llm.utils import ModelRef
from ...observability import get_logger
from ...utils import hf_offline_enabled
from ..base import ConfigCheck, IssueCollector, MetadataCheck
from ..helpers import require_import
from ..types import ConfigView, MetadataView

logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...config.training import TrainingHyperparams

__all__ = [
    "CUDAAvailabilityCheck",
    "HFModelAvailabilityCheck",
    "InferenceModelCheck",
    "VRAMComponentEstimate",
    "VRAMHeadroomCheck",
    "bytes_per_base_weight",
    "estimate_base_model_params",
    "estimate_params_from_shape",
    "estimate_training_vram_components",
    "param_count_from_empty_model",
]


class CUDAAvailabilityCheck(ConfigCheck):
    """Validate CUDA GPU availability."""

    name = "gpu.cuda"
    label = "CUDA availability"
    category = "environment"

    @override
    def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
        torch = require_import(
            collector,
            "torch",
            code="torch_missing",
            message="PyTorch is not installed; cannot verify GPU availability.",
        )
        if torch is None:
            return

        if torch.cuda.is_available():
            return

        collector.error("no_gpu", "No CUDA GPU detected. Safe Synthesizer requires a CUDA-capable GPU.")


def param_count_from_empty_model(autoconfig: PretrainedConfig) -> int | None:
    """Count parameters by instantiating the model on the ``meta`` device.

    ``accelerate.init_empty_weights`` constructs the full ``nn.Module`` graph
    with every parameter on ``torch.device("meta")`` -- no storage is
    allocated and no weights are downloaded. ``AutoModelForCausalLM.from_config``
    consults the transformers model-class registry to pick the right
    architecture (handling Nemotron's non-gated MLP, MoE experts, biases,
    tied embeddings, and any future variant automatically).

    Returns ``None`` if accelerate/transformers are missing, the config
    doesn't map to a registered architecture (e.g. ``trust_remote_code``
    custom archs), or instantiation fails for any other reason. The caller
    should fall back to
    [estimate_params_from_shape][nemo_safe_synthesizer.preflight.checks.environment.estimate_params_from_shape].

    References:
        - HuggingFace accelerate, "Big Model Inference" --
          <https://huggingface.co/docs/accelerate/concept_guides/big_model_inference>
        - HuggingFace accelerate, "Model memory estimator" -- same
          meta-device technique exposed as ``accelerate estimate-memory``;
          reported accurate to within a few percent of real CUDA load.
          <https://huggingface.co/docs/accelerate/usage_guides/model_size_estimator>
        - PyTorch meta device --
          <https://docs.pytorch.org/docs/stable/meta.html>
    """
    try:
        from accelerate import init_empty_weights
        from transformers import AutoModelForCausalLM
    except ImportError:
        return None
    try:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(autoconfig)
        return sum(p.numel() for p in model.parameters())
    except Exception as exc:
        logger.runtime.debug(
            "param_count_from_empty_model failed for %r: %s: %s",
            getattr(autoconfig, "_name_or_path", None) or getattr(autoconfig, "model_type", "unknown"),
            type(exc).__name__,
            exc,
        )
        return None


def estimate_params_from_shape(autoconfig: PretrainedConfig) -> int | None:
    r"""Shape-only fallback param count used when meta-tensor construction fails.

    Models a decoder-only transformer with grouped-query attention (which
    degrades to multi-head when ``num_key_value_heads == num_attention_heads``)
    and a gated SwiGLU/GeGLU MLP -- the shape NSS sees on its supported
    model families (Llama, Qwen, Mistral, SmolLM, Granite, TinyLlama). For
    non-gated variants (e.g. Nemotron's squared-ReLU MLP) this over-counts
    MLP params by 50%, which is why the meta-tensor path is preferred.

    With hidden size \(H\), intermediate size \(I\), \(L\) layers,
    vocabulary \(V\), \(n_\text{kv}\) KV heads and per-head dim \(d\), the
    per-layer cost is

    \[
        \text{attn} = 2 H^2 + 2 H \, n_\text{kv} \, d, \qquad
        \text{mlp}  = 3 H I
    \]

    (full Q/O projections; K/V shrunk by GQA; gate/up/down for SwiGLU).
    Total parameters:

    \[
        N = V H + L (\text{attn} + \text{mlp}) + \begin{cases}
            0      & \text{tied embeddings} \\
            V H    & \text{untied LM head}
        \end{cases}
    \]

    References:
        - Ainslie, J. et al. "GQA: Training Generalized Multi-Query
          Transformer Models from Multi-Head Checkpoints" (2023) --
          reduced K/V projection shape. <https://arxiv.org/abs/2305.13245>
        - So, D. R. et al. "Primer: Searching for Efficient Transformers
          for Language Modeling" (2021) -- squared-ReLU MLP (Nemotron
          family), 2 projections; motivates the 50% over-count caveat
          above. <https://arxiv.org/abs/2109.08668>
    """
    H = getattr(autoconfig, "hidden_size", None)
    L = getattr(autoconfig, "num_hidden_layers", None)
    if not (H and L):
        return None
    V = getattr(autoconfig, "vocab_size", 32_000) or 32_000
    inter = getattr(autoconfig, "intermediate_size", None) or 4 * H
    n_heads = getattr(autoconfig, "num_attention_heads", None) or max(1, H // 64)
    kv_heads = getattr(autoconfig, "num_key_value_heads", None) or n_heads
    head_dim = getattr(autoconfig, "head_dim", None) or max(1, H // max(n_heads, 1))
    tied = bool(getattr(autoconfig, "tie_word_embeddings", False))

    # Q proj (H×H) + O proj (H×H) = 2H²; K proj + V proj shrunk by GQA: 2·H·kv_heads·d
    attn = 2 * H * H + 2 * H * kv_heads * head_dim
    # gate proj (H×I) + up proj (H×I) + down proj (I×H) = 3HI  (SwiGLU / GeGLU gated MLP)
    mlp = 3 * H * inter
    per_layer = attn + mlp
    embed = V * H  # token embedding table
    lm_head = 0 if tied else V * H  # unembedding; zero when tied to embed
    return embed + L * per_layer + lm_head


def estimate_base_model_params(autoconfig: PretrainedConfig) -> tuple[int, Literal["exact", "approximate"]] | None:
    r"""Return ``(n_params, method)`` for the base model, or ``None`` if unknown.

    ``method == "exact"`` means the meta-tensor path succeeded and the count
    is architecture-accurate. ``method == "approximate"`` means the shape
    formula was used as a fallback (see
    [estimate_params_from_shape][nemo_safe_synthesizer.preflight.checks.environment.estimate_params_from_shape]
    for its known error modes) and the caller should flag the downstream VRAM
    estimate as heuristic. Benchmarked fallback error on supported
    architectures: \(-22\%\) to \(+33\%\); hybrid Mamba-Transformer models
    (e.g. Nemotron-H) can drift further.
    """
    exact = param_count_from_empty_model(autoconfig)
    if exact is not None:
        return exact, "exact"
    approx = estimate_params_from_shape(autoconfig)
    if approx is None:
        return None
    return approx, "approximate"


def bytes_per_base_weight(training_cfg: TrainingHyperparams) -> float:
    r"""Return expected bytes/param for the base model load mode.

    NSS always trains via LoRA-style adapters, so the base model's storage
    precision dominates VRAM (LoRA adapter params, gradients, and
    optimizer state are comparatively negligible).

    Runtime quantization is controlled by ``training.quantize_model``. The
    PEFT type string alone is not enough: with ``quantize_model=False`` the
    base weights are loaded as bf16 even when ``peft_implementation`` is
    configured as ``"QLORA"``.

    - Quantized load: \(\text{bits}/8 + 0.1\) to cover quant state (absmax /
      block scales) and dequant workspace. Yields \(\approx 0.6\) for 4-bit,
      \(\approx 1.1\) for 8-bit.
    - Unquantized load: \(2\) bytes (bf16/fp16 base weights).

    References:
        - Hu, E. J. et al. "LoRA: Low-Rank Adaptation of Large Language
          Models" (2021) -- base weights frozen; adapter + gradients +
          optimizer state are small relative to \(N b\).
          <https://arxiv.org/abs/2106.09685>
        - Dettmers, T. et al. "QLoRA: Efficient Finetuning of Quantized
          LLMs" (2023) -- 4-bit NF4 quantization with block-wise absmax
          scales; the \(+0.1\) term accounts for these scales and the
          dequant workspace. <https://arxiv.org/abs/2305.14314>
    """
    if training_cfg.quantize_model:
        # Prefer the explicit scheme if set; otherwise fall back to the legacy
        # bits-based field. Both routes yield bits/param for memory estimation.
        if training_cfg.quantization_scheme is not None:
            bits = training_cfg.quantization_scheme.effective_bits
        else:
            bits = training_cfg.quantization_bits
        return bits / 8 + 0.1
    return 2.0


_VRAM_LEGACY_OVERHEAD_GIB = 2.0
r"""Legacy overhead when activation memory cannot be modeled.

Covers CUDA context, unchecked activations (under gradient checkpointing),
kernels, LoRA adapters, and optimizer footprint at a coarse level.

References:
    - Korthikanti, V. et al. "Reducing Activation Recomputation in Large
      Transformer Models" (2022) -- activation-memory scaling under
      selective recomputation / gradient checkpointing.
      <https://arxiv.org/abs/2205.05198>
"""

_VRAM_KERNEL_RESERVED_GIB = 0.5
"""Small reservation when activation memory is modeled explicitly (kernels, graphs)."""

_VRAM_HARD_FAIL_RATIO = 1.5
"""Error instead of warning when the estimate is this many times available VRAM."""


@dataclass(frozen=True)
class VRAMComponentEstimate:
    """Per-device training VRAM components for ``gpu.vram`` preflight."""

    base_weights_gib: float
    overhead_gib: float
    activation_gib: float | None
    total_gib: float


def _positive_int_scalar(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def activation_memory_gib(
    *,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_hidden_layers: int,
    bytes_per_activation_element: float = 2.0,
) -> float:
    r"""Rough activation VRAM on one device given micro-batch geometry.

    Uses ``training.batch_size`` (HF ``per_device_train_batch_size``), not
    ``gradient_accumulation_steps``. Matches bf16-ish training tensors at
    2 bytes/element:

    \[
        M_\text{act} \approx B \cdot S \cdot H \cdot L \cdot 2\text{ bytes}
    \]

    Omit attention \(O(B S^2)\) blocks and recomputation specifics; goal is
    order-of-magnitude headroom versus absurd ``batch_size`` values.

    References:
        - Korthikanti, V. et al. (2022) -- recomputation vs stored activations.
          <https://arxiv.org/abs/2205.05198>
    """
    nbytes = batch_size * seq_len * hidden_size * num_hidden_layers * bytes_per_activation_element
    return nbytes / (1024**3)


def estimate_training_vram_components(
    *,
    n_params: int,
    training_cfg: TrainingHyperparams,
    batch_size: int,
    seq_len: int | None,
    hidden_size: int | None,
    num_hidden_layers: int | None,
    bytes_per_activation_element: float = 2.0,
) -> VRAMComponentEstimate:
    """Compose base weights, overhead, and optional activation estimate (GiB)."""
    bpw = bytes_per_base_weight(training_cfg)
    base_weights_gib = (n_params * bpw) / (1024**3)

    b_sz = _positive_int_scalar(batch_size)
    seq = _positive_int_scalar(seq_len) if seq_len is not None else None
    h_sz = _positive_int_scalar(hidden_size) if hidden_size is not None else None
    n_layers = _positive_int_scalar(num_hidden_layers) if num_hidden_layers is not None else None

    activation_gib: float | None
    overhead_gib: float
    if b_sz is not None and seq is not None and h_sz is not None and n_layers is not None:
        activation_gib = activation_memory_gib(
            batch_size=b_sz,
            seq_len=seq,
            hidden_size=h_sz,
            num_hidden_layers=n_layers,
            bytes_per_activation_element=bytes_per_activation_element,
        )
        overhead_gib = _VRAM_KERNEL_RESERVED_GIB
    else:
        activation_gib = None
        overhead_gib = _VRAM_LEGACY_OVERHEAD_GIB

    total_gib = base_weights_gib + overhead_gib + (activation_gib if activation_gib is not None else 0.0)

    return VRAMComponentEstimate(
        base_weights_gib=base_weights_gib,
        overhead_gib=overhead_gib,
        activation_gib=activation_gib,
        total_gib=total_gib,
    )


class VRAMHeadroomCheck(MetadataCheck):
    r"""Estimate whether GPU VRAM is sufficient for training.

    The estimate is intentionally *conservative/heuristic*, not worst-case-accurate.

    Parameter counts come from ``estimate_base_model_params`` via meta tensors
    when possible.

    Activation memory uses ``estimate_training_vram_components`` when
    ``metadata.max_seq_length`` and transformer shape fields resolve to
    positive integers; missing inputs leave activations unspecified and revert
    to a legacy lumped overhead. Per-device VRAM compares against
    ``get_max_vram(max_vram_fraction=training.max_vram_fraction)`` headroom.

    LoRA adapters, full optimizer footprint, \(O(B S^2)\) attention material,
    and quantization workspace are partially covered only by residual overhead --
    passing does not guarantee a fit; failing is a strong signal of OOM risk.

    References:
        - EleutherAI, "Transformer Math 101".
          <https://blog.eleuther.ai/transformer-math/>
    """

    name = "gpu.vram"
    label = "VRAM headroom"
    category = "environment"
    requires = ("gpu.cuda",)

    @override
    def check(self, ctx: MetadataView, collector: IssueCollector) -> None:
        import torch

        from ...llm.utils import get_max_vram

        config = ctx.config
        vram_map = get_max_vram(max_vram_fraction=config.training.max_vram_fraction)
        # ModelMetadata.autoconfig is populated by ``from_config`` but set to
        # ``None`` by ``ModelMetadata.stub`` (used on ``--validate`` when the
        # model is not cached / no network). Skip in that case: without the
        # model shape we cannot estimate parameter count.
        autoconfig = getattr(ctx.metadata, "autoconfig", None)
        if not vram_map or autoconfig is None:
            return

        result = estimate_base_model_params(autoconfig)
        if result is None:
            return
        n_params, method = result
        model_name = getattr(autoconfig, "_name_or_path", None) or getattr(autoconfig, "model_type", "model")
        if method == "exact":
            logger.info(
                "VRAM estimate: counted %.2fB parameters for %s via meta-tensor instantiation",
                n_params / 1e9,
                model_name,
            )
        else:
            logger.info(
                "VRAM estimate: meta-tensor instantiation unavailable for %s; falling back to shape "
                "heuristic (~%.2fB parameters). This estimate is approximate; actual VRAM usage may "
                "differ by 20-30%% or more for non-standard architectures (e.g. Nemotron, Mamba hybrids).",
                model_name,
                n_params / 1e9,
            )

        seq_len = getattr(ctx.metadata, "max_seq_length", None)
        comp = estimate_training_vram_components(
            n_params=n_params,
            training_cfg=config.training,
            batch_size=config.training.batch_size,
            seq_len=seq_len,
            hidden_size=getattr(autoconfig, "hidden_size", None),
            num_hidden_layers=getattr(autoconfig, "num_hidden_layers", None),
        )
        estimated_gib = comp.total_gib
        # frac is the usable fraction of device memory (gpu_memory_utilization);
        # take the best single-device headroom — multi-GPU tensor-parallel splits are not modelled here
        max_free_gib = max(
            frac * torch.cuda.get_device_properties(dev).total_memory / (1024**3) for dev, frac in vram_map.items()
        )
        if max_free_gib < estimated_gib:
            qualifier = "" if method == "exact" else " (approximate base param count; shape-heuristic fallback)"
            ratio = estimated_gib / max_free_gib if max_free_gib > 0 else float("inf")
            hard_fail = ratio >= _VRAM_HARD_FAIL_RATIO
            report_issue = collector.error if hard_fail else collector.warning
            issue_code = "vram_exceeds_capacity" if hard_fail else "low_vram"
            oom_risk = "Training is expected to OOM" if hard_fail else "Training may OOM"
            if comp.activation_gib is not None:
                report_issue(
                    issue_code,
                    (
                        f"Estimated required VRAM ~{estimated_gib:.1f} GiB total"
                        f" (~{comp.base_weights_gib:.1f} GiB base weights, "
                        f"~{comp.activation_gib:.1f} GiB bf16 compute activations, "
                        f"~{comp.overhead_gib:.1f} GiB reserved){qualifier} "
                        f"exceeds available ~{max_free_gib:.1f} GiB "
                        f"(training.max_vram_fraction={config.training.max_vram_fraction:.2g}). "
                        f"Per-device batch_size={config.training.batch_size}. {oom_risk}. "
                        "This remains an estimate -- attention blocks, adapters, optimizer state, and "
                        "checkpointing materially affect real usage."
                    ),
                )
            else:
                report_issue(
                    issue_code,
                    (
                        f"Estimated required VRAM (~{estimated_gib:.1f} GiB){qualifier} "
                        f"exceeds available ~{max_free_gib:.1f} GiB. "
                        f"{oom_risk}. This is an estimate -- actual usage depends on "
                        "batch size, sequence length, activation checkpointing, and quantization."
                    ),
                )


class InferenceModelCheck(ConfigCheck):
    """Placeholder for inference checks used by a future PII implementation."""

    name = "env.inference"
    label = "Inference configuration"
    category = "environment"

    @override
    def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
        # Legacy PII inference checks were removed with the replacement engine.
        # A later implementation will restore this check.
        return


def _has_hf_token() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def _is_missing_local_path(model_name: str) -> bool:
    path = Path(model_name)
    return path.is_absolute() or model_name.startswith(".")


class HFModelAvailabilityCheck(ConfigCheck):
    """Validate local model, HF cache, and online HF access readiness."""

    name = "env.hf_model_availability"
    label = "HF model availability"
    category = "environment"

    @override
    def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
        model_name = ctx.config.training.pretrained_model
        if not model_name:
            collector.error("model_ref_empty", "`training.pretrained_model` must not be empty.")
            return

        model_ref = ModelRef.parse(model_name)
        if model_ref.local_path is not None and Path(model_name).resolve(strict=False) == model_ref.local_path.resolve(
            strict=False
        ):
            self._check_local_path(model_ref.local_path, collector, model_ref=model_ref)
            return

        if _is_missing_local_path(model_name):
            collector.error(
                "local_model_missing",
                f"`training.pretrained_model` points to missing local path '{model_name}'.",
            )
            return

        if model_ref.repo_id is None:
            collector.error(
                "model_ref_invalid",
                (
                    f"`training.pretrained_model` value '{model_name}' is neither an existing local path "
                    "nor a valid Hugging Face model ID."
                ),
            )
            return

        snapshot_path = model_ref.local_path or model_ref.partial_cached_snapshot()
        if snapshot_path is None:
            self._report_missing_cache(model_ref, collector)
            return

        missing = ModelRef.missing_required_components(snapshot_path)
        if missing:
            message = (
                f"Cached Hugging Face model '{model_ref.repo_id}' at '{snapshot_path}' is missing {', '.join(missing)}."
            )
            if hf_offline_enabled():
                collector.error(
                    "hf_model_cache_incomplete",
                    f"{message} Offline Hugging Face mode is enabled, so model loading will fail.",
                )
                return
            collector.warning(
                "hf_model_cache_incomplete",
                f"{message} Model loading will contact Hugging Face unless the full model snapshot is pre-downloaded.",
            )
            self._report_missing_hf_token(collector)
        self._report_missing_remote_code(model_ref, snapshot_path, collector)

    @staticmethod
    def _check_local_path(model_path: Path, collector: IssueCollector, *, model_ref: ModelRef) -> None:
        if not model_path.is_dir():
            collector.error(
                "local_model_not_directory",
                f"`training.pretrained_model` points to '{model_path}', but local models must be directories.",
            )
            return

        missing = ModelRef.missing_required_components(model_path)
        if missing:
            collector.error(
                "local_model_incomplete",
                f"Local model directory '{model_path}' is missing {', '.join(missing)}.",
            )
        HFModelAvailabilityCheck._report_missing_remote_code(model_ref, model_path, collector)

    @staticmethod
    def _report_missing_cache(model_ref: ModelRef, collector: IssueCollector) -> None:
        message = (
            f"Hugging Face model '{model_ref.repo_id}' is not present in the local cache at '{model_ref.cache_root}'."
        )
        if hf_offline_enabled():
            collector.error(
                "hf_model_not_cached",
                f"{message} Offline Hugging Face mode is enabled, so model loading will fail.",
            )
            return
        collector.warning(
            "hf_model_not_cached",
            f"{message} Model loading will contact Hugging Face unless the model is pre-downloaded.",
        )
        HFModelAvailabilityCheck._report_missing_hf_token(collector)

    @staticmethod
    def _report_missing_remote_code(model_ref: ModelRef, model_path: Path, collector: IssueCollector) -> None:
        if not model_ref.trust_remote_code:
            return

        missing = ModelRef.missing_remote_code_components(model_path)
        if not missing:
            return

        message = (
            f"Trusted Hugging Face model '{model_ref.repo_id}' at '{model_path}' references remote code "
            f"that is not cached locally: {', '.join(missing)}."
        )
        if hf_offline_enabled():
            collector.error(
                "hf_remote_code_not_cached",
                f"{message} Offline Hugging Face mode is enabled, so Transformers cannot fetch it.",
            )
            return
        collector.warning(
            "hf_remote_code_not_cached",
            f"{message} Model loading may contact Hugging Face to fetch it.",
        )
        HFModelAvailabilityCheck._report_missing_hf_token(collector)

    @staticmethod
    def _report_missing_hf_token(collector: IssueCollector) -> None:
        if _has_hf_token():
            return
        collector.warning(
            "hf_token_missing",
            (
                "HF_TOKEN is not set. Model downloads from gated repos will fail. "
                "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN in your environment."
            ),
        )
