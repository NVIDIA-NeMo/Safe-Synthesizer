# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-stage checks: GPU, VRAM, tokens, log settings."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

from ...config.replace_pii import has_inference_key
from ...observability import get_logger
from ..base import ConfigCheck
from ..helpers import require_import
from ..types import IssueCollector, PreflightContext

logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...config.training import TrainingHyperparams

__all__ = [
    "CUDAAvailabilityCheck",
    "HFTokenCheck",
    "InferenceKeyCheck",
    "VRAMHeadroomCheck",
    "bytes_per_base_weight",
    "estimate_base_model_params",
    "estimate_params_from_shape",
    "param_count_from_empty_model",
]


class CUDAAvailabilityCheck(ConfigCheck):
    """Validate CUDA GPU availability."""

    name = "gpu.cuda"
    label = "CUDA availability"
    category = "environment"

    def check(self, ctx: PreflightContext, collector: IssueCollector) -> None:
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

        # both unsloth and peft require a CUDA GPU; we don't care about unsloth specific errors
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

    attn = 2 * H * H + 2 * H * kv_heads * head_dim
    mlp = 3 * H * inter
    per_layer = attn + mlp
    embed = V * H
    lm_head = 0 if tied else V * H
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
    r"""Return expected bytes/param for the base model given PEFT mode.

    NSS always trains via LoRA or QLoRA, so the base model's storage
    precision dominates VRAM (LoRA adapter params, gradients, and
    optimizer state are comparatively negligible).

    - QLoRA: \(\text{bits}/8 + 0.1\) to cover quant state (absmax / block
      scales) and dequant workspace. Yields \(\approx 0.6\) for 4-bit,
      \(\approx 1.1\) for 8-bit.
    - LoRA (unquantized): \(2\) bytes (bf16/fp16 base weights).

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
    if training_cfg.peft_implementation.upper() == "QLORA":
        return training_cfg.quantization_bits / 8 + 0.1
    return 2.0


_VRAM_FIXED_OVERHEAD_GIB = 2.0
r"""CUDA kernels, activations (under gradient checkpointing), and runtime fudge.

Intentionally a single constant: activation memory scales like
\(\mathcal{O}(B \cdot S \cdot H \cdot \sqrt{L})\) (Megatron-style, where
\(B\) is batch, \(S\) sequence length, \(H\) hidden size, \(L\) layers),
which adds another axis of estimation error for marginal gain versus the
order-of-magnitude correction this heuristic is already making over the
previous \(6 H L\) formula. Tune if we see systematic false negatives.

References:
    - Korthikanti, V. et al. "Reducing Activation Recomputation in Large
      Transformer Models" (2022) -- activation-memory scaling under
      selective recomputation / gradient checkpointing.
      <https://arxiv.org/abs/2205.05198>
"""


class VRAMHeadroomCheck(ConfigCheck):
    r"""Estimate whether GPU VRAM is sufficient for training.

    The estimate is intentionally a *lower bound*:

    \[
        \text{VRAM}_\text{est} = N \cdot b + C
    \]

    where \(N\) is the base-model parameter count (see
    [estimate_base_model_params][nemo_safe_synthesizer.preflight.checks.environment.estimate_base_model_params];
    exact via the meta-tensor path, or the shape-heuristic fallback),
    \(b\) is the bytes-per-param for the selected PEFT mode (see
    [bytes_per_base_weight][nemo_safe_synthesizer.preflight.checks.environment.bytes_per_base_weight]),
    and \(C\) is a fixed overhead for CUDA kernels and checkpointed
    activations. The expression excludes the fine-grained activation
    term \(\mathcal{O}(B \cdot S \cdot H \cdot L)\), LoRA adapter
    parameters, gradients, and optimizer state. Those are typically
    small compared to the base weights for parameter-efficient
    fine-tuning, but not zero. Passing this check does not guarantee
    training will fit in VRAM; failing it is a strong signal that it
    will OOM.

    References:
        - EleutherAI, "Transformer Math 101" -- grounds the rule of thumb
          that inference adds ~20% over raw weights; training adds
          considerably more. <https://blog.eleuther.ai/transformer-math/>
    """

    name = "gpu.vram"
    label = "VRAM headroom"
    category = "environment"
    requires = ("gpu.cuda",)

    def check(self, ctx: PreflightContext, collector: IssueCollector) -> None:
        import torch

        from ...llm.utils import get_max_vram

        config = ctx.config
        vram_map = get_max_vram()
        autoconfig = getattr(config, "_metadata_autoconfig", None) or getattr(config, "autoconfig", None)
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

        bytes_per_param = bytes_per_base_weight(config.training)
        estimated_gib = (n_params * bytes_per_param) / (1024**3) + _VRAM_FIXED_OVERHEAD_GIB
        max_free_gib = max(
            frac * torch.cuda.get_device_properties(dev).total_memory / (1024**3) for dev, frac in vram_map.items()
        )
        if max_free_gib < estimated_gib:
            qualifier = "" if method == "exact" else " (approximate; shape-heuristic fallback)"
            collector.warning(
                "low_vram",
                (
                    f"Estimated required VRAM (~{estimated_gib:.1f} GiB){qualifier} "
                    f"exceeds available ~{max_free_gib:.1f} GiB. "
                    "Training may OOM. This is an estimate -- actual usage depends on "
                    "batch size, sequence length, and activation checkpointing."
                ),
            )


class InferenceKeyCheck(ConfigCheck):
    """Check NSS_INFERENCE_KEY environment variable."""

    name = "env.inference_key"
    label = "Inference key"
    category = "environment"

    def check(self, ctx: PreflightContext, collector: IssueCollector) -> None:
        config = ctx.config
        if config.replace_pii is not None and config.replace_pii.globals.classify.enable_classify is not False:
            if not has_inference_key():
                collector.warning(
                    "inference_key_missing",
                    "NSS_INFERENCE_KEY is not set. PII column classification will run in degraded mode.",
                )


class HFTokenCheck(ConfigCheck):
    """Check HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variable."""

    name = "env.hf_token"
    label = "HF token"
    category = "environment"

    def check(self, ctx: PreflightContext, collector: IssueCollector) -> None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            collector.warning(
                "hf_token_missing",
                (
                    "HF_TOKEN is not set. Model downloads from gated repos will fail. "
                    "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN in your environment."
                ),
            )
