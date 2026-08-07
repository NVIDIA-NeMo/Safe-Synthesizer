# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import math
import warnings
from enum import StrEnum
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
)

from pydantic import Field, model_validator

from ..configurator.parameters import (
    Parameters,
)
from ..configurator.validators import (
    ValueValidator,
    range_validator,
)
from .base import LRScheduler
from .types import (
    AUTO_STR,
    AutoFloatParam,
    AutoIntParam,
    OptionalAutoInt,
)

if TYPE_CHECKING:
    from transformers.utils.quantization_config import QuantizationConfigMixin

__all__ = [
    "QuantizationScheme",
    "TrainingHyperparams",
]


class QuantizationScheme(StrEnum):
    """Quantization schemes supported when ``quantize_model=True``.

    Members are string values so they serialize cleanly through pydantic
    and JSON configs. The enum also owns construction of the corresponding
    transformers ``quantization_config`` object; optional ML dependencies stay
    locally imported in that construction path.

    Selection guide:
    - ``bnb-4bit`` / ``bnb-8bit``: bitsandbytes NF4 / int8. Widest hardware
      support (Ampere+), works with QLoRA and LoftQ. Default for training.
    - ``fp8``: transformers ``FineGrainedFP8Config``. Float8 with block-wise
      scaling. Requires Hopper (sm_90+) or Blackwell. Inference-leaning.
    - ``nvfp4``: NVIDIA FP4 via ``torchao.prototype.mx_formats.NVFP4WeightOnlyConfig``
      wrapped in ``TorchAoConfig``. Requires Blackwell (sm_100+). Weight-only.
    - ``mxfp4``: OCP Microscaling FP4 via transformers ``Mxfp4Config``.
      Hardware support varies by torch/torchao version.
    """

    BNB_4BIT = "bnb-4bit"
    BNB_8BIT = "bnb-8bit"
    FP8 = "fp8"
    NVFP4 = "nvfp4"
    MXFP4 = "mxfp4"

    @property
    def effective_bits(self) -> int:
        """Per-parameter bit width for memory estimation."""
        return {
            QuantizationScheme.BNB_4BIT: 4,
            QuantizationScheme.BNB_8BIT: 8,
            QuantizationScheme.FP8: 8,
            QuantizationScheme.NVFP4: 4,
            QuantizationScheme.MXFP4: 4,
        }[self]

    @property
    def is_bitsandbytes(self) -> bool:
        """Whether the scheme is implemented via bitsandbytes (QLoRA-compatible)."""
        return self in (QuantizationScheme.BNB_4BIT, QuantizationScheme.BNB_8BIT)

    @classmethod
    def from_alias(cls, scheme: QuantizationScheme | str | Literal[4, 8]) -> QuantizationScheme:
        """Normalize string and legacy bit-count aliases to a scheme."""
        if isinstance(scheme, int):
            legacy_aliases = {
                4: cls.BNB_4BIT,
                8: cls.BNB_8BIT,
            }
            try:
                return legacy_aliases[scheme]
            except KeyError as exc:
                raise ValueError(f"Unknown quantization bit-count alias: {scheme!r}. Expected 4 or 8.") from exc
        return cls(scheme)

    def to_transformers_config(self) -> QuantizationConfigMixin:
        """Build the transformers quantization config for this scheme."""
        match self:
            case QuantizationScheme.BNB_4BIT:
                return self._bnb_4bit_config()
            case QuantizationScheme.BNB_8BIT:
                return self._bnb_8bit_config()
            case QuantizationScheme.FP8:
                return self._fp8_config()
            case QuantizationScheme.NVFP4:
                return self._nvfp4_config()
            case QuantizationScheme.MXFP4:
                return self._mxfp4_config()
        raise ValueError(f"Unknown quantization scheme: {self!r}")

    @staticmethod
    def _bnb_4bit_config() -> QuantizationConfigMixin:
        import torch
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    @staticmethod
    def _bnb_8bit_config() -> QuantizationConfigMixin:
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(load_in_8bit=True)

    @staticmethod
    def _fp8_config() -> QuantizationConfigMixin:
        from transformers import FineGrainedFP8Config

        return FineGrainedFP8Config()

    @staticmethod
    def _nvfp4_config() -> QuantizationConfigMixin:
        from transformers import TorchAoConfig

        # Keep this dynamic because the experimental torchao module is runtime-only
        # in some type-checker environments.
        torchao_mx_formats = importlib.import_module("torchao.prototype.mx_formats")
        nvfp4_weight_only_config = torchao_mx_formats.NVFP4WeightOnlyConfig
        return TorchAoConfig(quant_type=nvfp4_weight_only_config())

    @staticmethod
    def _mxfp4_config() -> QuantizationConfigMixin:
        from transformers.utils.quantization_config import Mxfp4Config

        return Mxfp4Config()


ValueGTZero = ValueValidator(lambda p: range_validator(p, lambda v: v >= 0))


def is_valid_warmup(value: float) -> bool:
    """Whether a warmup setting is a usable ratio or step count.

    Mirrors how transformers interprets ``warmup_steps``: values below 1 are a
    ratio of total training steps, values of 1 or more are an absolute step
    count. Fractional values of 1 or more are rejected because transformers
    truncates them (``1.5`` silently becomes ``1``), and non-finite values are
    rejected because they raise ``OverflowError`` once converted to an integer.
    """
    return math.isfinite(value) and value > 0 and (value < 1 or float(value).is_integer())


class TrainingHyperparams(Parameters):
    """Hyperparameters that control the training process behavior.

    This class contains all the fine-tuning hyperparameters that control how the model
    learns, including learning rates, batch sizes, LoRA configuration, and optimization
    settings. These parameters directly affect training performance and quality.
    """

    num_input_records_to_sample: Annotated[
        AutoIntParam,
        ValueGTZero,
        Field(
            title="num_input_records_to_sample",
            description=(
                "Number of records the model will see during training. This parameter is a "
                "proxy for training time. For example, if its value is the same size as the "
                "input dataset, this is like training for a single epoch. If its value "
                "is larger, this is like training for multiple (possibly fractional) epochs. "
                "If its value is smaller, this is like training for a fraction of an epoch. "
                "Supports 'auto' where a reasonable value is chosen based on other config "
                "params and data."
            ),
        ),
    ] = AUTO_STR

    batch_size: Annotated[
        int,
        ValueValidator(value_func=lambda v: v >= 1),
        Field(
            title="batch_size",
            description="The batch size per device for training. Must be >= 1.",
        ),
    ] = 1

    gradient_accumulation_steps: Annotated[
        int,
        ValueValidator(value_func=lambda v: v >= 1),
        Field(
            title="gradient_accumulation_steps",
            description=(
                "Number of update steps to accumulate the gradients for, before "
                "performing a backward/update pass. This technique increases "
                "the effective batch size that will fit into GPU memory. Must be >= 1."
            ),
        ),
    ] = 8

    weight_decay: Annotated[
        float,
        ValueValidator(value_func=lambda v: 0 < v < 1),
        Field(
            title="weight_decay",
            description=(
                "The weight decay to apply to all layers except all bias and "
                "LayerNorm weights in the AdamW optimizer. Must be in (0, 1)."
            ),
        ),
    ] = 0.01

    warmup_steps: Annotated[
        float,
        ValueValidator(value_func=is_valid_warmup),
        Field(
            title="warmup_steps",
            description=(
                "Linear warmup from 0 to the learning rate. "
                "A whole number of 1 or more sets the exact number of warmup steps; "
                "a float in (0, 1) is treated as a ratio of total training steps. "
                "Must be finite and > 0, and cannot be fractional at or above 1."
            ),
        ),
    ] = 0.05

    warmup_ratio: Annotated[
        float | None,
        ValueValidator(value_func=lambda v: v is None or is_valid_warmup(v)),
        Field(
            title="warmup_ratio",
            description="Deprecated. Use warmup_steps instead.",
            exclude=True,
        ),
    ] = None

    @model_validator(mode="after")
    def _migrate_warmup_ratio(self) -> TrainingHyperparams:
        if self.warmup_ratio is not None:
            warnings.warn(
                "warmup_ratio is deprecated and will be removed in a future release. Use warmup_steps instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if "warmup_steps" not in self.model_fields_set:
                self.warmup_steps = self.warmup_ratio
        return self

    lr_scheduler: Annotated[
        str,
        Field(
            title="lr_scheduler",
            description=(
                "The scheduler type to use. See the HuggingFace documentation of ``SchedulerType`` for all possible values."
            ),
        ),
    ] = LRScheduler.COSINE.value

    learning_rate: Annotated[
        AutoFloatParam,
        ValueValidator(lambda p: range_validator(p, lambda v: 0 < v < 1)),
        Field(
            title="learning_rate",
            description=(
                "The initial learning rate for `AdamW` optimizer. Must be in (0, 1). "
                "Setting to 'auto' uses a model-specific default if one exists."
            ),
        ),
    ] = AUTO_STR

    lora_r: Annotated[
        int,
        ValueValidator(value_func=lambda v: v > 0),
        Field(
            title="lora_r",
            description=(
                "The rank of the LoRA update matrices. "
                "Lower rank results in smaller update matrices with fewer trainable parameters. "
                "Must be > 0."
            ),
        ),
    ] = 32

    lora_alpha_over_r: Annotated[
        float,
        ValueValidator(value_func=lambda v: (v >= 0.5) and (v <= 3)),
        Field(
            title="lora_alpha_over_r",
            description=(
                "The ratio of the LoRA scaling factor (alpha) to the LoRA rank. "
                "Empirically, this parameter works well when set to 0.5, 1, or 2. "
                "Must be in [0.5, 3]."
            ),
        ),
    ] = 1.0

    lora_target_modules: Annotated[
        list[str],
        Field(
            title="lora_target_modules",
            description=(
                "The list of transformer modules to apply LoRA to. Possible modules: "
                "'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'."
            ),
        ),
    ] = ["q_proj", "k_proj", "v_proj", "o_proj"]

    rope_scaling_factor: Annotated[
        OptionalAutoInt,
        ValueValidator(lambda p: range_validator(p, lambda v: v >= 1)),
        Field(
            title="rope_scaling_factor",
            description="Scale the base LLM's context length by this factor using RoPE scaling. Must be >= 1 or 'auto'.",
        ),
    ] = AUTO_STR

    validation_ratio: Annotated[
        float,
        ValueValidator(value_func=lambda v: 0 <= v <= 1),
        Field(
            title="validation_ratio",
            description=(
                "The fraction of the training data used for validation. Must be in [0, 1]. "
                "If set to 0, no validation will be performed. "
                "If set larger than 0, validation loss will be computed and reported "
                "throughout training."
            ),
        ),
    ] = 0.0

    validation_steps: Annotated[
        int,
        ValueValidator(value_func=lambda v: v > 0),
        Field(
            title="validation_steps",
            description="The number of steps between validation checks for the HF Trainer arguments. Must be > 0.",
        ),
    ] = 15

    pretrained_model: Annotated[
        str,
        Field(
            title="pretrained_model",
            description=(
                "Pretrained model to use for fine-tuning. Defaults to SmolLM3. "
                "May be a Hugging Face model ID (loaded from the Hugging Face Hub or cache) "
                "or a local path. See security note in docs before using untrusted sources."
            ),
        ),
    ] = "HuggingFaceTB/SmolLM3-3B"

    quantize_model: Annotated[
        bool,
        Field(
            title="quantize_model",
            description=(
                "Whether to quantize the model during training. This can reduce memory usage "
                "and potentially speed up training, but may also impact model accuracy."
            ),
        ),
    ] = False

    quantization_bits: Annotated[
        Literal[4, 8],
        Field(
            title="quantization_bits",
            deprecated=True,
            description=(
                "Deprecated: use ``quantization_scheme`` instead. Bit width for "
                "bitsandbytes quantization when ``quantization_scheme`` is not set "
                "(back-compat alias: 4 → bnb-4bit, 8 → bnb-8bit)."
            ),
        ),
    ] = 8

    quantization_scheme: Annotated[
        QuantizationScheme | None,
        Field(
            title="quantization_scheme",
            description=(
                "Quantization scheme to use when ``quantize_model=True``. Accepts "
                "``bnb-4bit``, ``bnb-8bit``, ``fp8``, ``nvfp4``, or ``mxfp4``. "
                "If unset, falls back to ``quantization_bits`` for backward "
                "compatibility. Non-bitsandbytes schemes are incompatible with "
                "``peft_implementation='loftq'``."
            ),
        ),
    ] = None

    peft_implementation: Annotated[
        str,
        Field(
            title="peft_implementation",
            description=(
                "The PEFT (Parameter-Efficient Fine-Tuning) implementation to use. "
                "Options: 'lora' for Low-Rank Adaptation, 'QLORA' for Quantized LoRA."
            ),
        ),
    ] = "QLORA"

    max_vram_fraction: Annotated[
        float,
        ValueValidator(value_func=lambda v: 0 <= v <= 1),
        Field(
            title="max_vram_fraction",
            description="The fraction of the total VRAM to use for training. Modify this to allow longer sequences. Must be in [0, 1].",
        ),
    ] = 0.80

    attn_implementation: Annotated[
        str,
        Field(
            title="attn_implementation",
            description=(
                "The attention implementation to use for model loading. "
                "Default uses 'sdpa' (PyTorch scaled dot product attention) for broad compatibility. "
                "Other common values: 'flash_attention_2' (requires flash-attn pip package), "
                "'flash_attention_3' (requires flash-attn-3 support), 'eager' (standard PyTorch). "
                "Custom HuggingFace Kernels Hub paths (e.g. 'kernels-community/flash-attn2') are also supported."
            ),
        ),
    ] = "sdpa"

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size = ``batch_size * gradient_accumulation_steps``.

        This is the number of examples that contribute to each optimizer
        update (the "global" batch seen by the loss curve). Canonical
        source for any caller that needs this product -- used by preflight
        checks and logged by the training callbacks.
        """
        return self.batch_size * self.gradient_accumulation_steps
