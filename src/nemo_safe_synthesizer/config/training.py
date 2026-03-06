# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import (
    Annotated,
    Literal,
)

from pydantic import (
    Field,
)

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
    AutoBoolParam,
    AutoIntParam,
    OptionalAutoInt,
)

__all__ = [
    "TrainingHyperparams",
]

ValueGTZero = ValueValidator(lambda p: range_validator(p, lambda v: v >= 0))


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

    warmup_ratio: Annotated[
        float,
        ValueValidator(value_func=lambda v: v > 0),
        Field(
            title="warmup_ratio",
            description="Ratio of total training steps used for a linear warmup from 0 to the learning rate. Must be > 0.",
        ),
    ] = 0.05

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
        float,
        ValueValidator(value_func=lambda v: 0 < v < 1),
        Field(
            title="learning_rate",
            description="The initial learning rate for the ``AdamW`` optimizer. Must be in (0, 1).",
        ),
    ] = 0.0005

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

    use_unsloth: Annotated[
        AutoBoolParam,
        ValueValidator(value_func=lambda v: v is not None),
        Field(
            title="use_unsloth",
            description="Whether to use Unsloth for optimized training.",
        ),
    ] = AUTO_STR

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
            description="Pretrained model to use for fine tuning. Uses default of SmolLM3.",
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
            description="The number of bits to use for quantization if ``quantize_model`` is ``True``. Accepts 8 or 4.",
        ),
    ] = 8

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
                "Default uses Flash Attention 3 via the HuggingFace Kernels Hub "
                "(requires the 'kernels' pip package; falls back to 'sdpa' if unavailable). "
                "Other common values: 'flash_attention_2' (requires flash-attn pip package), "
                "'sdpa' (PyTorch scaled dot product attention), 'eager' (standard PyTorch). "
                "Custom HuggingFace Kernels Hub paths (e.g. 'kernels-community/flash-attn2') "
                "are also supported."
            ),
        ),
    ] = "kernels-community/vllm-flash-attn3"
