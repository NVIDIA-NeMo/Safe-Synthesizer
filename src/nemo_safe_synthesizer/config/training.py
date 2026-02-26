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
    AutoFloatParam,
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

    Attributes:
        num_input_records_to_sample: Number of records the model will see during training.
            This parameter is a proxy for training time. For example, if its value is the same
            size as the input dataset, this is like training for a single epoch. If its value
            is larger, this is like training for multiple (possibly fractional) epochs. If its
            value is smaller, this is like training for a fraction of an epoch. Supports 'auto'
            where a reasonable value is chosen based on other config params and data.
        batch_size: The batch size per device for training.
        gradient_accumulation_steps: Number of update steps to accumulate the gradients for,
            before performing a backward/update pass. This technique increases the effective
            batch size that will fit into GPU memory.
        weight_decay: The weight decay to apply (if not zero) to all layers except all
            bias and LayerNorm weights in the AdamW optimizer.
        warmup_ratio: Ratio of total training steps used for a linear warmup from 0
            to the learning rate.
        lr_scheduler: The scheduler type to use. See the HuggingFace documentation of
            `SchedulerType` for all possible values.
        learning_rate: The initial learning rate for `AdamW` optimizer.
        lora_r: The rank of the LoRA update matrices, expressed in int. Lower
            rank results in smaller update matrices with fewer trainable parameters.
        lora_alpha_over_r: The ratio of the LoRA scaling factor (alpha) to
            the LoRA rank. Empirically, this parameter works well when set to 0.5, 1, or 2.
        lora_target_modules: The list of transformer modules to apply LoRA to.
            Possible modules: 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
        use_unsloth: Whether to use unsloth.
        rope_scaling_factor: Scale the base LLM's context length by this factor using RoPE scaling.
        validation_ratio: The fraction of the training data that will be used for validation.
            The range should be 0 to 1. If set to 0, no validation will be performed.
            If set larger than 0, validation loss will be computed and reported throughout training.
        validation_steps: The number of steps between validation checks for the HF Trainer arguments.
        pretrained_model: Pretrained model to use for fine-tuning. Uses default of TinyLlama.
        quantize_model: Whether to quantize the model during training. This can reduce memory usage
            and potentially speed up training, but may also impact model accuracy.
        quantization_bits: The number of bits to use for quantization if `quantize_model` is True.
            Common values are 8 or 4 bits.
        peft_implementation: The PEFT (Parameter-Efficient Fine-Tuning) implementation to use.
            Options include 'lora' for Low-Rank Adaptation, QLoRA for Quantized LoRA. Each method has its own trade-offs in terms of performance
            and resource requirements.
        attn_implementation: The attention implementation to use for model loading.
            Default uses Flash Attention 3 via the HuggingFace Kernels Hub. Falls back to 'sdpa'
            if the kernels package is not installed. Other common values include 'flash_attention_2',
            'sdpa', and 'eager'.
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
            description="The batch size per device for training",
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
                "the effective batch size that will fit into GPU memory."
            ),
        ),
    ] = 8

    weight_decay: Annotated[
        float,
        ValueValidator(value_func=lambda v: 0 < v < 1),
        Field(
            title="weight_decay",
            description=(
                "The weight decay to apply (if not zero) to all layers except all bias and "
                "LayerNorm weights in the AdamW optimizer."
            ),
        ),
    ] = 0.01

    warmup_ratio: Annotated[
        float,
        ValueValidator(value_func=lambda v: v > 0),
        Field(
            title="warmup_ratio",
            description="Ratio of total training steps used for a linear warmup from 0 to the learning rate.",
        ),
    ] = 0.05

    lr_scheduler: Annotated[
        str,
        Field(
            title="lr_scheduler",
            description=(
                "The scheduler type to use. See the HuggingFace documentation of `SchedulerType` for all possible values."
            ),
        ),
    ] = LRScheduler.COSINE.value

    learning_rate: Annotated[
        AutoFloatParam,
        ValueValidator(value_func=lambda v: True if v == AUTO_STR else (0 < v < 1)),
        Field(
            title="learning_rate",
            description=(
                "The initial learning rate for `AdamW` optimizer. "
                "Setting to 'auto' uses learning rate of 0.0005 for all models except for Mistral that is set to 0.0001"
            ),
        ),
    ] = AUTO_STR

    lora_r: Annotated[
        int,
        ValueValidator(value_func=lambda v: v > 0),
        Field(
            title="lora_r",
            description=(
                "The rank of the LoRA update matrices, expressed in int. "
                "Lower rank results in smaller update matrices with fewer trainable parameters."
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
                "Empirically, this parameter works well when set to 0.5, 1, or 2."
            ),
        ),
    ] = 1.0

    lora_target_modules: Annotated[
        list[str],
        Field(
            title="lora_target_modules",
            description=(
                "The list of transformer modules to apply LoRA to. Possible modules: "
                "'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'"
            ),
        ),
    ] = ["q_proj", "k_proj", "v_proj", "o_proj"]

    use_unsloth: Annotated[
        AutoBoolParam,
        ValueValidator(value_func=lambda v: v is not None),
        Field(
            title="use_unsloth",
            description="Whether to use unsloth.",
        ),
    ] = AUTO_STR

    rope_scaling_factor: Annotated[
        OptionalAutoInt,
        ValueValidator(lambda p: range_validator(p, lambda v: v >= 1)),
        Field(
            title="rope_scaling_factor",
            description="Scale the base LLM's context length by this factor using RoPE scaling.",
        ),
    ] = AUTO_STR

    validation_ratio: Annotated[
        float,
        ValueValidator(value_func=lambda v: 0 <= v <= 1),
        Field(
            title="validation_ratio",
            description=(
                "The fraction of the training data that will be used for validation."
                "The range should be 0 to 1. If set to 0, no validation will be performed."
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
            description="The number of steps between validation checks for the HF Trainer arguments.",
        ),
    ] = 15

    pretrained_model: Annotated[
        str,
        Field(
            title="pretrained_model",
            description="Pretrained model to use for fine tuning. Uses default of TinyLlama.",
        ),
    ] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

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
            description=(
                "The number of bits to use for quantization if `quantize_model` is True. Common values are 8 or 4 bits."
            ),
        ),
    ] = 8

    peft_implementation: Annotated[
        str,
        Field(
            title="peft_implementation",
            description=(
                "The PEFT (Parameter-Efficient Fine-Tuning) implementation to use. "
                "Options include 'lora' for Low-Rank Adaptation or QLoRA for Quantized LoRA. "
                "Each method has its own trade-offs in terms of performance and resource requirements."
            ),
        ),
    ] = "QLORA"

    max_vram_fraction: Annotated[
        float,
        ValueValidator(value_func=lambda v: 0 <= v <= 1),
        Field(
            title="max_vram_fraction",
            description="The fraction of the total VRAM to use for training. Default is 0.9. Modify this to allow longer sequences to be used.",
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
