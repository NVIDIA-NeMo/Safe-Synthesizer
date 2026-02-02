# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Literal, Tuple, Union

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from peft import (
    PeftModel,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)

if TYPE_CHECKING:
    from unsloth import FastLanguageModel  # noqa: F401  # ty: ignore[unresolved-import]

from ..observability import get_logger

logger = get_logger(__name__)


def trust_remote_code_for_model(model_name: str | Path) -> bool:
    """Determines whether the model should be loaded with
    trusting remote code.

    Currently, this function only returns true when the model being
    loaded from HF Hub is a gretelai/nvidia model.

    Returns:
        whether to load the model with trusting remote code.
    """
    mn = str(model_name)
    return mn.startswith("nvidia/") or mn.startswith("gretel/")


def cleanup_memory() -> None:
    """Run garbage collection and empty the CUDA cache."""
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()


def gpu_stats():
    """
    Get the GPU stats.
    """

    def round_gb(value: float) -> float:
        return round(value / 1024 / 1024 / 1024, 3)

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round_gb(torch.cuda.max_memory_reserved())
    max_memory = round_gb(gpu_stats.total_memory)
    logger.info(f"{start_gpu_memory} GB of memory reserved.")
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")


def get_max_vram(
    memory_fraction: float | None = None, as_string: bool = True, as_fraction: bool = False
) -> dict[int, float | str]:
    """
    Calculate max memory allocation for each available GPU and CPU.

    Args:
        memory_fraction: Fraction of total GPU memory to allocate (default 0.8 for 80%)

    Returns:
        Dictionary mapping device IDs to memory limits
    """
    if memory_fraction is None:
        memory_fraction = 0.8
    max_memory = {}

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            free, total = torch.cuda.mem_get_info(device=i)
            safe_free = free - (2 * 1024**3)
            gpu_memory_utilization = min(memory_fraction, safe_free / total)
            memory_gib = gpu_memory_utilization * total / (1024**3)
            if as_fraction:
                max_memory[i] = gpu_memory_utilization
            else:
                max_memory[i] = memory_gib if not as_string else f"{memory_gib:.2f}GiB"
            logger.info(
                f"GPU {i}: Will allocate {memory_gib:.2f}GiB ({memory_fraction * 100}% of {total / (1024**3):.2f}GiB)"
            )

    return max_memory


def add_bos_eos_tokens_to_tokenizer(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """
    Configure the tokenizer with bos, eos tokens for sample splitting.
    """
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def get_param_from_config(
    param: str,
    default_value: Any | None = None,
    model_name: str | None = None,
    trust_remote_code: bool | None = None,
    config: AutoConfig | None = None,
) -> str | None:
    """
    Get a parameter from the model's AutoConfig.
    """
    if config is None:
        if model_name is None:
            raise ValueError("model_name is required if config is not provided")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    return getattr(config, param, default_value)


def _get_auto_tokenizer(
    model_name: Path | str,
    max_position_embeddings: int,
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_position_embeddings,
    )
    tokenizer = add_bos_eos_tokens_to_tokenizer(tokenizer)
    return tokenizer


def get_device_map(
    model_target: str,
    autoconfig: AutoConfig | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    force_single_device: int | None = None,
):
    """
    Fetch the device map for a given model and optionally override.

    Args:
        model_target: The name or path of the pre-trained model.
        revision: The specific model version to use.
        trust_remote_code: Whether to trust remote code when loading the model.
        local_files_only: Whether to only use local files when loading the model.
        force_single_device: If provided, all layers will be set to this device.

    Returns:
        OrderedDict: An ordered dictionary representing the device map,
            where keys are layers and values are device IDs.
    """

    config = autoconfig or AutoConfig.from_pretrained(
        model_target,
        revision=revision,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    # Create an empty model with the configuration
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    device_map = infer_auto_device_map(model)
    if force_single_device is not None:
        for key in device_map:
            device_map[key] = force_single_device
    return device_map


def count_trainable_params(model: PeftModel) -> Tuple[int, int]:
    """Determines the number of trainable and overall params of a model.

    Returns:
        int, int - the number of trainable params, and all parameters,
            respectively.
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return trainable_params, all_params


@contextmanager
def optimize_for_inference(
    model: Union["FastLanguageModel", "AutoModelForCausalLM"],
) -> Generator[None, Any, Any]:
    """
    Apply unsloth inference-time optimizations within a context manager,
    and revert to train-time settings on exiting the with block.
    Usage: "with optimize_for_inference(model):..."
    """
    if torch.cuda.is_available() and type(model).__name__ == "FastLanguageModel":
        from unsloth import FastLanguageModel  # noqa: F401  # ty: ignore[unresolved-import]

        FastLanguageModel.for_inference(model)
        yield
        FastLanguageModel.for_training(model)
    yield


def get_quantization_config(quantization_bits: Literal[4, 8]) -> BitsAndBytesConfig:
    """
    Get the quantization config for a given number of bits. Model independent.

    Args:
        quantization_bits: The number of bits to use for quantization.

    Returns:
        The quantization config.
    """
    if quantization_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization_bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Invalid quantization bits: {quantization_bits}")
