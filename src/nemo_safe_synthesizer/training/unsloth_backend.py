# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimized training backend using Unsloth."""

import os

import torch

from ..llm.utils import add_bos_eos_tokens_to_tokenizer
from ..observability import get_logger
from ..training.huggingface_backend import HuggingFaceBackend

logger = get_logger(__name__)


class UnslothTrainer(HuggingFaceBackend):
    """Training backend using Unsloth for optimized LLM fine-tuning.

    Extends [`HuggingFaceBackend`][nemo_safe_synthesizer.training.huggingface_backend.HuggingFaceBackend]
    to leverage Unsloth's optimized
    training routines, providing faster training speeds and reduced memory
    usage compared to standard HuggingFace implementations.

    In addition to the arguments accepted by the parent class, ``**kwargs``
    may include:

    * ``rope_scaling`` -- RoPE scaling configuration from model metadata.
    * ``torch_dtype`` -- Data type for model weights.
    * ``quantization_config`` -- Configuration for model quantization.

    See Also:
        HuggingFaceBackend: Parent class providing base training functionality.

    Raises:
        RuntimeError: If CUDA is not available.
    """

    def __init__(self, *args, **kwargs):
        from unsloth import FastLanguageModel  # ty: ignore[unresolved-import]

        super().__init__(*args, **kwargs)
        self.model_loader_type = FastLanguageModel

        if not torch.cuda.is_available():
            raise RuntimeError("Cannot use unsloth without GPU.")
        self.prepare_config(**kwargs)
        self._update_for_unsloth(**kwargs)

    def _update_for_unsloth(self, **model_args):
        """Translate HuggingFace-style framework params to Unsloth conventions."""
        # Unsloth uses max_seq_length instead of max_position_embeddings and passes it internally
        # to AutoModelForCausalLM.from_pretrained(), so we must remove it to avoid duplicate kwargs
        self.framework_load_params.pop("max_position_embeddings", None)
        # Use model_metadata.max_seq_length which already includes rope_scaling.factor
        self.framework_load_params["max_seq_length"] = self.model_metadata.max_seq_length
        self.framework_load_params["dtype"] = model_args.pop("dtype", None) or model_args.pop("torch_dtype", None)
        # Unsloth uses `model_name` instead of `pretrained_model_name_or_path`
        if "pretrained_model_name_or_path" in self.framework_load_params:
            self.framework_load_params["model_name"] = self.framework_load_params.pop("pretrained_model_name_or_path")

        if "local_files_only" not in (self.framework_load_params | model_args):
            lfo = os.environ.get("LOCAL_FILES_ONLY", None)
            if lfo is None:
                logger.info("LOCAL_FILES_ONLY: not set, using default of False")
                lfo = False
            else:
                logger.info(f"LOCAL_FILES_ONLY: set to {lfo}")
                lfo = lfo in [
                    "true",
                    "True",
                    1,
                    "1",
                ]
            self.framework_load_params["local_files_only"] = lfo

        if (model_args.get("quantization_config") or self.__dict__.get("quantization_config")) is not None:
            if qb := self.params.get("quantization_bits"):
                bits_d = {4: "load_in_4bit", 8: "load_in_8bit"}
                bits_key = bits_d.get(self.params.training.quantization_bits, None)
                if not bits_key:
                    raise ValueError(f"Invalid quantization bits: {qb}")
                self.framework_load_params[bits_key] = True

            self.framework_load_params["bias"] = "none"
            self.framework_load_params["use_gradient_checkpointing"] = "unsloth"

        else:
            self.framework_load_params["load_in_4bit"] = False
            self.framework_load_params["load_in_8bit"] = False

    def maybe_quantize(self):
        """Apply PEFT wrapping via Unsloth's ``FastLanguageModel.get_peft_model``.

        This method configures and applies Parameter-Efficient Fine-Tuning (PEFT)
        using Unsloth's optimized implementation. The PEFT wrapping is always
        applied to ensure the adapter is saved correctly.

        Note:
            Unlike the parent class implementation, this method uses Unsloth's
            ``FastLanguageModel.get_peft_model``.
        """
        from unsloth import FastLanguageModel  # ty: ignore[unresolved-import]

        self._prepare_quantize_base()
        qparams = self.quant_params.copy()
        # unsloth infers the task type from the model, so we need to remove it from the quant params
        qparams.pop("task_type", None)
        # Always wrap the model as a PEFT model to ensure adapter is saved correctly
        self.model = FastLanguageModel.get_peft_model(self.model, **qparams)

    def _load_pretrained_model(self, **model_args):
        """Load model and tokenizer via Unsloth and add BOS/EOS tokens."""
        model, tokenizer = self.model_loader_type.from_pretrained(**self.framework_load_params)

        self.tokenizer = add_bos_eos_tokens_to_tokenizer(
            tokenizer,
        )
        self.model = model

    def load_model(self, **model_args):
        """Load a pretrained model using Unsloth's ``FastLanguageModel``.

        Applies a workaround that disables Unsloth's LLAMA32 support
        check to prevent unnecessary HuggingFace Hub requests, then
        calls :meth:`prepare_config`, :meth:`_load_pretrained_model`,
        and :meth:`maybe_quantize` in sequence.

        Args:
            **model_args: Additional keyword arguments for model configuration.

        Note:
            This method applies a workaround that disables Unsloth's LLAMA32
            support check to prevent unnecessary HuggingFace Hub requests.
            See: https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py#L235
        """
        # NOTE: this hack stops unsloth from reaching out to huggingface, see
        # https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py#L235
        from unsloth.models import loader  # ty: ignore[unresolved-import]

        loader.SUPPORTS_LLAMA32 = False
        logger.info(f"load_model: Loading model {self.params.training.pretrained_model} with args: {model_args}")

        self.prepare_config(**model_args)
        self._load_pretrained_model(**model_args)

        self.maybe_quantize(**model_args)
