# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract class for training backend and shared result dataclass."""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from datasets import Dataset
from peft import PeftModel
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from ..cli.artifact_structure import Workdir
from ..config import SafeSynthesizerParameters
from ..data_processing.actions.data_actions import ActionExecutor
from ..data_processing.assembler import TrainingExamples
from ..llm.metadata import ModelMetadata
from ..observability import get_logger
from ..privacy.dp_transformers.dp_utils import (
    OpacusDPTrainer,
)

if TYPE_CHECKING:
    from unsloth import FastLanguageModel  # ty: ignore[unresolved-import]

logger = get_logger()


@dataclass
class NSSTrainerResult:
    """Stores all outputs from a training run.

    Attributes:
        training_complete: ``True`` if training finished normally; ``False``
            if it was stopped early (e.g. by the inference-eval callback).
        config: The resolved parameters used for the run.
        log_history: Per-step log entries recorded by the HuggingFace Trainer.
        adapter_path: Filesystem path to the saved LoRA adapter.
        df_train: Training DataFrame (after preprocessing).
        df_ml_utility_holdout: Optional hold-out split for ML-utility evaluation.
        elapsed_time: Wall-clock training duration in seconds.
    """

    training_complete: bool
    config: SafeSynthesizerParameters
    log_history: list[dict]
    adapter_path: Path
    df_train: pd.DataFrame
    df_ml_utility_holdout: pd.DataFrame | None
    elapsed_time: float


class TrainingBackend(metaclass=abc.ABCMeta):
    """Abstract base class for LLM fine-tuning backends.

    Lifecycle: ``__init__`` -> ``prepare_training_data`` -> ``load_model``
    -> ``prepare_params`` -> ``train`` -> ``save_model``.

    Subclasses must implement every abstract method.  Two concrete
    implementations are provided:
    :class:`~.huggingface_backend.HuggingFaceBackend` (standard
    HuggingFace Trainer with full DP-SGD support) and
    :class:`~.unsloth_backend.UnslothTrainer` (Unsloth-optimized
    training with lower memory usage and faster throughput, but no
    DP support).

    Args:
        params: NSS pipeline configuration.
        model_metadata: Pretrained model metadata (prompt template,
            sequence length, RoPE scaling, etc.).
        training_dataset: HuggingFace ``Dataset`` to fine-tune on.
        data_fraction: Ratio of ``num_input_records_to_sample`` to
            total training records.  Passed to the Opacus DP trainer
            to compute per-step sampling probability for privacy
            accounting.  Computed automatically during
            :meth:`prepare_training_data`; only supply explicitly when
            resuming a partially completed run.
        logging_level: Python logging verbosity level.
        true_dataset_size: Total number of records (or groups, when
            grouping is enabled) in the training set before
            subsampling.  Used by the Opacus DP trainer alongside
            ``data_fraction`` for privacy accounting.
        eval_dataset: Optional evaluation dataset.
        generation_eval: If ``True``, attach the
            :class:`~.callbacks.InferenceEvalCallback` during training.
        callbacks: Extra HuggingFace ``TrainerCallback`` instances.
        action_executor: Optional preprocessing action executor.
        workdir: Working directory for artifacts; required.

    Raises:
        ValueError: If ``workdir`` is not provided.
    """

    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizer
    quant_params: dict
    load_params: dict
    trainer_type: type[OpacusDPTrainer | Trainer | FastLanguageModel]
    trainer: OpacusDPTrainer | Trainer
    callbacks: list[TrainerCallback]
    results: NSSTrainerResult
    training_examples: TrainingExamples
    df_train: pd.DataFrame
    df_test: pd.DataFrame | None
    dataset_schema: dict | None
    training_output_dir: Path
    workdir: Workdir
    train_args: TrainingArguments | dict
    data_fraction: float | None
    elapsed_time: float

    def __init__(
        self,
        params: SafeSynthesizerParameters,
        model_metadata: ModelMetadata,
        training_dataset: Dataset | None = None,
        data_fraction: float | None = None,
        logging_level: int = logging.INFO,
        true_dataset_size: int | None = None,
        eval_dataset: Dataset | None = None,
        generation_eval: bool = False,
        callbacks: list[TrainerCallback] | None = None,
        action_executor: ActionExecutor | None = None,
        workdir: Workdir | None = None,
        *args,
        **kwargs,
    ):
        self.params = params
        self.model_metadata = model_metadata
        self.dataset_schema = None
        self.framework_load_params: dict = {}
        self.data_fraction = data_fraction
        self.logging_level = logging_level
        self.true_dataset_size = true_dataset_size
        self.eval_dataset = eval_dataset
        self.training_dataset = training_dataset
        self.generation_eval = generation_eval
        self.callbacks = callbacks or []
        self.action_executor = action_executor
        if not workdir:
            raise ValueError("workdir is required")
        self.workdir = workdir

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "prepare_training_data")
            and callable(subclass.prepare_training_data)
            and hasattr(subclass, "load_model")
            and callable(subclass.load_model)
            and hasattr(subclass, "prepare_training_args")
            and callable(subclass.prepare_params)
            and hasattr(subclass, "maybe_quantize")
            and callable(subclass.maybe_quantize)
            and hasattr(subclass, "train")
            and callable(subclass.train)
            and hasattr(subclass, "save_model")
            and callable(subclass.save_model)
            or NotImplemented
        )

    @abc.abstractmethod
    def prepare_training_data(self):
        """Load, validate, and tokenize the training dataset."""

    @abc.abstractmethod
    def prepare_config(self):
        """Set framework-specific model-loading parameters."""

    @abc.abstractmethod
    def prepare_params(self):
        """Build training arguments and instantiate the trainer."""

    @abc.abstractmethod
    def maybe_quantize(self, **quant_params: dict):
        """Apply PEFT / quantization wrapping to the loaded model."""

    @abc.abstractmethod
    def load_model(self):
        """Load the pretrained model and tokenizer."""

    @abc.abstractmethod
    def train(self):
        """Run the full training loop and populate :attr:`results`."""

    @abc.abstractmethod
    def save_model(self):
        """Persist the fine-tuned adapter and related artifacts."""

    def _trust_remote_code_for_model(self) -> bool:
        """Determines whether the model should be loaded with
        trusting remote code.

        Currently, this function only returns true when the model being
        loaded from HF Hub is an NVIDIA model.

        Returns:
            whether to load the model with trusting remote code.
        """
        return str(self.params.training.pretrained_model).startswith("nvidia/")
