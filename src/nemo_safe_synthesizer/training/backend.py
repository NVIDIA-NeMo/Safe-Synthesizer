# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training backend abstraction and result types.

Defines ``TrainingBackend``, the abstract base for all LLM fine-tuning
backends, and ``NSSTrainerResult``, the dataclass capturing outputs of a
completed training run.
"""

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
    """Stores all outputs from a training run."""

    training_complete: bool
    """``True`` if training finished normally; ``False`` if stopped early."""

    config: SafeSynthesizerParameters
    """The resolved parameters used for the run."""

    log_history: list[dict]
    """Per-step log entries recorded by the HuggingFace Trainer."""

    adapter_path: Path
    """Filesystem path to the saved LoRA adapter."""

    df_train: pd.DataFrame
    """Training DataFrame (after preprocessing)."""

    df_ml_utility_holdout: pd.DataFrame | None
    """Optional hold-out split for ML-utility evaluation."""

    elapsed_time: float
    """Wall-clock training duration in seconds."""


class TrainingBackend(metaclass=abc.ABCMeta):
    """Abstract base class for LLM fine-tuning backends.

    Subclasses must implement every abstract method.  Two concrete
    implementations are provided:
    [`HuggingFaceBackend`][nemo_safe_synthesizer.training.huggingface_backend.HuggingFaceBackend]
    (standard HuggingFace Trainer with full DP-SGD support) and
    [`UnslothTrainer`][nemo_safe_synthesizer.training.unsloth_backend.UnslothTrainer]
    (Unsloth-optimized training with lower memory usage and faster
    throughput, but no DP support).

    Args:
        params: NSS pipeline configuration.
        model_metadata: Pretrained model metadata (prompt template,
            sequence length, RoPE scaling, etc.).
        training_dataset: Raw tabular HuggingFace ``Dataset`` to fine-tune
            on.  Converted to a DataFrame and run through preprocessing,
            schema inference, and tokenization during
            ``prepare_training_data``.
        data_fraction: Ratio of ``num_input_records_to_sample`` to
            total training records.  Passed to the Opacus DP trainer
            to compute per-step sampling probability for privacy
            accounting.
        logging_level: Python logging verbosity level.
        true_dataset_size: Total number of records (or groups, when
            grouping is enabled) in the training set before
            subsampling.  Used by the Opacus DP trainer alongside
            ``data_fraction`` for privacy accounting.
        eval_dataset: Optional raw tabular ``Dataset`` used only as a
            flag to enable eval-step overrides.  The actual eval split
            passed to the Trainer is produced by the assembler
            (``training_examples.test``).
        generation_eval: If ``True``, attach the
            [`InferenceEvalCallback`][nemo_safe_synthesizer.training.callbacks.InferenceEvalCallback]
            during training.
        callbacks: Extra HuggingFace ``TrainerCallback`` instances.
        action_executor: Executor for user-defined data actions (column
            transforms, type conversions, etc.).  When provided, its
            ``preprocess`` phase runs on the training DataFrame before
            tokenization in ``prepare_training_data``.
        workdir: Working directory for artifacts; required.

    Raises:
        ValueError: If ``workdir`` is not provided.
    """

    model: PreTrainedModel | PeftModel
    """The loaded model, with LoRA/PEFT wrapping applied after ``maybe_quantize``."""

    tokenizer: PreTrainedTokenizer
    """Tokenizer corresponding to the pretrained model."""

    quant_params: dict
    """LoRA and optional quantization configuration populated by ``maybe_quantize``."""

    load_params: dict
    """Raw parameters used when calling ``from_pretrained``."""

    trainer_type: type[OpacusDPTrainer | Trainer | FastLanguageModel]
    """Trainer class to instantiate -- standard ``Trainer``, ``OpacusDPTrainer`` for DP, or ``FastLanguageModel`` for Unsloth."""

    trainer: OpacusDPTrainer | Trainer
    """Instantiated trainer, created during ``prepare_params``."""

    callbacks: list[TrainerCallback]
    """HuggingFace ``TrainerCallback`` instances attached to the trainer."""

    results: NSSTrainerResult
    """Outputs of the completed training run, populated by ``train``."""

    training_examples: TrainingExamples
    """Tokenized and assembled training (and optional validation) splits."""

    df_train: pd.DataFrame
    """Training DataFrame after preprocessing."""

    df_test: pd.DataFrame | None
    """Hold-out DataFrame for ML-utility evaluation, or ``None``."""

    dataset_schema: dict | None
    """JSON schema inferred from the training DataFrame."""

    training_output_dir: Path
    """Directory for trainer checkpoints and cache files."""

    workdir: Workdir
    """Working directory structure for all training artifacts."""

    train_args: TrainingArguments | dict
    """HuggingFace ``TrainingArguments`` (or raw dict before instantiation)."""

    data_fraction: float | None
    """Ratio of sampled records to total, used for DP privacy accounting."""

    elapsed_time: float
    """Wall-clock training duration in seconds."""

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
        """Load, validate, and tokenize the training dataset.

        Runs auto-config resolution, validates groupby/orderby columns,
        applies time-series processing and ``action_executor`` preprocessing,
        then assembles tokenized training examples.  Populates
        ``training_examples``, ``dataset_schema``, ``df_train``, and
        ``data_fraction``.
        """

    @abc.abstractmethod
    def prepare_config(self):
        """Set framework-specific model-loading parameters.

        Builds the ``framework_load_params`` dict consumed by
        ``from_pretrained``, including device map, attention implementation,
        quantization config, and RoPE scaling.  Idempotent -- returns
        immediately if parameters are already prepared.
        """

    @abc.abstractmethod
    def prepare_params(self):
        """Build training arguments and instantiate the trainer.

        Constructs ``TrainingArguments`` from the resolved config, configures
        DP or standard training, selects the data collator, and creates the
        ``trainer`` instance with attached callbacks.
        """

    @abc.abstractmethod
    def maybe_quantize(self, **quant_params: dict):
        """Apply PEFT / quantization wrapping to the loaded model.

        Configures LoRA parameters (rank, alpha, target modules) and wraps
        ``model`` as a PEFT model.  When quantization is enabled, applies
        k-bit preparation before wrapping.
        """

    @abc.abstractmethod
    def load_model(self):
        """Load the pretrained model and tokenizer.

        Calls ``prepare_config`` to resolve loading parameters, loads the
        model and tokenizer via the framework-specific loader (e.g.
        ``AutoModelForCausalLM`` or ``FastLanguageModel``), then applies
        PEFT/quantization via ``maybe_quantize``.
        """

    @abc.abstractmethod
    def train(self):
        """Run the full training loop and populate ``results``.

        Orchestrates the end-to-end pipeline: prepares data, builds
        training arguments, runs the trainer, and saves artifacts.
        Populates ``results`` with the training outcome.
        """

    @abc.abstractmethod
    def save_model(self):
        """Persist the fine-tuned adapter and related artifacts.

        Saves the LoRA adapter weights, model metadata, dataset schema,
        and resolved config to the ``workdir``.  Optionally frees the
        model from GPU memory after saving.
        """

    def _trust_remote_code_for_model(self) -> bool:
        """Determine whether the model should be loaded with ``trust_remote_code=True``.

        Currently returns ``True`` only for NVIDIA models on HuggingFace Hub.

        Returns:
            Whether to trust remote code when loading the model.
        """
        return str(self.params.training.pretrained_model).startswith("nvidia/")
