# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path

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

logger = get_logger()


@dataclass
class NSSTrainerResult:
    training_complete: bool
    config: SafeSynthesizerParameters
    log_history: list[dict]
    adapter_path: Path
    df_train: pd.DataFrame
    df_ml_utility_holdout: pd.DataFrame | None
    elapsed_time: float


class TrainingBackend(metaclass=abc.ABCMeta):
    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizer
    quant_params: dict
    load_params: dict
    trainer_type: type[Trainer | OpacusDPTrainer] | partial[OpacusDPTrainer]
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
        pass

    @abc.abstractmethod
    def prepare_config(self):
        pass

    @abc.abstractmethod
    def prepare_params(self):
        pass

    @abc.abstractmethod
    def maybe_quantize(self, **quant_params: dict):
        pass

    @abc.abstractmethod
    def load_model(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def save_model(self):
        pass

    def delete_trainable_model(self) -> None:
        """Delete the trainable model and clean up GPU memory. Override in subclasses."""

    def _trust_remote_code_for_model(self) -> bool:
        """Determines whether the model should be loaded with
        trusting remote code.

        Currently, this function only returns true when the model being
        loaded from HF Hub is an NVIDIA model.

        Returns:
            whether to load the model with trusting remote code.
        """
        return str(self.params.training.pretrained_model).startswith("nvidia/")
