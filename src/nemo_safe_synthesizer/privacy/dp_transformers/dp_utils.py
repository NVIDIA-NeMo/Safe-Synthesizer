# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0 AND MIT

# This file has been adapted from the `dp-transformers` library.
# Original source: https://github.com/microsoft/dp-transformers/blob/main/src/dp_transformers/dp_utils.py
# See THIRD_PARTY.md for the original MIT license terms.

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Union

import opacus
import pandas as pd
import torch
from accelerate.optimizer import AcceleratedOptimizer
from datasets import Dataset
from peft import PeftModel
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    modeling_utils,
    training_args,
    utils,
)
from transformers.trainer import TRAINING_ARGS_NAME

from . import linear  # imported for side effects  # noqa
from .privacy_args import (
    PrivacyArguments,
    SafeSynthesizerAccountant,
)
from .sampler import (
    PoissonEntitySampler,
    ShuffledEntitySampler,
)

if utils.is_safetensors_available():
    import safetensors.torch
from ...observability import get_logger

logger = get_logger(__name__)


class DPCallback(TrainerCallback):
    """
    This class registers all the necessary callbacks to make
    transformers.Trainer compatible with Opacus for differentially private
    learning.
    """

    def __init__(
        self,
        noise_multiplier: float,
        sampling_probability: float,
        accountant: SafeSynthesizerAccountant,
        max_epsilon: float = float("inf"),
    ) -> None:
        self.accountant = accountant
        self._max_epsilon = max_epsilon
        self._on_substep_end_was_called = False

        self.noise_multiplier = noise_multiplier
        self.sampling_probability = sampling_probability

    def on_substep_end(
        self,
        args: training_args.TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        optimizer=None,
        **kwargs,
    ):
        if optimizer is None:
            raise RuntimeError("Impossible to access optimizer from inside callback")
        if isinstance(optimizer, AcceleratedOptimizer):
            dp_optimizer = optimizer.optimizer
        else:
            dp_optimizer = optimizer
        dp_optimizer.signal_skip_step(do_skip=True)
        dp_optimizer.step()
        dp_optimizer.zero_grad()

        self._on_substep_end_was_called = True

    def on_step_end(
        self,
        args: training_args.TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        optimizer=None,
        **kwargs,
    ):
        if args.gradient_accumulation_steps > 1 and not self._on_substep_end_was_called:
            raise RuntimeError(
                "Gradient accumulation was specified but `on_substep_end` wasn't called. "
                "Make sure you're using a recent version of transformers (>=4.10.0) "
                "which has an appropriate callback in the trainer."
            )
        if optimizer is None:
            raise RuntimeError(
                "No optimizer provided to on_step_end callback, required for correct DP-SGD to call zero_grad()"
            )

        optimizer.zero_grad()  # Opacus needs .zero_grad() on the optimizer, HF doesn't call by default.
        if not self.accountant.use_prv:
            # Use RDPAccountant, which uses `.step()` to increment number of
            # steps, required for accurate epsilon calculation.
            self.accountant.accountant.step(
                noise_multiplier=self.noise_multiplier,
                sample_rate=self.sampling_probability,
            )

    def on_save(
        self,
        args: training_args.TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        return self._check_max_epsilon_exceeded(state, control)

    def on_evaluate(
        self,
        args: training_args.TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        return self._check_max_epsilon_exceeded(state, control)

    def _check_max_epsilon_exceeded(self, state: TrainerState, control: TrainerControl) -> TrainerControl:
        eps = self.accountant.compute_epsilon(steps=state.global_step + 1)
        if eps > self._max_epsilon:
            logger.info("Max epsilon will be exceeded if trained for one more step. Stopping training.")
            control.should_training_stop = True
        return control


class DataCollatorForPrivateCausalLanguageModeling(DataCollatorForLanguageModeling):
    """
    Trainer automatically uses incrementing integers for position_ids when not
    provided. This is implemented for model families in the transformers library
    and the process occurs during the `forward` method of the model. See
    https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/models/mistral/modeling_mistral.py#L882
    for an example. Opacus is unable to access this. This creates a problem with
    per-sample gradient accumulation. Instead we create `position_ids` during
    the data collation step so they are accessible to Opacus.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer=tokenizer, mlm=False)

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch = super().__call__(examples)

        if "position_ids" not in batch:
            input_ids = batch["input_ids"]
            batch["position_ids"] = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device).repeat(
                input_ids.shape[0], 1
            )
        return batch


class DataCollatorForPrivateTokenClassification(DataCollatorForTokenClassification):
    """Cf DataCollatorForPrivateCausalLanguageModelling"""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer=tokenizer)

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.TensorType]]]
    ) -> Dict[str, torch.Tensor]:
        batch = super().__call__(examples)

        if "position_ids" not in batch:
            input_ids = batch["input_ids"]
            batch["position_ids"] = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device).repeat(
                input_ids.shape[0], 1
            )
        return batch


class GradSampleModule(opacus.GradSampleModule):
    """
    Little wrapper to provide `no_sync` context which is assumed by Huggingface trainer.
    We don't need to do anything in addition here.
    """

    @contextmanager
    def no_sync(self):
        yield


def create_entity_mapping(entity_column_values: list) -> Sequence[Sequence[int]]:
    """
    Creates a mapping from entities to samples in a dataset.
    """
    entities = pd.DataFrame(data={"entity": entity_column_values})
    # Using `groupby("entity")` - note that the entities returned by groupby are
    # sorted, but the order of records in each group is preserved.
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
    # TODO: improve for use in sampler.py using a dictionary or such structure
    # with clearly defined entity_ids
    entity_mapping = [g.index.values for _, g in entities.groupby("entity")]
    return entity_mapping


class OpacusDPTrainer(Trainer):
    """
    Wrapper to modify Huggingface Trainer for use with PEFT fine tuning.
        - remove "loss = loss / self.args.gradient_accumulation_steps" operation
          in training_step as this is already handled by Opacus package.
        - enable entity-level DP training by modifing the sampler and the
           dataloader. In the case of sample-level DP, each sample can be
           represented by a unique entity.
        - wrap the optimizer with Opacus DPOptimizer
        - save only the LoRA adapters (PEFT)
    """

    def __init__(
        self,
        train_dataset: Dataset,
        model: Union[modeling_utils.PreTrainedModel, torch.nn.Module],
        args=None,
        privacy_args: PrivacyArguments | None = None,
        data_fraction: float | None = None,
        true_dataset_size: int | None = None,
        entity_column_values: list | None = None,
        callbacks: List[TrainerCallback] | None = None,
        secure_mode: bool | None = True,
        **kwargs: Dict,
    ) -> None:
        self.train_args = args
        self.privacy_args = privacy_args
        self.secure_mode = secure_mode

        if entity_column_values is None:
            # Record-level DP == mapping each sample to a unique entity.
            self.entity_mapping = [[i] for i in range(train_dataset.num_rows)]
        else:
            self.entity_mapping = create_entity_mapping(entity_column_values=entity_column_values)

        # Adjustments for NavFT
        self.true_num_epochs = self.train_args.num_train_epochs
        self.true_dataset_size = len(self.entity_mapping)

        if data_fraction is not None:
            self.true_num_epochs *= data_fraction
            logger.info(
                f"True number of epochs set to {self.true_num_epochs}",
            )
        if true_dataset_size is not None:
            self.true_dataset_size = true_dataset_size
            logger.info(
                (
                    f"Training dataset contains {self.true_dataset_size} unique "
                    f"{'groups' if entity_column_values else 'records'}; using this "
                    "value for differential privacy parameter determination."
                ),
            )

        if not self.privacy_args.is_initialized:
            self.privacy_args.initialize(
                sampling_probability=self.sampling_probability,
                num_steps=self.num_steps,
            )

        model = GradSampleModule(model)

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            callbacks=callbacks,
            **kwargs,
        )
        self.accountant = SafeSynthesizerAccountant(
            use_prv=self.privacy_args.use_prv,
            noise_multiplier=self.privacy_args.noise_multiplier,
            sampling_probability=self.sampling_probability,
            delta=self.privacy_args.target_delta,
            num_steps=self.num_steps,
        )
        self.dp_callback = DPCallback(
            noise_multiplier=self.privacy_args.noise_multiplier,
            sampling_probability=self.sampling_probability,
            accountant=self.accountant,
            max_epsilon=self.privacy_args.target_epsilon,
        )
        self.add_callback(self.dp_callback)

    def get_epsilon(self):
        """
        Calculate the epsilon after model training completes.
        """
        return self.accountant.compute_epsilon(self.state.global_step)

    @property
    def sampling_probability(self) -> float:
        """
        Calculate the probability of sampling an entity in a batch.

        This is trivial when using record-level DP, i.e. each record is an
        entity. This simply returns the total number of samples seen before
        a gradient update.

        When using entity-level DP, the probability of sampling an entity is
        likely higher (assuming multiple records correspond to an entity).
        If there is only one entity, then the sampling probability would be
        > 1 (i.e. we are sampling the same entity multiple times in a
        batch).

        `sampling_probability` is required to calculate privacy budget
        utilization during training.
        """
        return min(
            1.0,
            self.train_args.per_device_train_batch_size
            * self.train_args.gradient_accumulation_steps
            / self.true_dataset_size,
        )

    @property
    def num_steps(self) -> int:
        """
        Calculate the number of steps required to train the model. Either this
        is user supplied, or determined from num_train_epochs.

        When user specifies num_train_epochs, we determine num_steps based on
        sampling probability. This is we pass over each entity roughly once per
        epoch, similarly to how one expects to pass over each record once per
        epoch in regular record-level training.

        `num_steps` will always be >= 1. This is because we add 1 to
        1/sampling_probability. This can happen when there are fewer entities
        than batch_size * gradient_accumulation_steps (typically 4 * 8 = 32).

        This is used to determine the privacy budget utilization during
        training.
        """
        if self.true_num_epochs == -1:
            return self.train_args.max_steps
        else:
            _num_steps = int(self.true_num_epochs * (1 / self.sampling_probability + 1))
            if _num_steps == self.true_num_epochs:
                logger.warning(
                    "Number of entities in dataset is low. Consider lowering batch size or adding more entities to the dataset for better privacy budget utilization.",
                )
            return _num_steps

    def create_optimizer(self):
        _ = super().create_optimizer()

        class DPOptimizer(opacus.optimizers.DPOptimizer):  # ty: ignore[unresolved-attribute]
            """HF's AcceleratedOptimizer replaces the original reference to
            `original_optimizer.param_groups`, and the approach used by Opacus
            fails when we try to e.g., update the learning rate. We here use
            the same approach used in accelerate, which makes `param_groups` a
            proper 'pointer'."""

            @property
            def param_groups(self):
                return self.original_optimizer.param_groups

            @param_groups.setter
            def param_groups(self, param_groups):
                self.original_optimizer.param_groups = param_groups

        optimizer_generator = DPOptimizer

        # TODO: explore better mitigation for precision based attacks on finite
        # precision devices
        # https://tpdp.journalprivacyconfidentiality.org/2022/papers/HaneyDHSH22.pdf
        self.optimizer = optimizer_generator(
            optimizer=self.optimizer,
            noise_multiplier=self.privacy_args.noise_multiplier,
            max_grad_norm=self.privacy_args.per_sample_max_grad_norm,
            expected_batch_size=self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps,
            secure_mode=self.secure_mode,
        )

        return self.optimizer

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs. Overriden to inject custom
        behavior to scale loss for Opacus.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        # Compared to the original HF implementation (as of 4.48), we use
        # `num_items_in_batch=None` to avoid any extra scaling, since Opacus
        # already does it; we only divide the loss by the number of gradient
        # accumulation steps after loss.backward(), to get correct logging
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            try:
                loss = self.compute_loss(model, inputs, num_items_in_batch=None)
            except TypeError:  # older transformers
                loss = self.compute_loss(model, inputs)
        del inputs

        loss.backward()

        return loss.detach() / self.args.gradient_accumulation_steps

    def _get_train_sampler(self):
        """
        Provides entity sampler.
        """
        if self.privacy_args.poisson_sampling:
            # NOTE: sample_rate is set s.t. chosen batch size remains the same in average
            sample_rate = min(
                1.0,
                self.args.per_device_train_batch_size / self.train_dataset.num_rows,
            )
            logger.info(
                f"Poisson sampling is active, with a sampling rate of {sample_rate}",
            )
            train_sampler = PoissonEntitySampler(
                entity_mapping=self.entity_mapping,
                sample_rate=sample_rate,
            )
        else:
            train_sampler = ShuffledEntitySampler(
                entity_mapping=self.entity_mapping,
                batch_size=self.args.per_device_train_batch_size,
            )
        return train_sampler

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns a torch DataLoader that uses an entity-level sampler.
        """
        train_sampler = self._get_train_sampler()
        return DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Updated function `_save` in Trainer (transformers==4.37.2) to save the
        PeftModule when wrapping with Opacus GradSampleModule.
        TODO: When updating transformers, check for changes to this function.
        """
        if isinstance(self.model, GradSampleModule) and hasattr(self.model, "_module"):
            model_to_save = self.model._module
            if not isinstance(model_to_save, PeftModel):
                raise ValueError(f"Error saving model with type {type(model_to_save)}. Expected PeftModel.")
        else:
            model_to_save = self.model

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        supported_classes = (
            (modeling_utils.PreTrainedModel,)
            if not utils.is_peft_available()
            else (modeling_utils.PreTrainedModel, PeftModel)
        )
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(model_to_save, supported_classes):
            if state_dict is None:
                state_dict = model_to_save.state_dict()
            unwrapped_model = modeling_utils.unwrap_model(model_to_save)
            if isinstance(unwrapped_model, supported_classes):
                unwrapped_model.save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=self.args.save_safetensors,
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    try:
                        safetensors.torch.save_file(
                            state_dict,
                            os.path.join(output_dir, utils.SAFE_WEIGHTS_NAME),
                        )
                    except Exception as e:
                        logger.info(f"Error saving safetensors: {e}")
                        torch.save(state_dict, os.path.join(output_dir, utils.WEIGHTS_NAME))
                else:
                    torch.save(state_dict, os.path.join(output_dir, utils.WEIGHTS_NAME))
        else:
            model_to_save.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
