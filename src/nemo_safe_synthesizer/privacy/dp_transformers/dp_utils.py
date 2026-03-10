# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0 AND MIT

# This file has been adapted from the `dp-transformers` library.
# Original source: https://github.com/microsoft/dp-transformers/blob/main/src/dp_transformers/dp_utils.py
# See THIRD_PARTY.md for the original MIT license terms.

"""DP training utilities for Hugging Face Trainer and data collation.

Provides ``OpacusDPTrainer`` (DP-aware Trainer with entity-level sampling and
Opacus optimizer), ``DPCallback`` for Trainer hooks, data collators that
expose ``position_ids`` for per-sample gradients, and ``GradSampleModule``
wrapper with ``no_sync`` support.
"""

import os
from contextlib import contextmanager
from typing import Any, Optional, Sequence

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
    """Trainer callback that integrates Opacus DP-SGD with ``transformers.Trainer``.

    Handles per-step optimizer behavior (skip signal, step, zero_grad), optional
    RDP step accounting, and early stopping when ``max_epsilon`` is exceeded.
    Used with ``OpacusDPTrainer``; the trainer injects this callback when
    privacy arguments are enabled.

    Args:
        noise_multiplier: Gaussian noise scale for gradients.
        sampling_probability: Probability of a record being in a batch.
        accountant: Privacy accountant for epsilon computation and (if RDP) step tracking.
        max_epsilon: Stop training when computed epsilon exceeds this value.
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
        """Run DP optimizer step at the end of each gradient-accumulation substep.

        Signals the Opacus optimizer to skip the step, calls ``step()`` and
        ``zero_grad()`` on the underlying DP optimizer (or the optimizer itself
        if not wrapped by Accelerate). Required when using gradient accumulation
        so that the optimizer step runs once per micro-batch.

        Args:
            args: HF Trainer arguments.
            state: Current trainer state.
            control: Trainer control object (not modified).
            optimizer: The Trainer's optimizer (Opacus DP optimizer or AcceleratedOptimizer wrapping it).
            **kwargs: Additional callback keyword arguments.

        Raises:
            RuntimeError: If optimizer is None (callback cannot access optimizer).
        """
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
        """Clear gradients and update RDP accountant at the end of each optimizer step.

        Calls ``zero_grad()`` on the optimizer (Opacus expects this; Trainer does not
        call it by default). When using the RDP accountant (not PRV), increments the
        accountant step for accurate epsilon calculation.

        Args:
            args: Trainer training arguments (used to check gradient_accumulation_steps).
            state: Current trainer state.
            control: Trainer control object (not modified).
            optimizer: The Trainer's optimizer (required for ``zero_grad()``).
            **kwargs: Additional callback keyword arguments.

        Raises:
            RuntimeError: If gradient accumulation is used but ``on_substep_end`` was
                never called (e.g. transformers < 4.10.0), or if optimizer is None.
        """
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
        """Called when the Trainer is about to save a checkpoint. Ensures training
        stops before saving if the privacy budget would be exceeded.

        Args:
            args: HF Trainer arguments.
            state: Current trainer state (used for global_step).
            control: Trainer control object; ``should_training_stop`` may be set to True.
            **kwargs: Additional callback keyword arguments.

        Returns:
            TrainerControl with ``should_training_stop`` set to True if current
            epsilon exceeds ``max_epsilon``, otherwise unchanged.
        """
        return self._check_max_epsilon_exceeded(state, control)

    def on_evaluate(
        self,
        args: training_args.TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Check epsilon budget and stop training if ``max_epsilon`` is exceeded.

        Called when the Trainer runs evaluation. Ensures training stops before
        further steps if the privacy budget would be exceeded.

        Args:
            args: HF Trainer arguments.
            state: Current trainer state (used for global_step).
            control: Trainer control object; ``should_training_stop`` may be set to True.
            **kwargs: Additional callback keyword arguments.

        Returns:
            TrainerControl with ``should_training_stop`` set to True if current
            epsilon exceeds ``max_epsilon``, otherwise unchanged.
        """
        return self._check_max_epsilon_exceeded(state, control)

    def _check_max_epsilon_exceeded(self, state: TrainerState, control: TrainerControl) -> TrainerControl:
        """Set ``control.should_training_stop`` if computed epsilon exceeds ``max_epsilon``.

        Args:
            state: Current trainer state (uses ``global_step`` for epsilon computation).
            control: Trainer control object to update.

        Returns:
            The same ``control`` instance, with ``should_training_stop`` set to True
            when epsilon exceeds ``max_epsilon``.
        """
        eps = self.accountant.compute_epsilon(steps=state.global_step + 1)
        if eps > self._max_epsilon:
            logger.info("Max epsilon exceeded. Stopping training.")
            control.should_training_stop = True
        return control


class DataCollatorForPrivateCausalLanguageModeling(DataCollatorForLanguageModeling):
    """Adds ``position_ids`` for Opacus per-sample gradients.

    Trainer and model code often create ``position_ids`` inside the model
    forward pass, which Opacus cannot see. This collator builds ``position_ids``
    during batching so they are present in the batch and available for
    per-sample gradient computation. See https://github.com/huggingface/transformers/blob/5c1c72be5f864d10d0efe8ece0768d9ed6ee4fdd/src/transformers/models/mistral/modeling_mistral.py#L379
    for an example.

    Args:
        tokenizer: Tokenizer for padding and encoding.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer=tokenizer, mlm=False)

    def __call__(self, examples: list[list[int] | torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Collate examples into a batch and add ``position_ids`` if missing.

        Args:
            examples: List of tokenized examples (lists, tensors, or dicts).

        Returns:
            Batch dict with ``input_ids``, ``labels``, and ``position_ids``.
        """
        batch = super().__call__(examples)

        if "position_ids" not in batch:
            input_ids = batch["input_ids"]
            batch["position_ids"] = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device).repeat(
                input_ids.shape[0], 1
            )
        return batch


class DataCollatorForPrivateTokenClassification(DataCollatorForTokenClassification):
    """Collator for token classification that adds ``position_ids`` for Opacus.

    Same rationale as ``DataCollatorForPrivateCausalLanguageModeling``: ensures
    ``position_ids`` are in the batch for per-sample gradient computation.

    Args:
        tokenizer: Tokenizer for padding and encoding.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer=tokenizer)

    def __call__(
        self, examples: list[list[int] | torch.Tensor | dict[str, torch.TensorType]]
    ) -> dict[str, torch.Tensor]:
        """Collate examples into a batch and add ``position_ids`` if missing.

        Args:
            examples: List of tokenized examples (lists, tensors, or dicts).

        Returns:
            Batch dict with ``input_ids``, ``labels``, and ``position_ids``.
        """
        batch = super().__call__(examples)

        if "position_ids" not in batch:
            input_ids = batch["input_ids"]
            batch["position_ids"] = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device).repeat(
                input_ids.shape[0], 1
            )
        return batch


class GradSampleModule(opacus.GradSampleModule):
    """Opacus GradSampleModule with ``no_sync`` for Hugging Face Trainer.

    Trainer expects a ``no_sync`` context manager to defer gradient sync in
    distributed settings. This wrapper provides a no-op ``no_sync`` so the
    Trainer API is satisfied.
    """

    @contextmanager
    def no_sync(self):
        """Context manager that does nothing; required by Trainer's expected API."""
        yield


def create_entity_mapping(entity_column_values: list) -> Sequence[Sequence[int]]:
    """Build a mapping from each entity to its dataset indices.

    Groups rows by the entity column; each group's indices are the dataset
    positions for that entity. Entity order follows groupby sort; order within
    a group is preserved.

    Args:
        entity_column_values: List of entity IDs aligned with dataset rows
            (e.g. one value per row in the same order).

    Returns:
        Sequence of sequences: for entity i, result[i] is the list of dataset
        indices belonging to that entity.
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
    """DP-aware Trainer for PEFT/LoRA fine-tuning with Opacus.

    Adapts Hugging Face Trainer for differential privacy: uses entity-level
    (or record-level) sampling, wraps the model in ``GradSampleModule`` and
    the optimizer in Opacus ``DPOptimizer``, and avoids double-scaling of
    loss by gradient accumulation. Saves only the PEFT/LoRA adapter weights.

    Args:
        train_dataset: Dataset for training.
        model: Base model (will be wrapped with GradSampleModule).
        args: Training arguments (e.g. ``TrainingArguments``).
        privacy_args: DP parameters (epsilon, delta, noise, clipping). Required.
        data_fraction: If set, scales effective number of epochs for privacy math.
        true_dataset_size: Override number of entities/records for privacy accounting.
        entity_column_values: If set, entity-level DP; each value is the entity ID
            for the corresponding dataset row. If None, record-level DP (one entity
            per row).
        callbacks: Additional Trainer callbacks.
        secure_mode: If True, use secure RNG for noise (recommended).
        **kwargs: Passed to ``Trainer`` (e.g. eval_dataset, tokenizer, data_collator).

    Attributes:
        accountant: Privacy accountant used for epsilon computation.
        entity_mapping: For entity i, list of dataset indices in that entity.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        model: modeling_utils.PreTrainedModel | torch.nn.Module,
        args=None,
        privacy_args: PrivacyArguments | None = None,
        data_fraction: float | None = None,
        true_dataset_size: int | None = None,
        entity_column_values: list | None = None,
        callbacks: list[TrainerCallback] | None = None,
        secure_mode: bool | None = True,
        **kwargs: dict,
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

    def get_epsilon(self) -> float:
        """
        Uses the trainer's privacy accountant and the current number of
        optimizer steps to return the epsilon consumed so far.
        """
        return self.accountant.compute_epsilon(self.state.global_step)

    @property
    def sampling_probability(self) -> float:
        """Probability that an entity is included in a batch (capped at 1.0).

        For record-level DP (one entity per row), it is $min(1, (per_device_batch_size × gradient_accumulation_steps) / n_entities)$.
        For entity-level DP, n_entities can be small so the ratio may exceed 1;
        the result is capped at 1.0. Used as the sampling probability in the
        privacy accountant for ε computation.
        """
        return min(
            1.0,
            self.train_args.per_device_train_batch_size
            * self.train_args.gradient_accumulation_steps
            / self.true_dataset_size,
        )

    @property
    def num_steps(self) -> int:
        """The number of optimizer steps used for privacy accounting.

        Either user-supplied (via ``max_steps`` when ``true_num_epochs == -1``)
        or determined from ``num_train_epochs``. When the user specifies
        ``num_train_epochs``, we determine ``num_steps`` from
        ``sampling_probability`` so we pass over each entity roughly once per
        epoch, similarly to passing over each record once per epoch in
        record-level training.

        Always at least 1, because we add 1 to ``1 / sampling_probability``;
        this can happen when there are fewer entities than
        ``batch_size * gradient_accumulation_steps`` (e.g. 4 * 8 = 32).
        Used to determine the privacy budget (noise multiplier and epsilon)
        during training.
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
        """Create the base optimizer then wrap it with Opacus DPOptimizer."""
        _ = super().create_optimizer()

        class DPOptimizer(opacus.optimizers.DPOptimizer):  # ty: ignore[unresolved-attribute]
            """DPOptimizer that delegates ``param_groups`` to the inner optimizer.

            Hugging Face AcceleratedOptimizer replaces ``param_groups``; Opacus
            expects to mutate it. This subclass forwards get/set to the inner
            optimizer so learning rate scheduling and other param_group updates work.
            """

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
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        """Run one training step and return the loss scaled for logging.

        Forward pass and backward are performed as usual. Loss is not scaled by
        batch size or per-sample factors here: Opacus handles per-sample gradient
        scaling. The returned value is the raw loss divided by
        ``gradient_accumulation_steps`` so that the logged loss matches the
        effective per-step loss (averaged over accumulation steps).

        Args:
            model: The model to train (wrapped in ``GradSampleModule``).
            inputs: Batch of inputs (e.g. ``input_ids``, ``labels``, ``position_ids``).
            num_items_in_batch: Unused; passed for API compatibility. Opacus
                handles scaling; we pass ``None`` to avoid double-scaling.

        Returns:
            Detached loss tensor scaled by 1 / ``gradient_accumulation_steps``,
            for logging only (optimizer step is driven by the callback).
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
        """Return the entity-level (or record-level) sampler for training."""
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
        """DataLoader with entity-level sampler and DP data collator."""
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
        """Save the PEFT adapter (unwrap GradSampleModule) and tokenizer.

        Overrides Trainer._save so that when the model is wrapped with
        GradSampleModule we save the inner PEFT model, not the wrapper.
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
