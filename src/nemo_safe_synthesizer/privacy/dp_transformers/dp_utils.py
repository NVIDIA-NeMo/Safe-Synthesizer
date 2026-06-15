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

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from threading import Lock
from types import MethodType
from typing import Any, Literal, cast

import opacus
import pandas as pd
import safetensors.torch  # transformers v5 makes safetensors a hard dep
import torch
from accelerate.optimizer import AcceleratedOptimizer
from datasets import Dataset
from opacus.accountants import RDPAccountant
from packaging.version import InvalidVersion, Version
from peft import PeftModel
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
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

from ...config.training import TrainingMemoryControls
from ...observability import NvmlPeakSampler, get_logger
from ...training.training_observability import TRAINING_COMPLETE_EVENT, TrainingObservability
from . import linear  # imported for side effects  # noqa
from .privacy_args import (
    PrivacyArguments,
    SafeSynthesizerAccountant,
)
from .sampler import (
    PoissonEntitySampler,
    ShuffledEntitySampler,
)

logger = get_logger(__name__)

GradSampleMode = Literal["hooks", "ghost"]
_MIN_GHOST_CLIPPING_OPACUS_VERSION = Version("1.6.0")

_CAUSAL_LM_LOSS_MEMORY_PROBE_INSTALLED = False
_CAUSAL_LM_LOSS_MEMORY_PROBE_LOCK = Lock()

# Original Transformers loss callables saved at install time so the opt-in probe
# can be reverted. The probe monkeypatches process-global state in
# ``transformers.loss.loss_utils``; without teardown it would leak into every
# model in the process for the rest of its lifetime.
_CAUSAL_LM_LOSS_MEMORY_PROBE_ORIGINAL_FN: Any | None = None
_CAUSAL_LM_LOSS_MEMORY_PROBE_PATCHED_MAPPING_KEYS: list[Any] = []


def _log_cuda_loss_memory(stage: str, logits: torch.Tensor, *, enabled: bool) -> None:
    """Log CUDA memory around the causal-LM fp32 logits upcast (no-op when disabled)."""
    if not enabled:
        return
    if not torch.cuda.is_available() or not logits.is_cuda:
        return

    device = logits.device
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated_bytes = torch.cuda.memory_allocated(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    logger.warning(
        f"CausalLM loss memory probe {stage}: "
        f"logits_shape={tuple(logits.shape)} "
        f"logits_dtype={logits.dtype} "
        f"logits_gib={logits.numel() * logits.element_size() / 1024**3:.2f} "
        f"allocated_gib={allocated_bytes / 1024**3:.2f} "
        f"reserved_gib={reserved_bytes / 1024**3:.2f} "
        f"reserved_unallocated_gib={(reserved_bytes - allocated_bytes) / 1024**3:.2f} "
        f"free_gib={free_bytes / 1024**3:.2f} "
        f"total_gib={total_bytes / 1024**3:.2f}"
    )


def _chunked_cross_entropy(
    logits: torch.Tensor,
    shift_labels: torch.Tensor,
    *,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | int | None,
    ignore_index: int,
    chunk_size: int,
    **kwargs,
) -> torch.Tensor:
    """Compute causal-LM cross entropy without upcasting all logits at once."""
    flat_logits = logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1).to(flat_logits.device)

    loss_sum = flat_logits.new_zeros((), dtype=torch.float32)
    valid_token_count = flat_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, flat_logits.shape[0], chunk_size):
        end = min(start + chunk_size, flat_logits.shape[0])
        chunk_labels = flat_labels[start:end]
        keep = chunk_labels != ignore_index
        if not torch.any(keep):
            continue
        chunk_logits = flat_logits[start:end][keep].float()
        chunk_labels = chunk_labels[keep]
        loss_sum = loss_sum + F.cross_entropy(
            chunk_logits,
            chunk_labels,
            reduction="sum",
        )
        valid_token_count = valid_token_count + keep.sum().to(dtype=torch.float32)

    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss_sum.device)
        return loss_sum / num_items_in_batch
    return loss_sum / valid_token_count.clamp_min(1)


def _install_causal_lm_loss_memory_probe(
    *,
    debug_loss_memory: bool,
    chunked_loss: bool,
    chunk_tokens: int,
    peak_recorder: Callable[[int], None] | None = None,
) -> bool:
    """Install the opt-in Transformers causal-LM loss probe from config flags.

    Args:
        debug_loss_memory: Log CUDA memory around the fp32 logits upcast and
            feed ``peak_recorder`` (when supplied) the fp32 logits size.
        chunked_loss: Compute cross entropy in token chunks instead of a single
            full-logits fp32 upcast.
        chunk_tokens: Token chunk size used when ``chunked_loss`` is set.
        peak_recorder: Optional sink receiving the fp32 logits byte count on
            each loss call when ``debug_loss_memory`` is set (used to summarize
            the peak upcast spike for observability).

    Returns ``True`` when this call installed the process-global probe and the
    caller must later uninstall it. Returns ``False`` when neither probe feature
    is enabled.
    """
    if not debug_loss_memory and not chunked_loss:
        return False

    from transformers.loss import loss_utils

    def probed_for_causal_lm_loss(
        logits,
        labels,
        vocab_size: int,
        num_items_in_batch: torch.Tensor | None = None,
        ignore_index: int = -100,
        shift_labels: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if debug_loss_memory and peak_recorder is not None and logits is not None:
            # fp32 logits size -- the spike the upcast materializes (or that
            # chunked loss avoids). float32 == 4 bytes/element.
            peak_recorder(logits.numel() * 4)
        _log_cuda_loss_memory("before_logits_float", logits, enabled=debug_loss_memory)
        if shift_labels is None:
            labels = F.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()

        if chunked_loss:
            return _chunked_cross_entropy(
                logits,
                shift_labels,
                vocab_size=vocab_size,
                num_items_in_batch=num_items_in_batch,
                ignore_index=ignore_index,
                chunk_size=chunk_tokens,
                **kwargs,
            )

        logits = logits.float()
        _log_cuda_loss_memory("after_logits_float", logits, enabled=debug_loss_memory)
        logits = logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1).to(logits.device)
        loss = loss_utils.fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
        return loss

    global _CAUSAL_LM_LOSS_MEMORY_PROBE_ORIGINAL_FN, _CAUSAL_LM_LOSS_MEMORY_PROBE_PATCHED_MAPPING_KEYS
    global _CAUSAL_LM_LOSS_MEMORY_PROBE_INSTALLED
    with _CAUSAL_LM_LOSS_MEMORY_PROBE_LOCK:
        if _CAUSAL_LM_LOSS_MEMORY_PROBE_INSTALLED:
            raise RuntimeError(
                "Causal-LM loss memory probe is already installed; concurrent DP training runs "
                "with debug_loss_memory or chunked_causal_lm_loss are not supported."
            )

        original = loss_utils.ForCausalLMLoss
        _CAUSAL_LM_LOSS_MEMORY_PROBE_ORIGINAL_FN = original
        _CAUSAL_LM_LOSS_MEMORY_PROBE_PATCHED_MAPPING_KEYS = []
        setattr(loss_utils, "ForCausalLMLoss", probed_for_causal_lm_loss)
        for loss_name, loss_fn in loss_utils.LOSS_MAPPING.items():
            if loss_fn is original:
                loss_utils.LOSS_MAPPING[loss_name] = probed_for_causal_lm_loss
                _CAUSAL_LM_LOSS_MEMORY_PROBE_PATCHED_MAPPING_KEYS.append(loss_name)
        _CAUSAL_LM_LOSS_MEMORY_PROBE_INSTALLED = True
    logger.warning(
        "Installed causal-LM loss wrapper for memory diagnostics/experiments "
        f"(debug_loss_memory={debug_loss_memory}, chunked_loss={chunked_loss}, chunk_tokens={chunk_tokens})"
    )
    return True


def _uninstall_causal_lm_loss_memory_probe(installed: bool = True) -> None:
    """Revert the opt-in causal-LM loss probe, restoring Transformers globals.

    Idempotent and safe to call when the probe was never installed. Resets the
    installed flag so a subsequent training run re-installs cleanly.
    """
    if not installed:
        return

    global _CAUSAL_LM_LOSS_MEMORY_PROBE_INSTALLED, _CAUSAL_LM_LOSS_MEMORY_PROBE_ORIGINAL_FN
    from transformers.loss import loss_utils

    with _CAUSAL_LM_LOSS_MEMORY_PROBE_LOCK:
        if not _CAUSAL_LM_LOSS_MEMORY_PROBE_INSTALLED:
            return

        original = _CAUSAL_LM_LOSS_MEMORY_PROBE_ORIGINAL_FN
        if original is not None:
            setattr(loss_utils, "ForCausalLMLoss", original)
            for loss_name in _CAUSAL_LM_LOSS_MEMORY_PROBE_PATCHED_MAPPING_KEYS:
                loss_utils.LOSS_MAPPING[loss_name] = original

        _CAUSAL_LM_LOSS_MEMORY_PROBE_PATCHED_MAPPING_KEYS.clear()
        _CAUSAL_LM_LOSS_MEMORY_PROBE_ORIGINAL_FN = None
        _CAUSAL_LM_LOSS_MEMORY_PROBE_INSTALLED = False
    logger.warning("Reverted causal-LM loss wrapper installed for memory diagnostics/experiments")


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
        optimizer: torch.optim.Optimizer | None = None,
        **kwargs,
    ) -> None:
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
        dp_optimizer.signal_skip_step(do_skip=True)  # ty: ignore[unresolved-attribute]
        dp_optimizer.step()
        dp_optimizer.zero_grad()

        self._on_substep_end_was_called = True

    def on_step_end(
        self,
        args: training_args.TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        optimizer: torch.optim.Optimizer | None = None,
        **kwargs,
    ) -> None:
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
            acct = cast(RDPAccountant, self.accountant.accountant)
            acct.step(
                noise_multiplier=self.noise_multiplier,
                sample_rate=self.sampling_probability,
            )

    def on_save(
        self,
        args: training_args.TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
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
    ) -> TrainerControl:
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

    def __call__(
        self,
        features,
        return_tensors: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Collate examples into a batch and add ``position_ids`` if missing.

        Args:
            features: Tokenized examples expected by HF data collators.

        Returns:
            Batch dict with ``input_ids``, ``labels``, and ``position_ids``.
        """
        batch = super().__call__(features, return_tensors=return_tensors)

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
        self,
        features,
        return_tensors: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Collate examples into a batch and add ``position_ids`` if missing.

        Args:
            features: Tokenized examples expected by HF data collators.

        Returns:
            Batch dict with ``input_ids``, ``labels``, and ``position_ids``.
        """
        batch = super().__call__(features, return_tensors=return_tensors)

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
    def no_sync(self) -> Iterator[None]:
        """Context manager that does nothing; required by Trainer's expected API."""
        yield


@contextmanager
def _no_sync_context(_model: nn.Module) -> Iterator[None]:
    """Context manager that does nothing; required by Trainer's expected API."""
    yield


def _get_opacus_version() -> Version:
    """Return the installed Opacus package version."""
    try:
        return Version(version("opacus"))
    except PackageNotFoundError as err:
        raise RuntimeError("Could not determine installed Opacus version.") from err
    except InvalidVersion as err:
        raise RuntimeError("Could not parse installed Opacus version.") from err


def _load_ghost_clipping_classes() -> tuple[type[Any], type[Any], type[Any]]:
    """Load Opacus Fast/Ghost Gradient Clipping classes after version gating."""
    installed = _get_opacus_version()
    if installed < _MIN_GHOST_CLIPPING_OPACUS_VERSION:
        raise RuntimeError(
            "DP grad_sample_mode='ghost' requires opacus>=1.6.0 because causal-LM "
            f"ignore_index masking is required; found opacus=={installed}."
        )

    try:
        grad_sample_module = import_module("opacus.grad_sample.grad_sample_module_fast_gradient_clipping")
        optimizer_module = import_module("opacus.optimizers.optimizer_fast_gradient_clipping")
        loss_module = import_module("opacus.utils.fast_gradient_clipping_utils")
    except ImportError as err:
        raise RuntimeError("Could not import Opacus Fast/Ghost Gradient Clipping classes.") from err

    return (
        grad_sample_module.GradSampleModuleFastGradientClipping,
        optimizer_module.DPOptimizerFastGradientClipping,
        loss_module.DPLossFastGradientClipping,
    )


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
    entity_mapping = [g.index.tolist() for _, g in entities.groupby("entity")]
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
        grad_sample_mode: Opacus per-sample gradient mode (``"hooks"`` or ``"ghost"``).
        memory_controls: Advanced memory/OOM controls (chunked loss, bf16 disable,
            loss-memory diagnostics). Defaults to all-off when omitted.
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
        args: training_args.TrainingArguments | None = None,
        privacy_args: PrivacyArguments | None = None,
        grad_sample_mode: GradSampleMode = "hooks",
        memory_controls: TrainingMemoryControls | None = None,
        data_fraction: float | None = None,
        true_dataset_size: int | None = None,
        entity_column_values: list | None = None,
        callbacks: list[TrainerCallback] | None = None,
        secure_mode: bool | None = True,
        **kwargs: Any,
    ) -> None:
        if args is None:
            raise ValueError("TrainingArguments (args) is required for OpacusDPTrainer")
        if privacy_args is None:
            raise ValueError("PrivacyArguments is required for OpacusDPTrainer")
        self.train_args = args
        self.privacy_args = privacy_args
        self.grad_sample_mode = grad_sample_mode
        self.memory_controls = memory_controls or TrainingMemoryControls()
        self._peak_loss_logits_bytes = 0
        #: Last training-complete event, set by :meth:`train`; the backend forwards it to wandb.
        self.last_training_observability: TrainingObservability | None = None
        self.dp_loss: Any | None = None
        self._ghost_optimizer_cls: type[Any] | None = None
        self._ghost_loss_cls: type[Any] | None = None
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
        pa = self.privacy_args
        assert pa.use_prv is not None
        assert pa.noise_multiplier is not None

        if grad_sample_mode == "hooks":
            model = GradSampleModule(model)
        elif grad_sample_mode == "ghost":
            grad_sample_cls, self._ghost_optimizer_cls, self._ghost_loss_cls = _load_ghost_clipping_classes()
            model = grad_sample_cls(
                model,
                max_grad_norm=pa.per_sample_max_grad_norm,
                loss_reduction="mean",
            )
            model.no_sync = MethodType(_no_sync_context, model)
        else:
            raise ValueError(f"Unsupported DP grad_sample_mode: {grad_sample_mode!r}")

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            callbacks=callbacks,
            **kwargs,
        )
        self.accountant = SafeSynthesizerAccountant(
            use_prv=pa.use_prv,
            noise_multiplier=pa.noise_multiplier,
            sampling_probability=self.sampling_probability,
            delta=self.privacy_args.target_delta,
            num_steps=self.num_steps,
        )
        self.dp_callback = DPCallback(
            noise_multiplier=pa.noise_multiplier,
            sampling_probability=self.sampling_probability,
            accountant=self.accountant,
            max_epsilon=float("inf") if self.privacy_args.target_epsilon is None else self.privacy_args.target_epsilon,
        )
        self.add_callback(self.dp_callback)

    def get_epsilon(self) -> float:
        """Calculate the epsilon after model training completes."""
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

    def create_optimizer(self, model: nn.Module | None = None) -> torch.optim.Optimizer:
        """Create the base optimizer then wrap it with Opacus DPOptimizer."""
        _ = model  # Signature matches transformers v5; base method uses self.model.
        _ = super().create_optimizer()

        class DPOptimizer(opacus.optimizers.DPOptimizer):
            """DPOptimizer that delegates ``param_groups`` to the inner optimizer.

            Hugging Face AcceleratedOptimizer replaces ``param_groups``; Opacus
            expects to mutate it. This subclass forwards get/set to the inner
            optimizer so learning rate scheduling and other param_group updates work.
            """

            @property
            def param_groups(self) -> list:
                return self.original_optimizer.param_groups

            @param_groups.setter
            def param_groups(self, param_groups: list) -> None:
                self.original_optimizer.param_groups = param_groups

        # TODO: explore better mitigation for precision based attacks on finite
        # precision devices
        # https://tpdp.journalprivacyconfidentiality.org/2022/papers/HaneyDHSH22.pdf
        pa = self.privacy_args
        assert pa is not None and pa.per_sample_max_grad_norm is not None and pa.noise_multiplier is not None
        assert self.optimizer is not None
        optimizer_cls = self._make_ghost_optimizer_cls() if self.grad_sample_mode == "ghost" else DPOptimizer
        optimizer_kwargs: dict[str, Any] = {}
        if self.grad_sample_mode == "ghost":
            optimizer_kwargs["loss_reduction"] = "mean"

        self.optimizer = optimizer_cls(
            optimizer=self.optimizer,
            noise_multiplier=pa.noise_multiplier,
            max_grad_norm=pa.per_sample_max_grad_norm,
            expected_batch_size=self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps,
            secure_mode=self.secure_mode,
            **optimizer_kwargs,
        )
        if self.grad_sample_mode == "ghost":
            if self._ghost_loss_cls is None:
                raise RuntimeError("Ghost clipping loss class is not initialized.")
            criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
            self.dp_loss = self._ghost_loss_cls(
                self.model,
                self.optimizer,
                criterion,
                loss_reduction="mean",
            )

        return self.optimizer

    def _make_ghost_optimizer_cls(self) -> type[Any]:
        """Create a Fast/Ghost clipping optimizer class with Trainer-compatible param groups."""
        if self._ghost_optimizer_cls is None:
            raise RuntimeError("Ghost clipping optimizer class is not initialized.")
        base_cls = self._ghost_optimizer_cls

        class DPOptimizerGhostClipping(base_cls):
            """Fast/Ghost clipping optimizer with Trainer-compatible param groups."""

            @property
            def param_groups(self) -> list:
                return self.original_optimizer.param_groups

            @param_groups.setter
            def param_groups(self, param_groups: list) -> None:
                self.original_optimizer.param_groups = param_groups

        return DPOptimizerGhostClipping

    def _record_peak_loss_logits(self, nbytes: int) -> None:
        """Track the largest fp32 logits tensor seen by the loss (for observability)."""
        if nbytes > self._peak_loss_logits_bytes:
            self._peak_loss_logits_bytes = nbytes

    def _install_loss_memory_probe(self) -> bool:
        """Install the causal-LM loss probe from ``self.memory_controls`` (idempotent)."""
        mc = self.memory_controls
        if self.grad_sample_mode == "ghost":
            if mc.debug_loss_memory or mc.chunked_causal_lm_loss:
                logger.warning(
                    "DP memory controls debug_loss_memory and chunked_causal_lm_loss are "
                    "ignored for grad_sample_mode='ghost' because ghost clipping bypasses "
                    "Transformers' causal-LM loss."
                )
            return False

        return _install_causal_lm_loss_memory_probe(
            debug_loss_memory=mc.debug_loss_memory,
            chunked_loss=mc.chunked_causal_lm_loss,
            chunk_tokens=mc.chunked_causal_lm_loss_tokens,
            peak_recorder=self._record_peak_loss_logits,
        )

    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Run training, sampling peak VRAM and emitting a ``training.complete`` event.

        Installs the opt-in causal-LM loss probe, then wraps ``Trainer.train``
        in an :class:`NvmlPeakSampler`. In a ``finally``, emits a single
        :class:`TrainingObservability` event and reverts the probe. Install and
        teardown bracket the run in this one method, and the teardown runs even
        when training raises, so the process-global Transformers loss patch
        never leaks into a later model in the same process.
        """
        self._peak_loss_logits_bytes = 0
        probe_installed = self._install_loss_memory_probe()
        sampler = NvmlPeakSampler()
        try:
            with sampler:
                return super().train(*args, **kwargs)
        finally:
            self._emit_training_observability(sampler.peak_gb)
            _uninstall_causal_lm_loss_memory_probe(installed=probe_installed)

    def _emit_training_observability(self, peak_vram_gb: float | None) -> None:
        """Assemble the ``training.complete`` event, log it, and stash it.

        Writes the structured log line and stores the event on
        :attr:`last_training_observability` for the backend to mirror to wandb;
        this layer does not depend on the CLI/wandb module. Best-effort: any
        failure here is logged and swallowed so observability never masks a
        training error propagating through :meth:`train`'s ``finally``.
        """
        try:
            per_device = getattr(self.args, "per_device_train_batch_size", None)
            grad_accum = getattr(self.args, "gradient_accumulation_steps", None)
            effective = per_device * grad_accum if per_device is not None and grad_accum is not None else None
            peak_loss_logits_gb = self._peak_loss_logits_bytes / 1024**3 if self._peak_loss_logits_bytes > 0 else None
            event = TrainingObservability(
                peak_vram_gb=peak_vram_gb,
                peak_loss_logits_gb=peak_loss_logits_gb,
                per_device_train_batch_size=per_device,
                gradient_accumulation_steps=grad_accum,
                effective_batch_size=effective,
                grad_sample_mode=self.grad_sample_mode,
            )
            self.last_training_observability = event
            logger.runtime.info(TRAINING_COMPLETE_EVENT, extra={"ctx": event.model_dump()})
        except Exception as exc:  # noqa: BLE001 -- observability must never break training
            try:
                logger.warning(f"failed to emit training observability: {exc}")
            except Exception:  # noqa: BLE001 -- a faulty log handler must not raise from finally
                pass

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | int | None = None,
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
        getattr(self.optimizer, "train", lambda: None)()

        # Pass `num_items_in_batch=None` so the HF Trainer skips its built-in
        # per-token scaling; Opacus already applies per-sample gradient scaling
        # and we divide the logged loss by gradient_accumulation_steps below.
        # The loss memory probe is installed once in `train()`, not per step.
        inputs = self._prepare_inputs(inputs)
        if self.grad_sample_mode == "ghost":
            return self._ghost_clipping_training_step(model, inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=None)
        if isinstance(loss, tuple):
            loss = loss[0]
        del inputs

        loss.backward()

        return loss.detach() / self.args.gradient_accumulation_steps

    def _ghost_clipping_training_step(self, model: nn.Module, inputs: dict[str, Any]) -> torch.Tensor:
        """Run one Fast/Ghost Gradient Clipping training step."""
        with self.compute_loss_context_manager():
            loss = self._compute_ghost_clipping_loss(model, inputs)
        del inputs

        loss.backward()
        loss_value = torch.as_tensor(loss.item(), device=self.args.device)
        return loss_value / self.args.gradient_accumulation_steps

    def _compute_ghost_clipping_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
    ) -> Any:
        """Compute per-sequence causal-LM loss for Opacus ghost clipping."""
        if self.dp_loss is None:
            raise RuntimeError("Ghost clipping loss wrapper is not initialized.")
        labels = inputs.get("labels")
        if labels is None:
            raise RuntimeError("Ghost clipping DP training requires labels in the batch.")

        model_inputs = dict(inputs)
        labels = model_inputs.pop("labels")
        shift_labels = labels[..., 1:].contiguous()
        valid_positions = torch.nonzero((shift_labels != -100).any(dim=0), as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            raise RuntimeError("Ghost clipping DP training requires at least one non-ignored label.")
        model_inputs.setdefault("logits_to_keep", valid_positions.to(labels.device))

        forward_model = self.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
        outputs = forward_model(**model_inputs)
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, dict):
            logits = outputs.get("logits")
        if logits is None:
            raise RuntimeError("Ghost clipping DP training requires model outputs with logits.")
        if logits.ndim != 3 or logits.shape[1] != valid_positions.numel():
            raise RuntimeError(
                "Ghost clipping DP training expected the model to honor logits_to_keep "
                f"with {valid_positions.numel()} kept positions, got logits shape {tuple(logits.shape)}."
            )

        shift_logits = logits.contiguous()
        shift_labels = shift_labels.index_select(1, valid_positions.to(shift_labels.device)).to(shift_logits.device)
        return self.dp_loss(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            shape=shift_logits.shape,
        )

    def _get_train_sampler(self, train_dataset: Dataset | None = None) -> torch.utils.data.Sampler | None:  # ty: ignore[invalid-method-override] -- HF Trainer stub imprecision
        """Return the entity-level (or record-level) sampler for training."""
        ds = train_dataset if train_dataset is not None else self.train_dataset
        privacy_args = self.privacy_args
        if privacy_args is not None and privacy_args.poisson_sampling:
            assert isinstance(ds, Dataset), "train_dataset must be a Dataset"
            # NOTE: sample_rate is set s.t. chosen batch size remains the same in average
            num_rows = getattr(ds, "num_rows", len(ds))
            sample_rate = min(
                1.0,
                self.args.per_device_train_batch_size / num_rows,
            )
            logger.info(
                f"Poisson sampling is active, with a sampling rate of {sample_rate}",
            )
            return PoissonEntitySampler(
                entity_mapping=self.entity_mapping,
                sample_rate=sample_rate,
            )
        return ShuffledEntitySampler(
            entity_mapping=self.entity_mapping,
            batch_size=self.args.per_device_train_batch_size,
        )

    def get_train_dataloader(self) -> DataLoader:
        """Returns a torch DataLoader that uses an entity-level sampler."""
        train_dataset = self.train_dataset
        assert isinstance(train_dataset, Dataset)
        train_sampler = self._get_train_sampler(train_dataset)
        return DataLoader(
            cast(TorchDataset, train_dataset),
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _save(self, output_dir: str | None = None, state_dict: dict[str, Any] | None = None) -> None:
        """Save the PEFT adapter (unwrap GradSampleModule) and tokenizer.

        Overrides Trainer._save so that when the model is wrapped with
        GradSampleModule we save the inner PEFT model, not the wrapper.
        Both grad-sample modes wrap the PEFT model and expose it as ``_module``
        (hooks -> ``GradSampleModule``, ghost ->
        ``GradSampleModuleFastGradientClipping``), so unwrap whenever that
        attribute is present.
        TODO: When updating transformers, check for changes to this function.
        """
        if hasattr(self.model, "_module"):
            model_to_save = self.model._module
            if not isinstance(model_to_save, PeftModel):
                raise ValueError(f"Error saving model with type {type(model_to_save)}. Expected PeftModel.")
        else:
            model_to_save = self.model

        assert model_to_save is not None
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if not isinstance(output_dir, str):
            raise ValueError("output_dir must be a string (neither output_dir nor self.args.output_dir was set)")
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
                assert model_to_save is not None
                state_dict = model_to_save.state_dict()
            unwrapped_model = modeling_utils.unwrap_model(model_to_save)
            if isinstance(unwrapped_model, supported_classes):
                unwrapped_model.save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if getattr(self.args, "save_safetensors", False):
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
            )

        processor = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if processor is not None:
            processor.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
