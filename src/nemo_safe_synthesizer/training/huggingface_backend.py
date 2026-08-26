# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Trainer backend for LoRA fine-tuning."""

from __future__ import annotations

import io
import logging
import math
import time
from collections.abc import Callable
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from peft import LoftQConfig, LoraConfig, TaskType, prepare_model_for_kbit_training
from peft import get_peft_model as get_peft_model_hf
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForTokenClassification,
    EvalPrediction,
    IntervalStrategy,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PrinterCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_pt_utils import get_model_param_count
from transformers.utils.quantization_config import QuantizationConfigMixin

from .. import utils
from ..cli.artifact_structure import BoundDir
from ..config.autoconfig import AutoConfigResolver
from ..data_processing.assembler import TrainingExampleAssembler
from ..data_processing.dataset import make_json_schema
from ..defaults import (
    DEFAULT_VALID_RECORD_EVAL_BATCH_SIZE,
    EVAL_STEPS,
    FIXED_RUNTIME_LORA_ARGS,
)
from ..errors import DataError, ParameterError
from ..generation.processors import create_processor
from ..llm.utils import (
    ModelRef,
    cleanup_memory,
    get_device_map,
    get_max_memory_map,
    get_quantization_config,
    load_fast_tokenizer,
)
from ..observability import get_logger, traced_runtime, traced_user
from ..privacy.dp_transformers.dp_utils import (
    DataCollatorForPrivateTokenClassification,
    OpacusDPTrainer,
)
from ..privacy.dp_transformers.privacy_args import PrivacyArguments
from ..tokenization import WorkloadKind, bind_tokenizer
from ..training.backend import (
    NSSTrainerResult,
    TrainingBackend,
)
from ..training.callbacks import (
    InferenceEvalCallback,
    ProgressBarCallback,
    SafeSynthesizerWorkerCallback,
)
from ..training.timeseries_preprocessing import process_timeseries_data
from ..utils import write_json

if TYPE_CHECKING:
    from ..config.training import QuantizationScheme

logger = get_logger(__name__)

DEFAULT_ROPE_THETA = 10000.0

FIXED_RUNTIME_TRAINING_ARGS = {
    # the training time is set by the number of training records
    "num_train_epochs": 1,
    "save_strategy": IntervalStrategy.EPOCH,
    "logging_steps": 1,
    "logging_strategy": IntervalStrategy.STEPS,
    "per_device_eval_batch_size": 2,
    "optim": "paged_adamw_32bit",
    "bf16": True,
    "ddp_find_unused_parameters": False,
}
"""Training arguments fixed by Safe Synthesizer at runtime.

Training duration is controlled by ``num_input_records_to_sample`` and the
assembled ``data_fraction``, not by epochs. These values keep the HuggingFace
Trainer behavior stable across CLI and SDK entry points.
"""


class HuggingFaceBackend(TrainingBackend):
    """Training backend built on the HuggingFace ``Trainer``.

    Handles model loading (``AutoModelForCausalLM``), LoRA/QLoRA wrapping,
    RoPE scaling, optional differential-privacy training via
    [`OpacusDPTrainer`][nemo_safe_synthesizer.privacy.dp_transformers.dp_utils.OpacusDPTrainer],
    and artifact persistence (adapter, schema, metadata).

    Quantized training prepares the model with
    ``prepare_model_for_kbit_training`` before applying LoRA. Non-quantized
    training enables gradient checkpointing, input gradients, and disables
    ``use_cache`` before wrapping the model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer_type: type[Trainer] | partial[OpacusDPTrainer] = Trainer
        self.model_loader_type = AutoModelForCausalLM
        self.training_output_dir = Path(self.workdir.train.cache)
        self.model_ref = ModelRef.parse(self.params.training.pretrained_model)
        self.autoconfig = AutoConfig.from_pretrained(
            self.model_ref.target(),
            trust_remote_code=self.model_ref.trust_remote_code,
        )

    def _load_pretrained_model(self, **model_args: Any) -> None:
        """Load the pretrained model and tokenizer via ``AutoModelForCausalLM``."""
        self.autoconfig.max_position_embeddings = (
            model_args.pop("max_seq_length", None) or self.model_metadata.max_seq_length
        )
        self._normalize_rope_parameters()
        self.model = self.model_loader_type.from_pretrained(**self.framework_load_params, config=self.autoconfig)

        workload_kind = WorkloadKind.TIME_SERIES if self.params.time_series.is_timeseries else WorkloadKind.TABULAR
        persisted = self.model_metadata.tokenization
        if persisted is not None:
            if persisted.workload_kind != workload_kind:
                raise ParameterError("Persisted tokenizer workload does not match the training workload.")
            self.tokenization = persisted
        else:
            admitted_native = self.model_metadata.tokenizer
            if not isinstance(admitted_native, PreTrainedTokenizerBase):
                admitted_native = load_fast_tokenizer(
                    self.model_ref.target(),
                    trust_remote_code=self.model_ref.trust_remote_code,
                    model_max_length=model_args.get("max_seq_length", None),
                )
            if model_max_length := model_args.get("max_seq_length"):
                admitted_native.model_max_length = model_max_length

            self.tokenization = bind_tokenizer(
                admitted_native,
                self.model_metadata,
                workload_kind=workload_kind,
            )
        self.tokenizer = cast(PreTrainedTokenizer, self.tokenization.native)
        self.model_metadata.tokenization = self.tokenization
        self.model_metadata.tokenizer = self.tokenizer

    # Constants for filtering trainer-specific kwargs
    # These keys are handled specially and should not be passed directly to model loading.
    # rope_scaling_factor is processed by _resolve_rope_scaling_factor and converted to
    # the properly formatted rope_scaling dict by _apply_rope_scaling.
    _TRAINER_SPECIFIC_KEYS = frozenset(
        {
            "params",
            "model_metadata",
            "training_dataset",
            "data_fraction",
            "true_dataset_size",
            "eval_dataset",
            "generation_eval",
            "callbacks",
            "action_executor",
            "verbose_logging",
            "maybe_split_dataset",
            "artifact_path",
            "rope_scaling_factor",
            "workdir",
        }
    )

    def _filter_model_kwargs(self, kwargs: dict) -> dict:
        """Filter out trainer-specific kwargs that should not be passed to model loading.

        Args:
            kwargs: Dictionary of keyword arguments.

        Returns:
            Dictionary with trainer-specific keys removed.
        """
        return {k: v for k, v in kwargs.items() if k not in self._TRAINER_SPECIFIC_KEYS}

    def _resolve_attn_implementation(self, configured: str) -> str:
        """Resolve attention implementation, falling back to sdpa if kernels is unavailable.

        Args:
            configured: The configured attention implementation string.

        Returns:
            The resolved attention implementation string.
        """
        if configured.startswith("kernels-community/"):
            try:
                import kernels  # noqa: F401

                return configured
            except ImportError:
                logger.warning(
                    f"kernels package not installed, cannot use '{configured}'. "
                    "Falling back to 'sdpa'. Install with: pip install kernels"
                )
                return "sdpa"
        return configured

    def _build_base_framework_params(self, model_kwargs: dict) -> dict:
        """Build the base framework parameters for model loading.

        Args:
            model_kwargs: Filtered model keyword arguments.

        Returns:
            Dictionary of parameters for ``from_pretrained``.
        """
        return dict(
            pretrained_model_name_or_path=self.model_ref.target(),
            trust_remote_code=self.model_ref.trust_remote_code,
            device_map=model_kwargs.pop(
                "device_map",
                get_device_map(
                    self.model_ref.target(),
                    autoconfig=self.autoconfig,
                    trust_remote_code=self.model_ref.trust_remote_code,
                ),
            ),
            attn_implementation=model_kwargs.pop(
                "attn_implementation", self._resolve_attn_implementation(self.params.training.attn_implementation)
            ),
            dtype=model_kwargs.pop("dtype", torch.bfloat16),
            **model_kwargs,
        )

    def _resolve_quantization_scheme(self) -> "QuantizationScheme":
        """Resolve which quantization scheme to apply.

        Prefers the explicit ``quantization_scheme`` field; falls back to
        the legacy ``quantization_bits`` alias when unset.
        """
        from ..config.training import QuantizationScheme

        cfg = self.params.training
        if cfg.quantization_scheme is not None:
            return cfg.quantization_scheme
        return QuantizationScheme.BNB_4BIT if cfg.quantization_bits == 4 else QuantizationScheme.BNB_8BIT

    def _get_quantization_config_if_enabled(self) -> QuantizationConfigMixin | None:
        """Build the v5 quantization config for the configured scheme, or None.

        Returns ``None`` when ``quantize_model`` is disabled. Otherwise
        resolves the scheme (new ``quantization_scheme`` field, or legacy
        ``quantization_bits`` fallback) and returns the corresponding v5
        config object — ``BitsAndBytesConfig`` for the bnb-* schemes,
        ``FineGrainedFP8Config``/``TorchAoConfig``/``Mxfp4Config`` for the
        others.
        """
        if not self.params.training.quantize_model:
            return None

        scheme = self._resolve_quantization_scheme()
        logger.info(f"Quantizing model with scheme={scheme.value}")
        return get_quantization_config(scheme)

    def _normalize_rope_parameters(self) -> None:
        """Ensure Transformers v5 ``rope_parameters`` includes ``rope_theta``."""
        rope_parameters = getattr(self.autoconfig, "rope_parameters", None)
        if not isinstance(rope_parameters, dict) or "rope_theta" in rope_parameters:
            return

        theta = rope_parameters.get("theta")
        if not isinstance(theta, (int, float)):
            rope_scaling = getattr(self.model_metadata, "rope_scaling", None)
            theta = getattr(rope_scaling, "theta", None)
        if not isinstance(theta, (int, float)):
            theta = getattr(self.autoconfig, "rope_theta", DEFAULT_ROPE_THETA)
        if not isinstance(theta, (int, float)):
            theta = DEFAULT_ROPE_THETA

        rope_parameters["rope_theta"] = float(theta)

    def _apply_rope_scaling(self, framework_params: dict, **kwargs: Any) -> None:
        """Apply rope scaling from model_metadata to the config.

        The RopeScaling configuration is now managed by the ModelMetadata class,
        which reads the model's native theta and rope_type from the HuggingFace config.

        Args:
            framework_params: The dictionary to update with rope scaling parameters.
        """
        rope_scaling = self.model_metadata.rope_scaling

        if rope_scaling is None:
            # No rope scaling configured
            setattr(self.autoconfig, "rope_scaling", getattr(self.autoconfig, "rope_scaling", None))
            logger.warning(f"rope_scaling not configured, using autoconfig default: {self.autoconfig.rope_scaling}")
            return

        # Convert RopeScaling to HuggingFace dict format or a None
        rope_scaling_dict = rope_scaling.to_hf_dict()

        if self.model_metadata.rope_parameters_location == "autoconfig":
            setattr(self.autoconfig, "rope_scaling", rope_scaling_dict)
            msg = f"rope_scaling set on autoconfig: {rope_scaling_dict}"
        elif self.model_metadata.rope_parameters_location == "automodel":
            framework_params["rope_scaling"] = rope_scaling_dict
            msg = f"rope_scaling set on framework_params: {rope_scaling_dict}"
        else:
            raise ValueError(f"Unknown rope_parameters_location: {self.model_metadata.rope_parameters_location}")

        logger.warning(msg)

    @traced_runtime("prepare_config")
    def prepare_config(self, add_max_memory: bool = True, **kwargs: Any) -> None:
        """Set common model arguments for initializing a model.

        Args:
            add_max_memory: Whether to add max_memory to the model arguments.
            **kwargs: Additional keyword arguments, overriding default arguments when set.
        """
        if self.framework_load_params:
            logger.info("already prepared loading parameters")
            return

        logger.info(f"preparing parameters for HF Automodel with model: {self.params.training.pretrained_model}")

        model_kwargs = self._filter_model_kwargs(kwargs)
        # Pop unconditionally: max_vram_fraction is an NSS-internal kwarg that
        # transformers does not accept, so it must not leak into
        # framework_load_params even when add_max_memory is False.
        frac = model_kwargs.pop("max_vram_fraction", None)

        if add_max_memory:
            if frac is None:
                frac = self.params.training.max_vram_fraction
            model_kwargs["max_memory"] = get_max_memory_map(max_vram_fraction=frac)

        framework_params = self._build_base_framework_params(model_kwargs)
        quant_config = self._get_quantization_config_if_enabled()
        if quant_config is not None:
            framework_params["quantization_config"] = quant_config

        self._apply_rope_scaling(framework_params=framework_params, **kwargs)
        self.framework_load_params = framework_params

    def _prepare_quantize_base(self, **quantize_params: dict) -> None:
        """Populate ``quant_params`` with LoRA and optional quantization settings.

        ``peft_implementation="loftq"`` adds a ``LoftQConfig`` initialized
        with the configured quantization bit width.
        """
        self.quant_params = dict(
            task_type=TaskType.CAUSAL_LM,
            init_lora_weights=True,
            inference_mode=False,
            r=self.params.training.lora_r,
            target_modules=self.params.training.lora_target_modules,
            peft_type=self.params.training.peft_implementation,
            lora_alpha=int(self.params.training.lora_alpha_over_r * self.params.training.lora_r),
            use_rslora=FIXED_RUNTIME_LORA_ARGS["use_rslora"],
        )

        if self.params.training.quantize_model:
            scheme = self._resolve_quantization_scheme()
            self.quant_params = self.quant_params | quantize_params
            logger.info(f"Quantizing model with scheme={scheme.value} (~{scheme.effective_bits} bits/param)")
            self.quant_params["peft_type"] = self.params.training.peft_implementation.upper()
            self.quant_params["init_lora_weights"] = True
            if self.params.training.peft_implementation == "loftq":
                if not scheme.is_bitsandbytes:
                    raise ParameterError(
                        f"peft_implementation='loftq' requires a bitsandbytes scheme; "
                        f"got quantization_scheme={scheme.value}. Use bnb-4bit or bnb-8bit, "
                        f"or switch to peft_implementation='qlora'/'lora'."
                    )
                logger.info(f"using loftq with {scheme.effective_bits} bits")
                self.quant_params["loftq_config"] = LoftQConfig(loftq_bits=scheme.effective_bits)

    def maybe_quantize(self, **quant_params: dict) -> None:
        """Apply LoRA wrapping (and optional k-bit quantization) to the model."""
        self._prepare_quantize_base(**quant_params)
        lora_config = LoraConfig(**self.quant_params)
        if not self.params.training.quantize_model:
            self.model.gradient_checkpointing_enable()
            # see https://discuss.huggingface.co/t/i-used-to-have-no-problem-with-peft-fine-tuning-after-hundreds-of-trainings-but-now-i-have-encountered-the-error-runtimeerror-element-0-of-tensors-does-not-require-grad-and-does-not-have-a-grad-fn/168829/3
            self.model.enable_input_require_grads()  # critical with PEFT + checkpointing
            self.model.config.use_cache = False  # cache off during training
        else:
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)

        if not isinstance(self.model, PreTrainedModel):
            raise TypeError(f"Expected PreTrainedModel, got {type(self.model)}")
        peft_model = get_peft_model_hf(self.model, peft_config=lora_config)
        self.model = peft_model  # ty: ignore[invalid-assignment] -- get_peft_model returns PeftModel|PeftMixedModel; PeftMixedModel is never produced here (single adapter)
        parameter_count = get_model_param_count(self.model, trainable_only=True) / 1e6
        logger.info(
            f"Using PEFT - {parameter_count:.2f} million parameters are trainable",
        )

    def load_model(self, **model_args: Any) -> None:
        """Load an ``AutoModelForCausalLM`` instance with specified arguments.

        Args:
            **model_args: Additional keyword arguments for model configuration,
                passed directly to ``AutoModelForCausalLM.from_pretrained()``.
        """
        logger.info(f"loading pretrained model: {self.params.training.pretrained_model}")
        self.prepare_config(**model_args)
        self._load_pretrained_model(**model_args)
        self.maybe_quantize(**model_args)

    def _build_base_training_args(self) -> dict:
        """Build the base training arguments dictionary from params.

        Returns:
            Dictionary of training arguments.
        """
        evaluation_strategy = (
            IntervalStrategy.STEPS if self.params.training.validation_ratio > 0 else IntervalStrategy.NO
        )
        return dict(
            output_dir=Path(self.workdir.train.cache),
            per_device_train_batch_size=self.params.training.batch_size,
            gradient_accumulation_steps=self.params.training.gradient_accumulation_steps,
            lr_scheduler_type=self.params.training.lr_scheduler,
            learning_rate=self.params.training.learning_rate,
            eval_strategy=evaluation_strategy,
            weight_decay=self.params.training.weight_decay,
            warmup_steps=self.params.training.warmup_steps,
            eval_steps=EVAL_STEPS,
            do_eval=self.params.training.validation_ratio > 0,
            disable_tqdm=True,  # The 🤗 progress bar doesn't play nice with our logging.
            **FIXED_RUNTIME_TRAINING_ARGS,
        )

    def _apply_eval_dataset_overrides(self, training_args: dict) -> None:
        """Apply eval dataset-specific overrides to training args.

        When an explicit eval dataset is present, evaluation is step-based and
        includes loss for metrics with accumulation disabled.

        Args:
            training_args: The training arguments dictionary to modify.
        """
        if self.eval_dataset is not None:
            training_args["eval_steps"] = self.params.training.validation_steps
            training_args["eval_strategy"] = "steps"
            training_args["do_eval"] = True
            training_args["include_for_metrics"] = ["loss"]
            training_args["eval_accumulation_steps"] = 1

    def _configure_dp_training(self, training_args: dict) -> DataCollatorForPrivateTokenClassification:
        """Configure differential privacy training settings.

        DP uses ``DataCollatorForPrivateTokenClassification`` and
        ``OpacusDPTrainer``. The HuggingFace ``max_grad_norm`` is set to zero
        because Opacus handles per-sample clipping. Gradient checkpointing is
        disabled because it is not compatible with the Opacus optimizer wrapping.

        Args:
            training_args: The training arguments dictionary to modify.

        Returns:
            The data collator for DP training.

        Raises:
            ParameterError: If required DP parameters are missing.
        """
        privacy = self.params.privacy
        if privacy is None:
            raise ParameterError("Privacy configuration is required for DP training")

        eps = privacy.epsilon
        logger.user.info(
            f"Differentially-private training is enabled, ε is set to {eps}",
        )

        data_collator = DataCollatorForPrivateTokenClassification(
            tokenizer=cast(PreTrainedTokenizer, self.tokenization.native)
        )

        training_args["remove_unused_columns"] = False  # required for DP data processing
        training_args["max_grad_norm"] = 0.0  # required for opacus optimizer
        _ = training_args.pop("gradient_checkpointing", None)

        if hasattr(self, "model"):
            model = self.model
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            if hasattr(model, "config"):
                model.config.use_cache = False

        if self.true_dataset_size is None or self.data_fraction is None:
            raise ParameterError(
                "When DP is enabled, extra info must be passed with data_fraction and true_dataset_size"
            )

        privacy_args = PrivacyArguments(
            target_epsilon=eps,
            target_delta=privacy.delta,
            per_sample_max_grad_norm=privacy.per_sample_max_grad_norm,
        )

        self.trainer_type = partial(  # ty: ignore[invalid-assignment] -- partial is assignable at runtime
            OpacusDPTrainer,
            privacy_args=privacy_args,
            true_dataset_size=self.true_dataset_size,
            data_fraction=self.data_fraction,
        )

        return data_collator

    def _configure_standard_training(self, training_args: dict) -> DataCollatorForTokenClassification:
        """Configure standard (non-DP) training settings.

        Args:
            training_args: The training arguments dictionary to modify.

        Returns:
            The data collator for standard training.
        """
        # The 🤗 classification collator pads both the inputs and labels.
        data_collator = training_args.pop(
            "data_collator",
            DataCollatorForTokenClassification(tokenizer=self.tokenization.native),
        )
        # Gradient checkpointing is not working correctly with Opacus optimizer as it 'wraps' the model in a GradSampleModule.
        # can fix this by enabling gradient checkpointing directly on the model but will defer that for now.
        training_args["gradient_checkpointing"] = True

        return data_collator

    def _create_trainer(
        self,
        training_args: TrainingArguments,
        data_collator: DataCollatorForTokenClassification | DataCollatorForPrivateTokenClassification,
    ) -> Trainer:
        """Create the trainer instance with the configured parameters.

        Args:
            training_args: The HuggingFace TrainingArguments instance.
            data_collator: The data collator to use.

        Returns:
            The configured Trainer instance.
        """
        factory = cast(Callable[..., Trainer], self.trainer_type)
        trainer = factory(
            model=self.model,
            processing_class=self.tokenization.native,
            args=training_args,
            train_dataset=self.training_examples.train,
            eval_dataset=self.training_examples.test,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=self.callbacks,
        )
        return trainer

    def _configure_trainer_callbacks(self, trainer: Trainer, training_args: dict) -> None:
        """Configure callbacks on the trainer.

        Args:
            trainer: The Trainer instance to configure.
            training_args: The training arguments dictionary (for inference_eval_kwargs).
        """
        # We have our own logger - remove the 🤗 printer callback.
        trainer.remove_callback(PrinterCallback)

        # Add our own callbacks. The progress bar should be used internally.
        trainer.add_callback(
            SafeSynthesizerWorkerCallback() if logger.isEnabledFor(logging.INFO) else ProgressBarCallback()
        )

        for callback in self.callbacks or []:
            trainer.add_callback(callback)

        if self.generation_eval:
            self._add_inference_eval_callback(trainer, training_args)

    def _add_inference_eval_callback(self, trainer: Trainer, training_args: dict) -> None:
        """Add the inference evaluation callback to the trainer.

        Args:
            trainer: The Trainer instance to configure.
            training_args: The training arguments dictionary containing inference_eval_kwargs.
        """
        if self.dataset_schema is None:
            raise ParameterError("dataset_schema must be set before configuring inference eval callback")
        if not hasattr(self, "training_prompt_encoding"):
            raise ParameterError("training prompt encoding must be set before configuring inference eval callback")

        logger.info(
            "👀 Heads up -> Generation eval is enabled ✅",
        )
        trainer.add_callback(
            InferenceEvalCallback(
                schema=self.dataset_schema,
                metadata=self.model_metadata,
                processor=create_processor(
                    config=self.params,
                    schema=self.dataset_schema,
                    metadata=self.model_metadata,
                    tokenizer=self.tokenization.native,
                ),
                tokenization=self.tokenization,
                prompt_encoding=self.training_prompt_encoding,
                num_prompts_per_batch=DEFAULT_VALID_RECORD_EVAL_BATCH_SIZE,
                **training_args["inference_eval_kwargs"],
            )
        )

    @traced_runtime("prepare_params")
    def prepare_params(self, **training_args: Any) -> None:
        """Prepare training parameters and create the trainer.

        Args:
            **training_args: Additional training arguments (currently unused but kept for API compatibility).
        """
        if not hasattr(self, "model"):
            self.load_model()

        training_args = self._build_base_training_args()
        self._apply_eval_dataset_overrides(training_args)

        if self.params.privacy is not None and self.params.privacy.dp_enabled:
            data_collator = self._configure_dp_training(training_args)
        else:
            data_collator = self._configure_standard_training(training_args)

        # Enable W&B logging if a WANDB run is initialized
        training_args["report_to"] = "wandb" if wandb.run is not None else "none"
        self.train_args = TrainingArguments(**training_args)
        self.trainer = self._create_trainer(self.train_args, data_collator)
        self._configure_trainer_callbacks(self.trainer, training_args)

    def _apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply action_executor preprocessing if available.

        Args:
            df: The DataFrame to preprocess.

        Returns:
            The preprocessed DataFrame.
        """
        if self.action_executor is None:
            return df

        logger.info("Applying data_config preprocessing")
        logger.debug(f"Before preprocess: {utils.debug_fmt(df)}")
        df = self.action_executor.preprocess(df)
        logger.debug(f"After preprocess: {utils.debug_fmt(df)}")
        return df

    def _process_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process time series data if enabled.

        Args:
            df: The DataFrame to process.

        Returns:
            The processed DataFrame (potentially sorted and with timestamp column).
        """
        is_time_series = bool(self.params.time_series.is_timeseries)

        if not is_time_series:
            return df

        logger.info("Processing time series data")
        df, self.params = process_timeseries_data(df, self.params)
        return df

    def _create_example_assembler(self, hf_dataset: Dataset) -> TrainingExampleAssembler:
        """Create the example assembler for training.

        Args:
            hf_dataset: The HuggingFace Dataset to use.

        Returns:
            The configured example assembler.
        """
        return TrainingExampleAssembler.from_data(
            config=self.params,
            metadata=self.model_metadata,
            dataset=hf_dataset,
            tokenizer=None,
            tokenization=self.tokenization,
            cache_file_path=self.training_output_dir,
            is_timeseries=self.params.time_series.is_timeseries,
            timestamp_column=self.params.time_series.timestamp_column,
        )

    @traced_user("log_dataset_statistics")
    def _log_dataset_statistics(self, assembler: TrainingExampleAssembler) -> None:
        """Log statistics about the training and validation datasets.

        Args:
            assembler: The example assembler containing dataset statistics.
        """
        for kind in ["train"] + ["validation"] * (assembler.num_records_validation > 0):
            n_recs = f"Number of unique {kind} records: {getattr(assembler, f'num_records_{kind}')}"
            if self.params.data.group_training_examples_by:
                n_recs += f", Number of unique groups: {getattr(assembler, f'num_groups_{kind}')}"

            extra = {
                "ctx": {
                    "render_table": True,
                    "tabular_data": {"num_records": getattr(assembler, f"num_records_{kind}")},
                    "title": n_recs,
                }
            }
            logger.user.info("", extra=extra)

    def prepare_training_data(self) -> None:
        """Validate, preprocess, and tokenize the training dataset.

        Validates groupby/orderby columns, resolves auto-config values,
        runs time-series preprocessing, and assembles tokenized training examples.
        Populates ``training_examples``, ``dataset_schema``,
        ``training_df``, and ``data_fraction``.

        Raises:
            DataError: If the training dataset is missing or malformed.
        """
        logger.info("Preparing training data.")

        if self.training_dataset is None:
            raise DataError("training_dataset must be set before preparing training data")

        training_df = self.training_dataset.to_pandas()
        if not isinstance(training_df, pd.DataFrame):
            raise DataError("Expected DataFrame from to_pandas(), got an iterator")

        self.params = AutoConfigResolver(training_df, self.params).resolve()

        # Process time series data (sort by timestamp, infer intervals, etc.)
        training_df = self._process_timeseries(training_df)

        training_df = self._apply_preprocessing(training_df)
        test_df = None

        hf_dataset = Dataset.from_pandas(training_df, preserve_index=False)
        self.training_df = training_df
        self.test_df = test_df

        assembler = self._create_example_assembler(hf_dataset)
        self.training_prompt_encoding = assembler.prompt_encoding
        schema_df = training_df.loc[:, list(assembler.prompt_columns)]
        self.dataset_schema = make_json_schema(schema_df)

        # This is a proxy for the number of training steps.
        num_records_to_sample = self.params.training.num_input_records_to_sample
        if not isinstance(num_records_to_sample, int):
            raise DataError(
                f"num_input_records_to_sample was not resolved to an int after AutoConfigResolver: {num_records_to_sample!r}"
            )
        self.data_fraction = num_records_to_sample / assembler.num_records_train

        self._log_dataset_statistics(assembler)

        self.training_examples = assembler.assemble_training_examples(data_fraction=self.data_fraction)

        logger.user.info(
            f"Number of training examples: {len(self.training_examples.train)}",
        )

        # This info is needed inside the trainer for DP
        # Number of records, if group_training_examples_by is None, or else number of groups
        self.true_dataset_size = len(assembler.training_dataset)

        if self.params.time_series.is_timeseries:
            self.model_metadata.initial_prefill = assembler._get_initial_prefill()  # ty: ignore[unresolved-attribute]

        self._propagate_max_tokens_per_example()
        self._propagate_max_records_per_group()

    def _propagate_max_tokens_per_example(self) -> None:
        """Copy the assembler's ``tokens_per_example.max`` onto ``ModelMetadata``.

        Sized from the actual tokenized examples rather than a char-count
        heuristic, this bound feeds ``ModelMetadata.generation_max_tokens_for``
        so ``SamplingParams.max_tokens`` caps decoding at output lengths
        the model was trained to produce. Tight caps let the engine retire
        EOS-failed sequences promptly (a known fine-tuned-LoRA failure
        mode) instead of letting them decode wasted tokens to the full
        context window. No-op when the stat is absent or unpopulated.
        """
        tokens_per_example = self.training_examples.stats.get("tokens_per_example")
        if tokens_per_example is not None and tokens_per_example.count > 0:
            self.model_metadata.max_tokens_per_example = int(math.ceil(tokens_per_example.max))

    def _propagate_max_records_per_group(self) -> None:
        """Copy the grouped assembler's ``records_per_group.max`` onto ``ModelMetadata``.

        Only grouped runs populate this stat. It becomes the default bound on
        per-group record repetition in structured generation: the grammar forces
        the group-closing delimiter after at most the largest trained group, so
        completions terminate and parse instead of decoding to the token cap.
        No-op when the stat is absent (non-grouped) or unpopulated.
        """
        records_per_group = self.training_examples.stats.get("records_per_group")
        if records_per_group is not None and records_per_group.count > 0:
            self.model_metadata.max_records_per_group = int(math.ceil(records_per_group.max))

    @utils.time_function
    def train(self, **training_args: Any) -> None:
        """Run the full training pipeline and populate ``results``.

        Sequentially calls ``prepare_training_data``,
        ``prepare_params``, trains the model, and saves artifacts.
        """
        training_start = time.monotonic()
        self.prepare_training_data()
        self.prepare_params(**training_args)
        self.trainer.train()
        training_time_sec = time.monotonic() - training_start

        # Capture log_history before teardown deletes the trainer.
        log_history = self.trainer.state.log_history
        is_complete = "training_incomplete" not in sum([list(d.keys()) for d in log_history], [])

        self.save_model()
        self.teardown()

        self.results = NSSTrainerResult(
            training_df=self.training_df,
            df_ml_utility_holdout=self.test_df,
            config=self.params,
            training_complete=is_complete,
            log_history=log_history,
            adapter_path=self.model_metadata.adapter_path,
            elapsed_time=training_time_sec,
        )

    def save_model(self) -> None:
        """Save the fine-tuning adapter and related artifacts under ``self.workdir``.

        Does not release resources -- callers should invoke :meth:`teardown`
        explicitly when they are done with the backend (the abstract contract
        on :class:`TrainingBackend` keeps save and teardown as separate
        lifecycle events).
        """
        if self.dataset_schema is None:
            raise ParameterError("dataset_schema must be set before saving model")

        adapter_dir = self.workdir.train.adapter
        if not isinstance(adapter_dir, BoundDir):
            raise TypeError(f"Expected BoundDir, got {type(adapter_dir)}")
        self.workdir.ensure_directories()
        logger.user.info(f"Saving LoRA adapter to {adapter_dir}")
        with redirect_stdout(io.StringIO()) as stdout:
            self.model.save_pretrained(str(adapter_dir))
        logger.runtime.debug(stdout.getvalue())
        logger.user.info(f"Saving model metadata to {adapter_dir.metadata}")
        self.model_metadata.save_metadata()
        logger.user.info(f"Saving dataset schema to {adapter_dir.schema}")
        write_json(self.dataset_schema, adapter_dir.schema, indent=4)
        logger.user.info(f"Saving model parameters to {self.workdir.train.config}")
        write_json(
            self.params.model_dump(mode="json"),
            path=self.workdir.train.config,
            indent=4,
        )

    def teardown(self) -> None:
        """Release GPU memory, distributed resources, and trainer state. Idempotent -- safe to call multiple times."""
        if getattr(self, "_torn_down", False):
            return
        self._torn_down = True

        import torch.distributed as dist

        try:
            if hasattr(self, "trainer"):
                del self.trainer
        except Exception:  # pragma: no cover -- defensive; del on an instance attr does not raise
            logger.warning("trainer cleanup failed during teardown", exc_info=True)

        try:
            if hasattr(self, "model"):
                del self.model
        except Exception:  # pragma: no cover -- defensive; del on an instance attr does not raise
            logger.warning("model cleanup failed during teardown", exc_info=True)

        try:
            cleanup_memory()
        except Exception:
            logger.warning("cleanup_memory failed during teardown", exc_info=True)

        if TYPE_CHECKING:
            assert hasattr(dist, "destroy_process_group")
            assert hasattr(dist, "is_initialized")
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            logger.warning("destroy_process_group failed during teardown", exc_info=True)

    def __str__(self):
        f = f"HuggingFaceBackend(pretrained_model={self.params.training.pretrained_model}, params={self.params})"
        return f

    def info(self) -> None:
        """Print a summary of key trainer attributes to stdout."""
        fields = [
            "params",
            "training_output_dir",
            "save_path",
            "artifact_path",
        ]
        info = {field: getattr(self, field) for field in fields}
        msg = "Trainer Information"
        msg += "\n" + "-" * len(msg)
        msg += "\n" + "\n".join([f"{field}: {value}" for field, value in info.items()])
        msg += "\n" + "-" * len(msg)

        logger.info(msg)


def preprocess_logits_for_metrics(
    logits: tuple[torch.Tensor, ...], labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce logits to argmax predictions to avoid OOM during evaluation.

    The default Trainer stores full logit tensors across evaluation batches,
    which can exhaust GPU memory on large datasets. This callback replaces
    them with predicted token IDs immediately after the forward pass.

    See: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13

    Args:
        logits: Tuple of logits tensors from the model output.
        labels: Ground truth labels tensor.

    Returns:
        Tuple of ``(predicted_token_ids, labels)``.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


def compute_metrics(eval_preds: EvalPrediction) -> dict[str, float]:
    """Compute evaluation metrics from forward-pass losses.

    Metrics returned:

    - mean cross-entropy loss (``eval_loss``) -- average of per-batch
      losses collected during the evaluation loop.

    The per-batch losses are pre-computed during the forward pass
    (via ``include_for_metrics``).

    Args:
        eval_preds: Evaluation predictions object whose ``losses`` field
            contains per-batch losses collected during the eval loop.

    Returns:
        Dictionary mapping metric names to values.
    """
    # include_for_metrics has "loss", so the loss is already computed in the forward pass
    losses = eval_preds.losses if eval_preds.losses is not None else []
    metrics = {"eval_loss": np.mean(losses)}

    # Log the evaluation loss using the same style as callbacks.py
    if metrics["eval_loss"] is not None:
        logger.user.info(
            f"Evaluation loss: {metrics['eval_loss']:.4f}",
            extra={
                "ctx": {"eval_loss": round(metrics["eval_loss"], 4)},
            },
        )

    return metrics
