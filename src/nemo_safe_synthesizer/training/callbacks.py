# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Optional

from tqdm.auto import tqdm
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from ..defaults import (
    DEFAULT_SAMPLING_PARAMETERS,
    NUM_EVAL_BATCHES_GROUPED,
    NUM_EVAL_BATCHES_TABULAR,
)
from ..generation.batch import Batch
from ..generation.processors import (
    Processor,
    TabularDataProcessor,
)
from ..generation.results import (
    GenerationBatches,
    GenerationStatus,
)
from ..llm.metadata import ModelMetadata
from ..llm.utils import optimize_for_inference
from ..observability import get_logger
from ..utils import create_schema_prompt

logger = get_logger(__name__)


class InferenceEvalCallback(TrainerCallback):
    """🤗 Trainer callback that performs inference-based evaluation during training.

    This callback generates records using the current model and validates them against a schema.
    Empirically, the fraction of invalid records generated is a good indicator of the model's
    performance. The callback can stop training if the fraction of invalid records satisfies
    the stopping criteria specified by `invalid_fraction_threshold` and `patience`.

    Args:
        schema: Schema to validate the generated records against.
        num_prompts_per_batch: Number of prompts per batch.
        num_batches: Number of batches to generate.
        invalid_fraction_threshold: The fraction of invalid records that will stop generation after the `patience` limit is reached.
        patience: Number of consecutive generations where the `invalid_fraction_threshold` is reached before stopping generation.
        generate_kwargs: Keyword arguments to pass to the model's generate method.
    """

    def __init__(
        self,
        schema: dict,
        metadata: ModelMetadata,
        processor: Processor,
        num_prompts_per_batch: int = 16,
        num_batches: Optional[int] = None,
        patience: int = 3,
        invalid_fraction_threshold: float = 0.8,
        generate_kwargs: dict | None = None,
    ):
        self.schema = schema
        self.metadata = metadata
        self.templated_prompt = create_schema_prompt(
            schema["properties"].keys(),
            instruction=self.metadata.instruction,
            prompt_template=self.metadata.prompt_config.template,
        )
        self.num_prompts_per_batch = num_prompts_per_batch

        self.is_tabular_processor = isinstance(processor, TabularDataProcessor)
        self.num_batches = num_batches or (
            NUM_EVAL_BATCHES_TABULAR if self.is_tabular_processor else NUM_EVAL_BATCHES_GROUPED
        )

        self.generation = GenerationBatches(
            invalid_fraction_threshold=invalid_fraction_threshold,
            patience=patience,
        )
        self.processor = processor

        kws = generate_kwargs or {}
        self.generate_kwargs = {
            "temperature": kws.get("temperature", DEFAULT_SAMPLING_PARAMETERS["temperature"]),
            "top_p": kws.get("top_p", DEFAULT_SAMPLING_PARAMETERS["top_p"]),
            "top_k": kws.get("top_k", DEFAULT_SAMPLING_PARAMETERS["top_k"]),
            "repetition_penalty": kws.get("repetition_penalty", DEFAULT_SAMPLING_PARAMETERS["repetition_penalty"]),
        }

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not state.is_world_process_zero:
            return

        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]

        with optimize_for_inference(model):
            was_stopped = False

            logger.info(
                f"🔮 Starting inference-based evaluation with the '{self.processor.name}'",
            )

            for _ in range(self.num_batches):
                prompt_tokens = tokenizer(
                    [self.templated_prompt] * self.num_prompts_per_batch,
                    return_tensors="pt",
                )
                input_ids = prompt_tokens["input_ids"].to(model.device)
                attention_mask = prompt_tokens["attention_mask"].to(model.device)

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=tokenizer.model_max_length - len(input_ids[0]),
                    do_sample=True,
                    use_cache=True,
                    **self.generate_kwargs,
                )
                decoded = tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=self.is_tabular_processor,
                )

                start_time = time.perf_counter()
                batch = Batch(processor=self.processor)
                for idx, text in enumerate(decoded):
                    batch.process(idx, text)
                duration = time.perf_counter() - start_time
                self.generation.add_batch(batch)

                batch.log_summary()
                duration_string = f"{duration:.1f} seconds" if duration < 120 else f"{duration / 60:.1f} minutes"
                logger.info(f"Generation time: {duration_string}")

                if self.generation.status != GenerationStatus.IN_PROGRESS:
                    was_stopped = True
                    break

            if was_stopped:
                control.should_training_stop = True
                if self.generation.status == GenerationStatus.STOP_NO_RECORDS:
                    logger.error(
                        "🛑 Stopping generation prematurely. No records were generated. "
                        "Please consider adjusting the sampling parameters.",
                    )
                    state.log_history.append({"training_incomplete": "no_records"})
                elif self.generation.status == GenerationStatus.STOP_METRIC_REACHED:
                    logger.error(
                        "🛑 Stopping generation prematurely. The stopping "
                        "condition was reached with a running average invalid "
                        f"fraction of {self.generation.stop_condition.last_value:.2%}",
                    )
                    state.log_history.append({"training_incomplete": "stopping_condition_reached"})


class ProgressBarCallback(TrainerCallback):
    """A `TrainerCallback` that displays the progress of training or evaluation.

    Note: This callback can only be used during development.
    """

    def __init__(self):
        self.training_bar = None
        self.prediction_bar = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(
                total=state.max_steps,
                dynamic_ncols=True,
                desc="Training in progress",
            )
        self.current_step = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_prediction_step(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, eval_dataloader=None, **kwargs
    ):
        if not state.is_world_process_zero:
            return

        if not hasattr(eval_dataloader, "__len__"):
            return

        if self.prediction_bar is None:
            self.prediction_bar = tqdm(
                total=len(eval_dataloader),
                leave=self.training_bar is None,
                dynamic_ncols=True,
                desc="Evaluation in progress",
            )
        self.prediction_bar.update(1)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not isinstance(logs, dict):
            return

        if state.is_world_process_zero and self.training_bar is not None:
            # For the purpose of state.global_step, one step is a step with backprop,
            # which is conducted every args.gradient_accumulation_steps. Since the
            # value of the latter is opaque to the user, we show steps as the (approximate)
            # number of actual update steps. If a step limit is set, this number should however
            # never exceed that limit, as otherwise this might cause confusion for the user.
            logs["step"] = state.global_step * args.gradient_accumulation_steps

            _ = logs.pop("total_flos", None)
            if "loss" in logs:
                self.training_bar.set_description(f"Training in progress [loss = {logs['loss']: .4f}]")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.close()
            self.training_bar = None


class SafeSynthesizerWorkerCallback(TrainerCallback):
    """Trainer callback to log training information to the worker logs."""

    _start_ts: float
    _last_log_ts: float
    _log_interval: float
    _last_log_global_step: int

    def __init__(self, log_interval: float = 60.0):
        self._log_interval = log_interval
        self._last_log_ts = time.monotonic()
        self._last_log_global_step = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._start_ts = time.monotonic()

    def _checked_log_if(self, cond: bool, state: TrainerState, control: TrainerControl) -> Optional[TrainerControl]:
        # We need to keep track of the last global_step that was used when logging,
        # as triggering log handling twice for the same step leads to div by 0.
        # https://github.com/huggingface/transformers/blob/v4.29.0/src/transformers/trainer.py#L2279
        if state.is_local_process_zero and state.global_step > self._last_log_global_step and cond:
            control.should_log = True
            return control

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Ensure a log is emitted at the end of every epoch, regardless of
        # when the last log was emitted.
        return self._checked_log_if(True, state, control)

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Ensure a log is emitted at least on the given update interval. There
        # is nothing particularly interesting about substeps, it is just the
        # most fine-grained callback provided by the interface.
        # We leave it to `on_log` below to actually reset the last log timestamp.
        return self._checked_log_if(time.monotonic() - self._last_log_ts >= self._log_interval, state, control)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if not isinstance(logs, dict):
            return

        if state.is_local_process_zero:
            logs.pop("total_flos", None)

            # For the purpose of state.global_step, one step is a step with backprop,
            # which is conducted every args.gradient_accumulation_steps. Since the
            # value of the latter is opaque to the user, we show steps as the (approximate)
            # number of actual update steps. If a step limit is set, this number should however
            # never exceed that limit, as otherwise this might cause confusion for the user.
            user_step = state.global_step * args.gradient_accumulation_steps
            total_steps = state.max_steps * args.gradient_accumulation_steps

            if (which_loss := "eval_loss" if "eval_loss" in logs else "loss") in logs:
                logs["step"] = user_step
                complete_frac = user_step / total_steps
                log_to_user = {}
                log_to_user["progress"] = complete_frac
                log_to_user["epoch"] = state.epoch
                log_to_user["step"] = user_step
                log_to_user[which_loss] = round(logs[which_loss], 4)

                logger.runtime.info(
                    "",
                    extra={
                        "ctx": {
                            "render_table": True,
                            "tabular_data": log_to_user,
                            "title": "Training Progress",
                        }
                    },
                )

            self._last_log_ts = time.monotonic()
            self._last_log_global_step = state.global_step
