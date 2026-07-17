# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract generator backend."""

from __future__ import annotations

import abc
import time
from collections.abc import Callable

from .. import utils
from ..cli.artifact_structure import Workdir
from ..config import SafeSynthesizerParameters
from ..defaults import FIXED_RUNTIME_GENERATE_ARGS
from ..llm.metadata import ModelMetadata
from ..observability import get_logger, heartbeat
from .batch import Batch
from .processors import Processor, TabularDataProcessor
from .results import GenerateJobResults, GenerationBatches, GenerationStatus

logger = get_logger(__name__)


class GeneratorBackend(metaclass=abc.ABCMeta):
    """Abstract base class for generation backends.

    Lifecycle: ``initialize`` -> ``generate`` [-> ``generate`` ...] ->
    ``teardown``.

    ``generate`` is a concrete template method that owns the batch loop,
    stopping conditions, and result aggregation -- machinery that is
    identical across engines because only decoded text and token counts
    flow out of the model call. Subclasses provide the engine-specific
    pieces by implementing ``initialize``, ``prepare_params``,
    ``_generate_batch``, ``_get_prompt_token_count``, and ``teardown``.
    A subclass with a fundamentally different loop (e.g. the grouped
    time-series flow) may override ``generate`` wholesale.

    ``teardown`` must be idempotent and safe to call multiple times.
    Callers should use ``try/finally`` to guarantee ``teardown`` runs
    even if ``generate`` raises.  Each cleanup step should be isolated
    so one failure doesn't prevent the next from running.  The
    ``_torn_down`` guard flag pattern is recommended for teardown
    implementations.
    """

    gen_method: Callable | None = None
    """Callable used internally for LLM generation."""

    gen_results: GenerateJobResults
    """Results from the most recent generation run."""

    config: SafeSynthesizerParameters
    """Pipeline configuration."""

    model_metadata: ModelMetadata
    """Metadata for the fine-tuned model (prompt template, sequence length, adapter path, etc.)."""

    remote: bool
    """Whether the backend calls a remote inference endpoint."""

    elapsed_time: float
    """Wall-clock duration of the last generation run in seconds."""

    workdir: Workdir
    """Working directory containing model artifacts."""

    prompt: str
    """Templated generation prompt sent to the model for every record."""

    columns: list[str]
    """Schema column names, in order, used to assemble the result frame."""

    processor: Processor
    """Parser that turns raw model text into validated records."""

    use_detailed_logs: bool = False
    """Whether to emit verbose per-record error messages (may leak data)."""

    @abc.abstractmethod
    def initialize(self) -> None:
        """Acquire the resources the backend needs to serve generations.

        Called once before the first ``generate()`` invocation. What this
        entails is backend-specific: a local engine (e.g. vLLM) allocates GPU
        memory, instantiates the engine, and loads LoRA adapters, while a
        remote backend opens an HTTP client and connection pool.

        After this method returns, the backend must be ready to accept
        ``prepare_params()`` and ``generate()`` calls.
        """

    @abc.abstractmethod
    def prepare_params(self, **kwargs) -> None:
        """Translate caller-supplied sampling parameters into a backend-native form.

        Resolves, validates, and transforms high-level generation
        parameters (temperature, top-p, max tokens, structured-output
        constraints, etc.) into the format expected by the underlying
        inference engine.  The result is stored internally so that
        subsequent ``generate()`` calls use these settings.

        Must be called after ``initialize()`` and before ``generate()``.

        Args:
            **kwargs: Sampling parameters such as ``temperature``,
                ``top_p``, ``max_new_tokens``, ``repetition_penalty``,
                and backend-specific options.
        """

    @abc.abstractmethod
    def _get_prompt_token_count(self) -> int:
        """Return the templated prompt's tokenized length.

        Used to size the per-sample ``max_tokens`` budget so the prompt
        plus completion stays within the model's context window. Return
        ``0`` to disable the prompt-length clamp when no tokenizer is
        available.
        """

    @abc.abstractmethod
    def _generate_batch(
        self,
        num_prompts_per_batch: int,
        batch: Batch,
        **sampling_kwargs,
    ) -> Batch:
        """Run the engine on one batch of prompts and populate ``batch``.

        Implementations issue ``num_prompts_per_batch`` generations of
        ``self.prompt``, then for each completion record its finish
        reason and call ``batch.process(idx, text, completion_tokens=...)``
        with the decoded text and token count. The engine-native response
        objects must not escape this method -- only text and counts flow
        downstream.

        Args:
            num_prompts_per_batch: Number of prompts to run this batch.
            batch: Fresh batch carrying the configured processor.

        Returns:
            The same ``batch``, populated with parsed records and stats.
        """

    def generate(
        self,
        data_actions_fn: utils.DataActionsFn | None = None,
    ) -> GenerateJobResults:
        """Run the batch generation loop and return aggregated results.

        Repeatedly prompts the model via ``_generate_batch`` and processes
        each batch through the configured
        [`Processor`][nemo_safe_synthesizer.generation.processors.Processor]
        until the target record count is reached or a stopping condition
        fires (e.g. too many consecutive invalid batches). Progress and
        error statistics are logged after each batch.

        Non-tabular processors need BOS/EOS delimiters in the raw text, so
        generation keeps special tokens for those processors and strips
        them only for ``TabularDataProcessor``. Native EOS stopping
        remains enabled through ``ignore_eos=False``.

        Args:
            data_actions_fn: Optional post-processing / validation function
                applied to each batch of generated records. Typically
                reverses training-time preprocessing and enforces
                user-specified data constraints.

        Returns:
            Results containing the generated DataFrame, validity
            statistics, and timing information.
        """
        generation_start = time.monotonic()

        need_special_token_outputs = not isinstance(self.processor, TabularDataProcessor)
        sampling_kwargs = dict(
            temperature=self.config.generation.temperature,
            repetition_penalty=self.config.generation.repetition_penalty,
            top_p=self.config.generation.top_p,
            top_k=FIXED_RUNTIME_GENERATE_ARGS["top_k"],
            min_p=FIXED_RUNTIME_GENERATE_ARGS["min_p"],
            max_tokens=self.model_metadata.generation_max_tokens_for(self._get_prompt_token_count()),
            skip_special_tokens=not need_special_token_outputs,
            include_stop_str_in_output=need_special_token_outputs,
            ignore_eos=False,
        )

        self.prepare_params(**sampling_kwargs)

        # The batches object collects batches and keeps track of the stopping condition.
        batches = GenerationBatches(
            target_num_records=self.config.generation.num_records,
            invalid_fraction_threshold=self.config.generation.invalid_fraction_threshold,
            patience=self.config.generation.patience,
            data_actions_fn=data_actions_fn,
        )

        with heartbeat(
            "Generation",
            logger_name=__name__,
            target_records=self.config.generation.num_records,
            progress_note=("Long stretches with no new records are normal."),
        ):
            while batches.num_valid_records < self.config.generation.num_records:
                # Generate a batch from prompts and process the responses.
                num_prompts = batches.get_next_num_prompts()
                start_time = time.perf_counter()
                batch: Batch = self._generate_batch(
                    num_prompts_per_batch=num_prompts,
                    batch=Batch(processor=self.processor),
                    **sampling_kwargs,
                )
                duration = time.perf_counter() - start_time
                batches.add_batch(batch)

                # Log generation summary and progress.
                batch.log_summary(detailed_errors=self.use_detailed_logs)
                self._log_batch_timing_and_progress(batch=batch, duration=duration, batches=batches)
                # Check if the generation job should stop.
                if batches.status in [
                    GenerationStatus.STOP_NO_RECORDS,
                    GenerationStatus.STOP_METRIC_REACHED,
                ]:
                    break

        batches.job_complete()
        batches.log_status()

        max_num_records = (
            self.config.generation.num_records
            if self.config.data.group_training_examples_by is None and batches.status == GenerationStatus.COMPLETE
            else None
        )

        self.elapsed_time = time.monotonic() - generation_start
        self.gen_results = GenerateJobResults.from_batches(
            batches=batches,
            columns=self.columns,
            max_num_records=max_num_records,
            elapsed_time=self.elapsed_time,
        )

        return self.gen_results

    def _log_batch_timing_and_progress(
        self,
        batch: Batch,
        duration: float,
        batches: GenerationBatches,
    ) -> None:
        """Log batch timing and progress as a structured Rich table.

        Emits structured data via ``logger.user.info`` that is rendered
        as a Rich ASCII table on the console and as key/value pairs in
        JSON logs.
        """
        records_per_second = 0 if duration == 0 else batch.num_valid_records / duration

        # Build structured data - processor renders as table for console
        progress_data: dict[str, int | float] = {
            "records_per_second": round(records_per_second, 2),
            "duration_seconds": round(duration, 2),
            "valid_records_generated": batches.num_valid_records,
            "target_records": self.config.generation.num_records,
            "progress_fraction": round(batches.num_valid_records / self.config.generation.num_records, 4),
        }
        if batch.total_completion_tokens > 0 and duration > 0:
            progress_data["tokens_per_second"] = round(batch.total_completion_tokens / duration, 1)
            progress_data["valid_tokens_per_second"] = round(batch.total_valid_record_tokens / duration, 1)

        # Pass structured data - processor renders for console, JSON keeps as-is
        logger.user.info(
            "",
            extra={
                "ctx": {
                    "render_table": True,
                    "tabular_data": progress_data,
                    "title": "Batch Progress",
                }
            },
        )

    @abc.abstractmethod
    def teardown(self) -> None:
        """Release all resources held by this backend.

        Frees GPU memory, destroys distributed process groups, and
        cleans up any temporary state.  Must be idempotent -- safe to
        call multiple times.  Implementations should use the
        ``_torn_down`` guard flag and isolate each cleanup step so one
        failure doesn't prevent subsequent cleanup.

        Callers should wrap ``generate()`` in ``try/finally`` to
        guarantee this runs even when generation raises.
        """
