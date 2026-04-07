# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation result containers and multi-batch accumulator."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Self

import pandas as pd

from ..data_processing.actions.utils import (
    MetadataColumns,
)
from ..data_processing.stats import RunningStatistics
from ..defaults import (
    EPS,
    MAX_NUM_PROMPTS_PER_BATCH,
)
from ..errors import GenerationError
from ..generation.batch import Batch
from ..generation.stopping import (
    GenerationStopCondition,
)
from ..generation.utils import GenerationStatus
from ..observability import get_logger
from ..utils import DataActionsFn

NUM_PROMPT_BUFFER = 10

logger = get_logger(__name__)


@dataclass
class GenerateJobResults:
    """Results of a complete generation job.

    Encapsulates the generated DataFrame along with validity statistics,
    prompt counts, and timing information.  Built from a
    [`GenerationBatches`][nemo_safe_synthesizer.generation.results.GenerationBatches]
    accumulator via ``from_batches``.
    """

    df: pd.DataFrame
    """DataFrame containing the generated records."""

    status: GenerationStatus
    """Overall generation status derived from the processed batches."""

    num_valid_records: int
    """Total number of records that passed validation."""

    num_invalid_records: int
    """Total number of records that failed validation."""

    num_prompts: int
    """Total number of prompts processed during generation."""

    valid_record_fraction: float
    """Fraction of valid records among all generated records."""

    batch_valid_record_fractions: list[float]
    """Per-batch valid record fractions, in batch order."""

    elapsed_time: float | None = None
    """Wall-clock generation duration in seconds, or ``None`` if not yet set."""

    @classmethod
    def from_batches(
        cls, batches: GenerationBatches, max_num_records: int | None, columns: list[str], elapsed_time: float
    ) -> Self:
        """Build results from a completed :class:`GenerationBatches` accumulator.

        Args:
            batches: Accumulated generation batches.
            max_num_records: If set, truncate the output DataFrame to
                this many rows.
            columns: Column names to select from the generated records.
            elapsed_time: Wall-clock generation duration in seconds.

        Returns:
            Populated results instance.
        """
        df = batches.to_dataframe(columns, max_num_records)
        status = batches.status
        num_valid_records = batches.num_valid_records
        num_invalid_records = batches.num_invalid_records
        num_prompts = batches.num_prompts
        valid_record_fraction = (
            batches.num_valid_records / (batches.num_valid_records + batches.num_invalid_records)
            if (batches.num_valid_records + batches.num_invalid_records) > 0
            else 0.0
        )
        batch_valid_record_fractions = [batch.valid_record_fraction for batch in batches._batches]
        return cls(
            df=df,
            status=status,
            num_valid_records=num_valid_records,
            num_invalid_records=num_invalid_records,
            num_prompts=num_prompts,
            valid_record_fraction=valid_record_fraction,
            batch_valid_record_fractions=batch_valid_record_fractions,
            elapsed_time=elapsed_time,
        )


class GenerationBatches:
    """Accumulator that tracks batches during the generation phase.

    Manages the stopping condition, running statistics, and optional
    post-processing via ``data_actions_fn``.

    Args:
        target_num_records: Target number of valid records to generate.
        batches: Pre-existing batches to seed the accumulator with.
        max_num_prompts_per_batch: Maximum prompts per LLM generation
            call.
        invalid_fraction_threshold: Fraction of invalid records that
            triggers stopping after ``patience`` consecutive batches.
        patience: Consecutive batch count before the threshold triggers
            a stop.
        data_actions_fn: Optional function that post-processes and
            validates records from each batch.

    Attributes:
        status: Current generation status.
        running_stopping_metric: Exponential running average of the
            invalid-record fraction.
        stop_condition: The patience-based stopping condition, or
            ``None`` if thresholds were not provided.
    """

    def __init__(
        self,
        target_num_records: int | None = None,
        batches: list[Batch] | None = None,
        max_num_prompts_per_batch: int = MAX_NUM_PROMPTS_PER_BATCH,
        invalid_fraction_threshold: float | None = None,
        patience: int | None = None,
        data_actions_fn: DataActionsFn | None = None,
    ):
        self._batches = batches or []
        self._start_time = time.perf_counter()
        self.target_num_records = target_num_records
        self.max_num_prompts_per_batch = max_num_prompts_per_batch
        self.status = GenerationStatus.IN_PROGRESS
        self.running_stopping_metric = RunningStatistics()

        self.stop_condition = None
        if invalid_fraction_threshold is not None or patience is not None:
            if invalid_fraction_threshold is None or patience is None:
                raise ValueError("Invalid fraction threshold and patience must be provided together.")
            self.stop_condition = GenerationStopCondition(
                invalid_fraction_threshold=invalid_fraction_threshold,
                patience=patience,
            )

        self.data_actions_fn = data_actions_fn
        self._batches_df: pd.DataFrame | None = None

    def _apply_data_actions_fn(self, batch: Batch) -> None:
        """Post-process and validate a batch via ``data_actions_fn``.

        Converts the batch's record lists into a DataFrame, runs the
        configured post-processing (reversing training-time preprocessing)
        and user-specified validation rules, then maps the results back
        onto the batch's ``valid_records`` and ``invalid_records``.

        Each record is tagged with a temporary ID so that validated rows
        can be mapped back to their originating response.

        Args:
            batch: The batch whose records will be post-processed and
                filtered in place.
        """
        if self.data_actions_fn is None:
            return

        # Add a special key to each record, giving it a unique
        # identifier. This will make it easier to map original records
        # to the validated records, and ensure that we only allow through
        # the records that passed validation.
        record_id_key = MetadataColumns.INDEX.value
        record_id = 0
        for response in batch._responses:
            for record in response.valid_records:
                record[record_id_key] = record_id
                record_id += 1

        batch_df = batch.to_dataframe()
        if batch_df is None:
            return

        # In the first batch iteration, initialize `_batches_df`.
        if self._batches_df is None:
            self._batches_df = pd.DataFrame(columns=batch_df.columns)

        valid_df, rejected_df = self.data_actions_fn(batch_df, self._batches_df)

        # Incrementally add onto the `_batches_df` with each batch of valid records.
        self._batches_df = pd.concat([self._batches_df, valid_df])

        # map the records to their transformed records
        id_to_valid_records = dict(zip(valid_df[record_id_key], valid_df.to_dict("records")))
        id_to_rejected_records = dict(zip(rejected_df[record_id_key], rejected_df.to_dict("records")))
        for response in batch._responses:
            new_valid_records = []
            new_rejected_records = []
            for record in response.valid_records:
                if new_record := id_to_valid_records.get(record[record_id_key]):
                    del new_record[record_id_key]
                    new_valid_records.append(new_record)
                elif new_record := id_to_rejected_records.get(record[record_id_key]):
                    del new_record[record_id_key]
                    new_rejected_records.append(new_record)
                else:
                    raise AssertionError("Every record in response should map to either a valid or rejected row.")

            response.valid_records = new_valid_records
            for rejected_record in new_rejected_records:
                error = rejected_record_to_error(rejected_record)
                response.invalid_records.append(str(rejected_record))
                response.errors.append(error)

    @property
    def num_batches(self) -> int:
        """The number of batches in the generation job."""
        return len(self._batches)

    @property
    def num_prompts(self) -> int:
        """The total number of prompts processed in the generation job."""
        return sum([batch.num_prompts for batch in self._batches])

    @property
    def num_invalid_records(self) -> int:
        """The total number of invalid records generated in the generation job."""
        return sum([batch.num_invalid_records for batch in self._batches])

    @property
    def num_valid_records(self) -> int:
        """The total number of valid records generated in the generation job."""
        return sum([batch.num_valid_records for batch in self._batches])

    def add_batch(self, batch: Batch) -> None:
        """Add a batch and update the generation status.

        Stopping rules:

        * The very first batch producing zero valid records always
          triggers ``STOP_NO_RECORDS``.
        * When a ``stop_condition`` is configured, subsequent batches
          with zero valid records are tolerated until the patience-based
          threshold is reached.
        * Without a ``stop_condition``, any batch with zero valid
          records triggers ``STOP_NO_RECORDS``.

        Args:
            batch: The completed batch to add.
        """
        # TODO: Move application of the data_actions_fn deeper in the generation process
        self._apply_data_actions_fn(batch)
        self.running_stopping_metric.update(batch.stopping_metric)
        if self.stop_condition is None:
            if batch.num_valid_records == 0:
                self.status = GenerationStatus.STOP_NO_RECORDS
        else:
            if batch.num_valid_records == 0 and self.num_batches == 0:
                self.status = GenerationStatus.STOP_NO_RECORDS
            elif self.stop_condition.has_been_reached(self.running_stopping_metric.mean):
                self.status = GenerationStatus.STOP_METRIC_REACHED

        self._batches.append(batch)

    def get_next_num_prompts(self) -> int:
        """Return an estimate of the optimal number of prompts to process in the next batch."""
        num_prompts = self.max_num_prompts_per_batch

        if self.target_num_records is None:
            return num_prompts

        if self.num_valid_records > 0:
            num_records_remaining = self.target_num_records - self.num_valid_records
            valid_records_per_prompt = 0 if self.num_prompts == 0 else self.num_valid_records / self.num_prompts
            num_prompts_needed = round(num_records_remaining / (valid_records_per_prompt + EPS))
            num_prompts = min(num_prompts, num_prompts_needed + NUM_PROMPT_BUFFER)

        return num_prompts

    def job_complete(self) -> None:
        """Update the generation job status to a finished state and log the results."""
        self.duration = time.perf_counter() - self._start_time
        if self.status == GenerationStatus.IN_PROGRESS:
            if self.target_num_records is None or self.num_valid_records >= self.target_num_records:
                self.status = GenerationStatus.COMPLETE
            else:
                self.status = GenerationStatus.INCOMPLETE

    def log_status(self) -> None:
        """Log the current status of the generation process."""
        if self.status == GenerationStatus.COMPLETE:
            logger.info("🎉 Generation complete 🎉")
        elif self.status == GenerationStatus.IN_PROGRESS:
            logger.info(
                f"Generation in progress. {self.num_valid_records} out of {self.target_num_records} records generated.",
            )
        elif self.status == GenerationStatus.INCOMPLETE:
            logger.warning(
                f"😬 Generation incomplete -> {self.num_valid_records} out of "
                f"{self.target_num_records} records were generated.",
            )
        elif self.status == GenerationStatus.STOP_NO_RECORDS:
            logger.error(
                "🛑 Stopping generation prematurely. No valid records were generated due to model underfitting."
                " Please consider increasing the 'num_input_records_to_sample' parameter.",
            )
            raise GenerationError(
                "Generation stopped prematurely due to no valid records."
                " Please consider increasing the 'num_input_records_to_sample' parameter."
            )
        elif self.status == GenerationStatus.STOP_METRIC_REACHED:
            stop = self.stop_condition
            if stop is None:
                raise GenerationError("Generation stopped: metric reached but stop_condition is None.")
            stop_val: float | int | None = stop.last_value
            frac_str = f"{stop_val:.2%}" if stop_val is not None else "?"
            logger.error(
                "🛑 Stopping generation prematurely. The stopping "
                "condition was reached with a running average invalid "
                f"fraction of {frac_str}."
                " Please consider increasing the 'num_input_records_to_sample' parameter.",
            )
            raise GenerationError(
                "Generation stopped prematurely because "
                f"the average fraction of invalid records was higher than {frac_str}."
                " Please consider increasing the 'num_input_records_to_sample' parameter.",
            )

    def to_dataframe(self, columns: list[str], max_num_records: int | None = None) -> pd.DataFrame:
        """Combine valid records from all batches into a single DataFrame.

        Args:
            columns: Column names to include in the output.
            max_num_records: If set, truncate to this many rows.

        Returns:
            DataFrame of valid records, or an empty DataFrame if none
            were generated.
        """
        # return an empty DataFrame when any generated batch has 0 valid records.
        if self.num_valid_records == 0:
            return pd.DataFrame()
        df = pd.concat([batch.to_dataframe() for batch in self._batches], ignore_index=True)[columns]
        if isinstance(df, pd.Series):
            return df.to_frame().head(max_num_records or len(df))
        return df.head(max_num_records or len(df))


def rejected_record_to_error(record: dict) -> tuple[str, str]:
    """Convert a rejected record into a ``(detailed, summary)`` error tuple.

    Both elements are identical so that log output is consistent
    regardless of the ``detailed_errors`` setting.
    """
    error_msg = f"Failed data_config validation due to [{record[MetadataColumns.REJECT_REASON.value]}]"
    return (error_msg, error_msg)
