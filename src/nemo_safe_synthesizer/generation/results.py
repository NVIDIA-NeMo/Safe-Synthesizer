# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Self

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
    """
    Represents the results of a job that generates data from batches.

    This class encapsulates detailed information about the generated data,
    including statistics for valid and invalid records, total prompts processed,
    and computation time.

    Attributes:
        df: A DataFrame containing the generated records based on the
            input batches.
        status: Represents the overall generation status derived
            from the processed batches.
        num_valid_records: The total number of records deemed valid during
            generation.
        num_invalid_records: The total number of records deemed invalid during
            generation.
        num_prompts: The total number of prompts that were processed during
            generation.
        valid_record_fraction: The fraction of valid records among all records.
        batch_valid_record_fractions: A list of valid record fractions
            for individual batches.
        elapsed_time: The total time elapsed during the generation
            process in seconds. Defaults to None.
    """

    df: pd.DataFrame
    status: GenerationStatus

    num_valid_records: int
    num_invalid_records: int
    num_prompts: int
    valid_record_fraction: float
    batch_valid_record_fractions: list[float]
    elapsed_time: float | None = None

    @classmethod
    def from_batches(
        cls, batches: GenerationBatches, max_num_records: int | None, columns: list[str], elapsed_time: float
    ) -> Self:
        """Create a GenerateJobResults object primarily from a GenerationBatches object."""
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


class GenerationBatches(object):
    """Object that tracks the various batches during the generation phase of a job. Mostly internal for the Generator backend


    Args:
        target_num_records: The target number of records to generate.
        batches: A list of BatchResults objects.
        max_num_prompts_per_batch: The maximum number of prompts to process in a batch.
        invalid_fraction_threshold: The fraction of invalid records that will stop generation after the `patience` limit is reached.
        patience: Number of consecutive generations where the `invalid_fraction_threshold` is reached before stopping generation.
        data_actions_fn: A filtering function that'll postprocess and validate records from a batch.

    Attributes:
        status: The current status of the generation job.
        running_stopping_metric: A RunningStatistics object to track the stopping metric.
        stop_condition: The stopping condition object for the generation job.
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
        self._batches_df: Optional[pd.DataFrame] = None

    def _apply_data_actions_fn(self, batch: Batch) -> None:
        """
        This function will take the passed in `batch` and run it
        through the `data_actions_fn`. `data_actions_fn` will
        take a dataframe and do some transformations and filtering on the
        data.

        Usually, this `data_action_fn` first involves running some `postprocessing`,
        where we "undo" some of the `preprocessing` that was done to
        the data earlier in the navft lifecycle.
        Once the data has been `postprocessed`, we run the dataframe
        through a validation function that filters out results that don't
        pass user-specified rules.

        The `data_actions_fn` expects a `df`, though we only have lists of records
        at the time that this function is called per-batch. To filter the records, we must:
            1. tag each record with unique temporary id
            2. convert all the records to a batch_df
            3. run the batch_df through our `data_actions_fn`
            4. take the records the passed validation and return them in the batch
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
        """Return the number of batches in the generation job."""
        return len(self._batches)

    @property
    def num_prompts(self) -> int:
        """Return the total number of prompts processed in the generation job."""
        return sum([batch.num_prompts for batch in self._batches])

    @property
    def num_invalid_records(self) -> int:
        """Return the total number of invalid records generated in the generation job."""
        return sum([batch.num_invalid_records for batch in self._batches])

    @property
    def num_valid_records(self) -> int:
        """Return the total number of valid records generated in the generation job."""
        return sum([batch.num_valid_records for batch in self._batches])

    def add_batch(self, batch: Batch) -> None:
        """Add a Batch object to the generation job.
        Status of the stopped jobs depends on the following conditions:
        - Jobs with the first batch of all invalid records are stopped, regardless of the
         stop_condition status.
        - Jobs with a stop_condition will continue processing even if subsequent batches
          (batch number > 1) contain all invalid records, until the stop_condition is met.
        - Jobs with stop_condition of None will stop generation upon encountering any
          batch with all invalid records irrespective of the batch number.
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
            assert self.stop_condition is not None
            logger.error(
                "🛑 Stopping generation prematurely. The stopping "
                "condition was reached with a running average invalid "
                f"fraction of {self.stop_condition.last_value:.2%}."
                " Please consider increasing the 'num_input_records_to_sample' parameter.",
            )
            raise GenerationError(
                "Generation stopped prematurely because "
                f"the average fraction of invalid records was higher than {self.stop_condition.last_value:.2%}."
                " Please consider increasing the 'num_input_records_to_sample' parameter.",
            )

    def to_dataframe(self, columns: list[str], max_num_records: Optional[int] = None) -> pd.DataFrame:
        """Return a DataFrame of the valid records generated in the generation job."""
        # return an empty DataFrame when any generated batch has 0 valid records.
        if self.num_valid_records == 0:
            return pd.DataFrame()
        df = pd.concat([batch.to_dataframe() for batch in self._batches], ignore_index=True)[columns]
        if isinstance(df, pd.Series):
            return df.to_frame().head(max_num_records or len(df))
        return df.head(max_num_records or len(df))


def rejected_record_to_error(record: dict) -> tuple[str, str]:
    error_msg = f"Failed data_config validation due to [{record[MetadataColumns.REJECT_REASON.value]}]"

    # The ret[0]/ret[1] are the same because we don't want the log output
    # to differ when detailed logging is enabled/disabled.
    return (error_msg, error_msg)
