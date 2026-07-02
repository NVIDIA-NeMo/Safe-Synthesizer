# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from ..configurator.parameters import (
    Parameters,
)

__all__ = [
    "TimeSeriesColdStartTrainingParameters",
    "TimeSeriesParameters",
]


TimeSeriesColdStartStrategy = Literal["partial_record_prefix", "start_instruction"]


class TimeSeriesColdStartTrainingParameters(Parameters):
    """Experimental training controls for time-series cold-start strategies."""

    enabled: Annotated[
        bool,
        Field(
            description=(
                "Whether to add cold-start-shaped training examples at the beginning of each time-series group."
            ),
        ),
    ] = False

    strategies: list[TimeSeriesColdStartStrategy] = Field(
        default_factory=list,
        description=(
            "Cold-start strategies to add as start-shaped training examples. "
            "When empty and enabled, the active initialization_strategy is used if it is a cold-start strategy."
        ),
    )

    start_example_weight: Annotated[
        float,
        Field(
            ge=1.0,
            description=(
                "Multiplier for start-shaped training example exposure. "
                "1.0 preserves current training, 2.0 doubles the start examples, and 3.0 triples them."
            ),
        ),
    ] = 1.0

    start_example_records: Annotated[
        int | None,
        Field(
            ge=1,
            description=(
                "Number of initial records per group to include in each start-shaped example. "
                "When unset, time_series.prefill_context_records is used."
            ),
        ),
    ] = None


class TimeSeriesParameters(Parameters):
    """Configuration for time-series mode in the Safe Synthesizer pipeline.

    Controls whether a dataset is treated as time-series data, including
    timestamp column selection, interval inference, and format validation.
    The time-series pipeline is currently experimental.
    """

    is_timeseries: Annotated[
        bool,
        Field(
            description=(
                "Whether to treat the dataset as time series. When enabled, either ``timestamp_column`` or "
                "``timestamp_interval_seconds`` is required. "
                "For grouped time series, ``group_training_examples_by`` needs to be set."
            ),
        ),
    ] = False

    timestamp_column: Annotated[
        str | None,
        Field(
            description=(
                "Name of the column containing timestamps used to order records when ``is_timeseries`` is ``True``. "
                "Required only when ``is_timeseries`` is ``True`` and ``timestamp_interval_seconds`` is not provided."
            ),
        ),
    ] = None

    timestamp_interval_seconds: Annotated[
        int | None,
        Field(
            description="Interval in seconds between timestamps. If not provided, the timestamp column will be used to infer the interval.",
        ),
    ] = None

    timestamp_format: Annotated[
        str | None,
        Field(
            description=(
                "Format of the timestamp column. Accepts either: "
                "(1) Python strftime format codes for string timestamps "
                "(e.g., '%Y-%m-%d %H:%M:%S', '%m/%d/%Y'), or "
                "(2) 'elapsed_seconds' for numeric (int/float) timestamps representing seconds "
                "as an increasing counter (e.g., 0, 60, 120 for 1-minute intervals). "
                "If not provided, the format will be inferred from the data."
            ),
        ),
    ] = None

    @field_validator("timestamp_format")
    @classmethod
    def validate_timestamp_format(cls, v: str | None) -> str | None:
        """Validate that timestamp_format is a valid strftime format string."""
        if v is None or v == "elapsed_seconds":
            return v
        try:
            datetime.now().strftime(v)
        except ValueError as e:
            raise ValueError(f"Invalid strftime format '{v}': {e}") from e
        return v

    start_timestamp: Annotated[
        str | int | None,
        Field(
            description="Start timestamp. If not provided, the first timestamp in the timestamp column will be used.",
        ),
    ] = None

    stop_timestamp: Annotated[
        str | int | None,
        Field(
            description="Stop timestamp. If not provided, the last timestamp in the timestamp column will be used.",
        ),
    ] = None

    initialization_strategy: Annotated[
        Literal["training_prefill", "empty", "start_instruction", "partial_record_prefix"],
        Field(
            description=(
                "Experimental time-series generation initialization strategy. "
                "'training_prefill' preserves the existing behavior; cold-start strategies avoid injecting "
                "training rows into the initial prompt."
            ),
        ),
    ] = "training_prefill"

    prefill_context_records: Annotated[
        int,
        Field(
            ge=0,
            description=(
                "Number of recently generated records to keep in the rolling time-series prompt context. "
                "Defaults to 3 to match the current hard-coded behavior."
            ),
        ),
    ] = 3

    cold_start_instruction_template: Annotated[
        str | None,
        Field(
            description=(
                "Optional format string appended to the time-series generation instruction for cold-start "
                "experiments. Available fields include group_id, group_column, timestamp_column, "
                "start_timestamp, stop_timestamp, and timestamp_interval_seconds."
            ),
        ),
    ] = None

    cold_start_training: TimeSeriesColdStartTrainingParameters = Field(
        description="Experimental training augmentation controls for time-series cold-start strategies.",
        default_factory=TimeSeriesColdStartTrainingParameters,
    )

    @model_validator(mode="after")
    def check_timestamp_column_or_interval_when_timeseries(self) -> Self:
        """Validate that time-series mode has a timestamp source and non-timeseries mode has no `timestamp_column`."""
        if self.is_timeseries:
            if self.timestamp_column is None and self.timestamp_interval_seconds is None:
                raise ValueError(
                    "At least one of timestamp_column or timestamp_interval_seconds must be provided when is_timeseries is True."
                )
        else:
            if self.timestamp_column is not None:
                raise ValueError("timestamp_column can only be set when is_timeseries is True.")
            if self.timestamp_interval_seconds is not None:
                raise ValueError("timestamp_interval_seconds can only be set when is_timeseries is True.")
        if self.timestamp_interval_seconds is not None and self.timestamp_interval_seconds <= 0:
            raise ValueError("timestamp_interval_seconds must be a positive integer.")
        return self
