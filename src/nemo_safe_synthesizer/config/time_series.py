# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import Field, field_validator, model_validator

from ..configurator.parameters import (
    Parameters,
)

__all__ = [
    "TimeSeriesParameters",
]


class TimeSeriesParameters(Parameters):
    """Configuration for time series parameters.

    Attributes:
        is_timeseries: Whether to treat the dataset as time series. For grouped time series, `group_training_examples_by` must be set.
        timestamp_column: Name of the column containing timestamps used to order records when `is_timeseries` is True.
            Required only when `is_timeseries` is True and `timestamp_interval_seconds` is not provided.
        timestamp_interval_seconds: (Optional) Interval in seconds between timestamps. If not provided, the timestamp column will be used to infer the interval.
        timestamp_format: Format of the timestamp column using Python strftime format codes. If not provided, the format will be inferred.
        start_timestamp: (Optional) Start timestamp. If not provided, the first timestamp in the timestamp column will be used.
        stop_timestamp: (Optional) Stop timestamp. If not provided, the last timestamp in the timestamp column will be used.
    """

    is_timeseries: Annotated[
        bool,
        Field(
            description=(
                "Whether to treat the dataset as time series. When enabled, either `timestamp_column` or `timestamp_interval_seconds` is required. "
                "For grouped time series, `group_training_examples_by` needs to be set. "
            ),
        ),
    ] = False

    timestamp_column: Annotated[
        str | None,
        Field(
            description=(
                "Name of the column containing timestamps used to order records when `is_timeseries` is True. "
                "Required only when `is_timeseries` is True and `timestamp_interval_seconds` is not provided."
            ),
        ),
    ] = None

    timestamp_interval_seconds: Annotated[
        int | None,
        Field(
            description="(Optional) Interval in seconds between timestamps. If not provided, the timestamp column will be used to infer the interval.",
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
            description="(Optional) Start timestamp. If not provided, the first timestamp in the timestamp column will be used.",
        ),
    ] = None

    stop_timestamp: Annotated[
        str | int | None,
        Field(
            description="(Optional) Stop timestamp. If not provided, the last timestamp in the timestamp column will be used.",
        ),
    ] = None

    @model_validator(mode="after")
    def check_timestamp_column_or_interval_when_timeseries(self):
        """
        When is_timeseries is True, require that at least one of timestamp_column or timestamp_interval_seconds is not None.
        When is_timeseries is False, timestamp_column should not be set.
        """
        if self.is_timeseries:
            if self.timestamp_column is None and self.timestamp_interval_seconds is None:
                raise ValueError(
                    "At least one of timestamp_column or timestamp_interval_seconds must be provided when is_timeseries is True."
                )
        else:
            if self.timestamp_column is not None:
                raise ValueError("timestamp_column can only be set when is_timeseries is True.")
        return self
