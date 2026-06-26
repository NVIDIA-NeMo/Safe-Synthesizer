# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time series preprocessing utilities for Safe Synthesizer training."""

from __future__ import annotations

import pandas as pd

from ..config import SafeSynthesizerParameters
from ..config.time_series import TimeSeriesParameters
from ..data_processing.timeseries_validation import validate_timeseries_data
from ..data_processing.validation import (
    check_no_pseudo_column_collision,
)
from ..defaults import PSEUDO_GROUP_COLUMN
from ..observability import get_logger

logger = get_logger(__name__)


def _add_pseudo_group_if_needed(df: pd.DataFrame, config: SafeSynthesizerParameters) -> tuple[pd.DataFrame, str | None]:
    """Add pseudo-group column when no group column is specified.

    This allows unified processing of grouped and ungrouped time series.

    Args:
        df: The input DataFrame.
        config: The configuration object.

    Returns:
        Tuple of (DataFrame with pseudo-group if needed, group column name).

    Raises:
        DataError: If the DataFrame already contains a column with the reserved name.
    """
    group_by_col = config.data.group_training_examples_by

    if group_by_col is None:
        check_no_pseudo_column_collision(df)
        logger.info("No group column specified, treating entire dataset as a single sequence")
        df[PSEUDO_GROUP_COLUMN] = 0  # All rows belong to one "group"
        config.data.group_training_examples_by = PSEUDO_GROUP_COLUMN
        group_by_col = PSEUDO_GROUP_COLUMN

    return df, group_by_col


def _create_elapsed_time_column(
    df: pd.DataFrame,
    ts_config: TimeSeriesParameters,
    group_by_col: str | None,
) -> tuple[pd.DataFrame, bool]:
    """Create timestamp column with elapsed time values if not provided.

    Args:
        df: The input DataFrame.
        ts_config: Time series configuration.
        group_by_col: Column name used for grouping.

    Returns:
        Tuple of (DataFrame with timestamp column, is_elapsed_time flag).
    """
    if ts_config.timestamp_column is not None:
        return df, False

    if ts_config.timestamp_interval_seconds is None:
        raise ValueError("timestamp_interval_seconds must be set when creating elapsed timestamp column")
    interval = ts_config.timestamp_interval_seconds

    logger.info(f"Adding timestamp column with interval {interval} seconds")
    timestamp_col_name = "elapsed_seconds"
    if timestamp_col_name in df.columns:
        timestamp_col_name = "_elapsed_seconds"
    ts_config.timestamp_column = timestamp_col_name

    # Create elapsed time values (seconds since start of sequence)
    if group_by_col is not None:
        # For grouped data, reset elapsed time at the start of each group
        df[ts_config.timestamp_column] = df.groupby(group_by_col).cumcount() * interval
        logger.info("Created elapsed time timestamps per group (in seconds)")
    else:
        # Single sequence - use positional range (not df.index which may be non-contiguous)
        df[ts_config.timestamp_column] = pd.RangeIndex(len(df)) * interval
        logger.info("Created elapsed time timestamps (in seconds)")

    # Move the timestamp column to be the first column
    cols = [ts_config.timestamp_column] + [c for c in df.columns if c != ts_config.timestamp_column]
    df = df.loc[:, cols]
    ts_config.timestamp_format = "elapsed_seconds"

    return df, True


def _sort_by_group_and_timestamp(df: pd.DataFrame, group_by_col: str | None, timestamp_col: str) -> pd.DataFrame:
    """Sort DataFrame by group and timestamp columns.

    Args:
        df: The input DataFrame.
        group_by_col: Column name used for grouping (can be None).
        timestamp_col: Name of the timestamp column.

    Returns:
        Sorted DataFrame with reset index.
    """
    logger.info(
        f"Sorting dataset by timestamp column '{timestamp_col}' for sequential training",
    )

    if group_by_col is not None:
        return df.sort_values([group_by_col, timestamp_col]).reset_index(drop=True)
    else:
        return df.sort_values(timestamp_col).reset_index(drop=True)


def process_timeseries_data(
    training_df: pd.DataFrame,
    config: SafeSynthesizerParameters,
) -> tuple[pd.DataFrame, SafeSynthesizerParameters]:
    """Process time series data and validate/infer timestamp parameters.

    Normalizes grouped and ungrouped time series into the same training path.
    When no group column is configured, a reserved pseudo-group column
    (``PSEUDO_GROUP_COLUMN``) is added so the whole dataset is treated as one
    sequence. Timestamp format and interval metadata inferred here are saved
    back into the resolved config for generation.

    This function:
    1. Creates a timestamp column if one doesn't exist
    2. Validates the timestamp column exists and has no missing values
    3. Sorts the data by timestamp
    4. Infers timestamp_format from the data
    5. Validates or infers timestamp_interval_seconds
    6. Sets start_timestamp and stop_timestamp

    Args:
        training_df: The training DataFrame.
        config: The configuration object with time_series settings

    Returns:
        Tuple of (processed DataFrame, updated config)

    Raises:
        ParameterError: If the timestamp column is missing, if ``timestamp_format="elapsed_seconds"``
            is set on a non-numeric column, or if an explicit format fails to parse the data.
        DataError: If the timestamp column has missing values or intervals are inconsistent.
    """
    ts_config = config.time_series

    # Step 1: Add pseudo-group if needed
    training_df, group_by_col = _add_pseudo_group_if_needed(training_df, config)

    if group_by_col is None:
        raise RuntimeError("group_by_col should have been set by _add_pseudo_group_if_needed")

    # Step 2: Create elapsed time column if timestamp not provided
    training_df, is_elapsed_time = _create_elapsed_time_column(training_df, ts_config, group_by_col)

    # timestamp_column should be set by now
    if ts_config.timestamp_column is None:
        raise RuntimeError("timestamp_column should have been set by _create_elapsed_time_column")
    config.data.order_training_examples_by = ts_config.timestamp_column

    validation = validate_timeseries_data(training_df, config)
    training_df = validation.data
    ts_config.timestamp_column = validation.timestamp_column
    ts_config.timestamp_format = validation.timestamp_format
    ts_config.timestamp_interval_seconds = validation.timestamp_interval_seconds
    ts_config.start_timestamp = validation.start_timestamp
    ts_config.stop_timestamp = validation.stop_timestamp
    is_elapsed_time = validation.is_elapsed_time

    # Step 7: Convert timestamp back to string format
    # Skip string conversion for elapsed_seconds format (values are already numeric)
    if (
        not is_elapsed_time
        and ts_config.timestamp_format is not None
        and ts_config.timestamp_format != "elapsed_seconds"
    ):
        training_df[ts_config.timestamp_column] = training_df[ts_config.timestamp_column].dt.strftime(
            ts_config.timestamp_format
        )

    return training_df, config
