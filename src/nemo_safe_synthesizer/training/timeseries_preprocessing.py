# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time series preprocessing utilities for Safe Synthesizer training."""

from __future__ import annotations

import pandas as pd

from ..config import SafeSynthesizerParameters
from ..data_processing.timeseries_validation import validate_timeseries_data
from ..observability import get_logger

logger = get_logger(__name__)


def _reorder_timeseries_columns(
    dataframe: pd.DataFrame,
    group_by_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """Put time-series identity columns first in the persisted schema order.

    The resulting order is shared by the prompt schema, training JSONL, and
    partial-record generation prefix. The pseudo-group remains internal and is
    excluded later when the persisted schema is built.

    Args:
        dataframe: Validated and chronologically sorted training data.
        group_by_column: Real or pseudo group column.
        timestamp_column: Resolved timestamp column.

    Returns:
        A view with group and timestamp columns first, followed by all remaining
        columns in their original relative order.
    """
    leading_columns = list(dict.fromkeys((group_by_column, timestamp_column)))
    remaining_columns = [column for column in dataframe.columns if column not in leading_columns]
    return dataframe.loc[:, [*leading_columns, *remaining_columns]]


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
    7. Orders group and timestamp columns first for partial-prefix generation

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
    original_group_column = config.data.group_training_examples_by
    original_timestamp_column = ts_config.timestamp_column
    validation = validate_timeseries_data(training_df, config)
    training_df = validation.data
    if original_group_column is None:
        logger.info("No group column specified, treating entire dataset as a single sequence")
    if original_timestamp_column is None:
        logger.info(f"Added timestamp column '{validation.timestamp_column}' with elapsed seconds")
    config.data.group_training_examples_by = validation.group_by_column
    config.data.order_training_examples_by = validation.timestamp_column
    ts_config.timestamp_column = validation.timestamp_column
    ts_config.timestamp_format = validation.timestamp_format
    ts_config.timestamp_interval_seconds = validation.timestamp_interval_seconds
    ts_config.start_timestamp = validation.start_timestamp
    ts_config.stop_timestamp = validation.stop_timestamp
    is_elapsed_time = validation.is_elapsed_time
    logger.info(f"Resolved time-series timestamp format: {validation.timestamp_format}")
    if validation.timestamp_interval_seconds is not None:
        logger.info(f"Resolved timestamp_interval_seconds: {validation.timestamp_interval_seconds}s")
    logger.info(
        f"Time series range (consistent across {len(validation.group_stats)} groups): "
        f"{validation.start_timestamp} to {validation.stop_timestamp}",
    )

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

    training_df = _reorder_timeseries_columns(
        training_df,
        validation.group_by_column,
        validation.timestamp_column,
    )
    return training_df, config
