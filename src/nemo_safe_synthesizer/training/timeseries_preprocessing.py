# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time series preprocessing utilities for Safe Synthesizer training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from ..config import SafeSynthesizerParameters
from ..config.time_series import TimeSeriesParameters
from ..data_processing.actions.utils import guess_datetime_format
from ..defaults import PSEUDO_GROUP_COLUMN
from ..errors import DataError, ParameterError
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
        if PSEUDO_GROUP_COLUMN in df.columns:
            raise DataError(
                f"Column '{PSEUDO_GROUP_COLUMN}' is reserved for internal use. Please rename this column in your data."
            )
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

    logger.info(f"Adding timestamp column with interval {ts_config.timestamp_interval_seconds} seconds")
    timestamp_col_name = "elapsed_seconds"
    if timestamp_col_name in df.columns:
        timestamp_col_name = "_elapsed_seconds"
    ts_config.timestamp_column = timestamp_col_name

    # Create elapsed time values (seconds since start of sequence)
    if group_by_col is not None:
        # For grouped data, reset elapsed time at the start of each group
        df[ts_config.timestamp_column] = df.groupby(group_by_col).cumcount() * ts_config.timestamp_interval_seconds
        logger.info("Created elapsed time timestamps per group (in seconds)")
    else:
        # Single sequence - use positional range (not df.index which may be non-contiguous)
        df[ts_config.timestamp_column] = pd.RangeIndex(len(df)) * ts_config.timestamp_interval_seconds
        logger.info("Created elapsed time timestamps (in seconds)")

    # Move the timestamp column to be the first column
    cols = [ts_config.timestamp_column] + [c for c in df.columns if c != ts_config.timestamp_column]
    df = df.loc[:, cols]
    ts_config.timestamp_format = "elapsed_seconds"

    return df, True


def _validate_timestamp_column(df: pd.DataFrame, timestamp_column: str) -> None:
    """Validate that the timestamp column exists and has no nulls.

    Args:
        df: The input DataFrame.
        timestamp_column: Name of the timestamp column.

    Raises:
        ParameterError: If timestamp column is not found.
        DataError: If timestamp column has missing values.
    """
    if timestamp_column not in df.columns:
        raise ParameterError(f"Timestamp column '{timestamp_column}' not found in the input data.")

    if df[timestamp_column].isnull().any():
        raise DataError(f"Timestamp column '{timestamp_column}' has missing values. Please clean the column.")


def _sort_by_group_and_timestamp(df: pd.DataFrame, group_by_col: str, timestamp_col: str) -> pd.DataFrame:
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


def _infer_and_convert_timestamp_format(df: pd.DataFrame, ts_config: TimeSeriesParameters) -> pd.DataFrame:
    """Infer timestamp format and convert column to datetime.

    Args:
        df: The input DataFrame.
        ts_config: Time series configuration.

    Returns:
        DataFrame with timestamp column converted to datetime.

    Raises:
        ParameterError: If user-provided timestamp_format doesn't match the data.
        DataError: If timestamp conversion produces invalid (NaT) values or DataFrame is empty.
    """
    if len(df) == 0:
        raise DataError("Cannot infer timestamp format from empty DataFrame")

    first_timestamp = df[ts_config.timestamp_column].iloc[0]
    user_provided_format = ts_config.timestamp_format is not None

    if ts_config.timestamp_format is None:
        inferred_format = guess_datetime_format(str(first_timestamp))
        if inferred_format is not None:
            ts_config.timestamp_format = inferred_format
            logger.info(f"Inferred timestamp format: {inferred_format}")
        else:
            logger.warning("Could not infer timestamp format from data")
    else:
        # Validate user-provided format matches the data
        try:
            datetime.strptime(str(first_timestamp), ts_config.timestamp_format)
        except ValueError as e:
            # Try to infer the correct format to help the user
            inferred_format = guess_datetime_format(str(first_timestamp))
            suggestion = ""
            if inferred_format is not None:
                suggestion = f" Did you mean: '{inferred_format}'?"
            raise ParameterError(
                f"Provided timestamp_format '{ts_config.timestamp_format}' does not match the data. "
                f"First timestamp value: '{first_timestamp}'.{suggestion}"
            ) from e

    df[ts_config.timestamp_column] = pd.to_datetime(df[ts_config.timestamp_column], errors="coerce")

    # Check for NaT values after conversion
    nat_count = df[ts_config.timestamp_column].isna().sum()
    if nat_count > 0:
        format_source = "provided" if user_provided_format else "inferred"
        raise DataError(
            f"Failed to parse {nat_count} timestamp values using {format_source} format "
            f"'{ts_config.timestamp_format}'. Please check your data or provide a valid timestamp_format."
        )

    return df


def process_timeseries_data(
    df_all: pd.DataFrame,
    config: SafeSynthesizerParameters,
) -> tuple[pd.DataFrame, SafeSynthesizerParameters]:
    """Process time series data and validate/infer timestamp parameters.

    This function:
    1. Creates a timestamp column if one doesn't exist
    2. Validates the timestamp column exists and has no missing values
    3. Sorts the data by timestamp
    4. Infers timestamp_format from the data
    5. Validates or infers timestamp_interval_seconds
    6. Sets start_timestamp and stop_timestamp

    Args:
        df_all: The input DataFrame
        config: The configuration object with time_series settings

    Returns:
        Tuple of (processed DataFrame, updated config)

    Raises:
        ParameterError: If timestamp column is not found
        DataError: If timestamp column has missing values or intervals are inconsistent
    """
    ts_config = config.time_series

    # Step 1: Add pseudo-group if needed
    df_all, group_by_col = _add_pseudo_group_if_needed(df_all, config)

    if group_by_col is None:
        raise RuntimeError("group_by_col should have been set by _add_pseudo_group_if_needed")

    # Step 2: Create elapsed time column if timestamp not provided
    df_all, is_elapsed_time = _create_elapsed_time_column(df_all, ts_config, group_by_col)

    # timestamp_column should be set by now
    if ts_config.timestamp_column is None:
        raise RuntimeError("timestamp_column should have been set by _create_elapsed_time_column")
    config.data.order_training_examples_by = ts_config.timestamp_column

    # Step 3: Validate timestamp column
    _validate_timestamp_column(df_all, ts_config.timestamp_column)

    # Step 4: Sort by group and timestamp
    df_all = _sort_by_group_and_timestamp(df_all, group_by_col, ts_config.timestamp_column)

    # Step 5: Infer format and convert to datetime (if not elapsed time)
    # Skip datetime conversion for elapsed_seconds format (either created or user-provided)
    if not is_elapsed_time and ts_config.timestamp_format != "elapsed_seconds":
        df_all = _infer_and_convert_timestamp_format(df_all, ts_config)

    # Step 6: Process groups and validate consistency
    ts_config = _process_grouped_timestamps(df_all, ts_config, group_by_col, is_elapsed_time)

    # Step 7: Convert timestamp back to string format
    # Skip string conversion for elapsed_seconds format (values are already numeric)
    if (
        not is_elapsed_time
        and ts_config.timestamp_format is not None
        and ts_config.timestamp_format != "elapsed_seconds"
    ):
        df_all[ts_config.timestamp_column] = df_all[ts_config.timestamp_column].dt.strftime(ts_config.timestamp_format)

    return df_all, config


@dataclass
class _GroupTimestampStats:
    """Statistics collected from a single group's timestamps."""

    group_name: Any
    """Identifier for the group."""

    start_timestamp: Any
    """First timestamp in the group."""

    stop_timestamp: Any
    """Last timestamp in the group."""

    interval_seconds: int | None
    """Seconds between consecutive timestamps, or ``None`` if inconsistent within the group."""


def _collect_group_timestamp_stats(
    df: pd.DataFrame,
    timestamp_col: str | None,
    group_by_col: str | None,
    is_elapsed_time: bool,
) -> list[_GroupTimestampStats]:
    """Collect timestamp statistics for each group.

    Args:
        df: The DataFrame with timestamp column.
        timestamp_col: Name of the timestamp column.
        group_by_col: Column name used for grouping.
        is_elapsed_time: If True, timestamps are integer elapsed seconds.

    Returns:
        List of statistics for each group.
    """
    stats_list = []

    for group_name, group_df in df.groupby(group_by_col):
        timestamps = group_df[timestamp_col]

        start_ts = timestamps.iloc[0]
        stop_ts = timestamps.iloc[-1]

        # Calculate interval for this group
        interval = None
        time_diffs = timestamps.diff().dropna()
        if not time_diffs.empty:
            if is_elapsed_time:
                interval_seconds = time_diffs
            else:
                interval_seconds = time_diffs.dt.total_seconds()
            unique_intervals = interval_seconds.unique()

            # Check if this group has consistent intervals
            if len(unique_intervals) == 1 or (unique_intervals.max() - unique_intervals.min()) < 0.1:
                interval = int(round(interval_seconds.iloc[0]))

        stats_list.append(
            _GroupTimestampStats(
                group_name=group_name,
                start_timestamp=start_ts,
                stop_timestamp=stop_ts,
                interval_seconds=interval,
            )
        )

    return stats_list


def _validate_interval_consistency(
    df: pd.DataFrame,
    ts_config: TimeSeriesParameters,
    group_by_col: str,
    is_elapsed_time: bool,
    group_stats: list[_GroupTimestampStats],
) -> None:
    """Validate or infer consistent interval across groups.

    Args:
        df: The DataFrame with timestamp column.
        ts_config: Time series configuration (modified in place).
        group_by_col: Column name used for grouping.
        is_elapsed_time: If True, timestamps are integer elapsed seconds.
        group_stats: Pre-collected statistics for each group.
    """
    timestamp_col = ts_config.timestamp_column

    if ts_config.timestamp_interval_seconds is not None:
        # Validate that provided interval is correct for all groups
        expected_interval = ts_config.timestamp_interval_seconds
        tolerance = 0.1

        for group_name, group_df in df.groupby(group_by_col):
            timestamps = group_df[timestamp_col]
            time_diffs = timestamps.diff().dropna()
            if not time_diffs.empty:
                if is_elapsed_time:
                    interval_seconds = time_diffs
                else:
                    interval_seconds = time_diffs.dt.total_seconds()
                if not all(abs(interval_seconds - expected_interval) <= tolerance):
                    logger.warning(
                        f"Provided timestamp_interval_seconds ({expected_interval}s) does not match "
                        f"actual intervals in group '{group_name}'.",
                    )
                    break
    else:
        # Try to infer interval - all groups must have same consistent interval
        valid_intervals = [s.interval_seconds for s in group_stats if s.interval_seconds is not None]
        if valid_intervals and len(set(valid_intervals)) == 1:
            inferred_interval = valid_intervals[0]
            ts_config.timestamp_interval_seconds = inferred_interval
            logger.info(
                f"Inferred timestamp_interval_seconds: {inferred_interval}s (consistent across all groups)",
            )
        else:
            logger.info(
                "Timestamp intervals vary across groups. timestamp_interval_seconds will remain unset.",
            )


def _validate_start_stop_consistency(
    group_stats: list[_GroupTimestampStats],
) -> tuple[str, str]:
    """Validate all groups have same start/stop timestamps.

    Args:
        group_stats: Pre-collected statistics for each group.

    Returns:
        Tuple of (start_timestamp, stop_timestamp) as strings.

    Raises:
        DataError: If start or stop timestamps differ across groups.
    """
    unique_starts = set(s.start_timestamp for s in group_stats)
    unique_stops = set(s.stop_timestamp for s in group_stats)

    if len(unique_starts) > 1:
        raise DataError(
            f"Start timestamps differ across groups. Found {len(unique_starts)} different start timestamps: "
            f"{sorted([str(t) for t in list(unique_starts)[:5]])}{'...' if len(unique_starts) > 5 else ''}. "
            f"All groups must have the same start timestamp."
        )

    if len(unique_stops) > 1:
        raise DataError(
            f"Stop timestamps differ across groups. Found {len(unique_stops)} different stop timestamps: "
            f"{sorted([str(t) for t in list(unique_stops)[:5]])}{'...' if len(unique_stops) > 5 else ''}. "
            f"All groups must have the same stop timestamp."
        )

    return str(group_stats[0].start_timestamp), str(group_stats[0].stop_timestamp)


def _process_grouped_timestamps(
    df: pd.DataFrame,
    ts_config: TimeSeriesParameters,
    group_by_col: str,
    is_elapsed_time: bool = False,
) -> TimeSeriesParameters:
    """Process timestamps for grouped time series data.

    Validates that all groups have consistent intervals and same start/stop timestamps.

    Args:
        df: The DataFrame with timestamp column already converted to datetime
        ts_config: TimeSeriesParameters configuration object
        group_by_col: Column name used for grouping
        is_elapsed_time: If True, timestamps are integer elapsed seconds

    Returns:
        Updated ts_config with validated/inferred parameters

    Raises:
        DataError: If start/stop timestamps differ across groups
    """
    # Step 1: Collect statistics for each group
    group_stats = _collect_group_timestamp_stats(df, ts_config.timestamp_column, group_by_col, is_elapsed_time)

    # Step 2: Validate/infer interval consistency
    _validate_interval_consistency(df, ts_config, group_by_col, is_elapsed_time, group_stats)

    # Step 3: Validate start/stop consistency and get values
    start_ts, stop_ts = _validate_start_stop_consistency(group_stats)
    ts_config.start_timestamp = start_ts
    ts_config.stop_timestamp = stop_ts

    logger.info(
        f"Time series range (consistent across {len(group_stats)} groups): "
        f"{ts_config.start_timestamp} to {ts_config.stop_timestamp}",
    )

    return ts_config
