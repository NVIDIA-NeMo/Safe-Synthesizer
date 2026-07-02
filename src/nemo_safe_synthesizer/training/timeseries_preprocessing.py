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
from ..data_processing.validation import (
    check_no_pseudo_column_collision,
)
from ..data_processing.validation import (
    check_timestamp_column as _check_timestamp_column,
)
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


def _detect_elapsed_seconds_format(df: pd.DataFrame, ts_config: TimeSeriesParameters) -> bool:
    """Detect whether the timestamp column should be treated as elapsed seconds.

    Two cases are recognized:

    - The user explicitly set ``timestamp_format="elapsed_seconds"``. The column
      must be numeric; otherwise we fail fast with a ``ParameterError`` rather
      than a non-obvious ``TypeError`` later in interval inference.
    - ``timestamp_format`` is unset and the column is integer-typed. Auto-detection
      is intentionally restricted to integer dtypes because downstream
      interval/start/stop handling assumes integer-second resolution (see
      ``_collect_group_timestamp_stats``); accepting floats would silently
      truncate sub-second values. Users with fractional-second data must set
      ``timestamp_format`` explicitly.

    Side effect: when auto-detection succeeds, ``ts_config.timestamp_format`` is
    set to ``"elapsed_seconds"``.

    Args:
        df: DataFrame containing the already-validated timestamp column.
        ts_config: Time series configuration (may be mutated).

    Returns:
        ``True`` if the column should be treated as elapsed seconds, ``False`` otherwise.

    Raises:
        ParameterError: If ``timestamp_format="elapsed_seconds"`` is set on a non-numeric column.
    """
    column = df[ts_config.timestamp_column]

    if ts_config.timestamp_format == "elapsed_seconds":
        if not pd.api.types.is_numeric_dtype(column):
            raise ParameterError(
                f"timestamp_format='elapsed_seconds' requires timestamp column "
                f"'{ts_config.timestamp_column}' to be numeric, but got dtype '{column.dtype}'."
            )
        return True

    if ts_config.timestamp_format is None and pd.api.types.is_integer_dtype(column):
        ts_config.timestamp_format = "elapsed_seconds"
        logger.info(
            f"Timestamp column '{ts_config.timestamp_column}' is integer-typed; treating as elapsed seconds",
        )
        return True

    return False


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
            raise ParameterError(
                f"Could not infer timestamp format from column '{ts_config.timestamp_column}' "
                f"(first value: '{first_timestamp}'). "
                f"If the column contains numeric elapsed time values, set timestamp_format='elapsed_seconds'. "
                f"Otherwise, provide an explicit timestamp_format (e.g. '%Y-%m-%d %H:%M:%S')."
            )
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

    # Step 3: Validate timestamp column -- run before any dtype checks so a missing
    # column raises ParameterError with actionable guidance rather than KeyError.
    _check_timestamp_column(training_df, ts_config.timestamp_column)

    if not is_elapsed_time:
        is_elapsed_time = _detect_elapsed_seconds_format(training_df, ts_config)

    # Step 4: Sort by group and timestamp
    training_df = _sort_by_group_and_timestamp(training_df, group_by_col, ts_config.timestamp_column)

    # Step 5: Infer format and convert to datetime (if not elapsed time)
    # Skip datetime conversion for elapsed_seconds format (either created or user-provided)
    if not is_elapsed_time and ts_config.timestamp_format != "elapsed_seconds":
        training_df = _infer_and_convert_timestamp_format(training_df, ts_config)

    # Step 6: Process groups and validate consistency
    ts_config = _process_grouped_timestamps(training_df, ts_config, group_by_col, is_elapsed_time)

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
