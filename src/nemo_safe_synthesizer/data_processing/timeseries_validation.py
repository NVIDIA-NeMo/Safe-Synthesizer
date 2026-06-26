# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared time-series validation for preflight and training preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from ..config.parameters import SafeSynthesizerParameters
from ..config.time_series import TimeSeriesParameters
from ..defaults import PSEUDO_GROUP_COLUMN
from ..errors import DataError, ParameterError
from .actions.utils import guess_datetime_format
from .validation import check_groupby_column, check_no_pseudo_column_collision, check_timestamp_column

__all__ = [
    "TimeSeriesDataValidationError",
    "TimeSeriesGroupTimestampStats",
    "TimeSeriesParameterValidationError",
    "TimeSeriesValidationError",
    "TimeSeriesValidationResult",
    "validate_start_stop_consistency",
    "validate_timeseries_data",
]


class TimeSeriesValidationError:
    """Mixin for time-series validation errors that carry preflight issue codes."""

    code: str


class TimeSeriesDataValidationError(TimeSeriesValidationError, DataError):
    """Data-side time-series validation failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class TimeSeriesParameterValidationError(TimeSeriesValidationError, ParameterError):
    """Parameter-side time-series validation failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class TimeSeriesGroupTimestampStats:
    """Statistics collected from a single group's timestamps."""

    group_name: Any
    """Identifier for the time-series group."""

    start_timestamp: Any
    """First timestamp in the sorted group."""

    stop_timestamp: Any
    """Last timestamp in the sorted group."""

    interval_seconds: int | None
    """Seconds between consecutive timestamps, or ``None`` when no consistent interval exists."""

    record_count: int = 0
    """Number of records in the group."""


@dataclass(frozen=True)
class TimeSeriesValidationResult:
    """Validated time-series data and inferred timestamp metadata."""

    data: pd.DataFrame
    """Validated and timestamp-sorted DataFrame copy."""

    group_by_column: str
    """Column used to group time-series records."""

    timestamp_column: str
    """Column used as the timestamp after generated-column resolution."""

    timestamp_format: str
    """Resolved timestamp format, either ``"elapsed_seconds"`` or a strftime format."""

    is_elapsed_time: bool
    """Whether timestamps are numeric elapsed seconds."""

    timestamp_interval_seconds: int | None
    """Validated or inferred interval between timestamps, if available."""

    start_timestamp: str
    """Common start timestamp shared by all groups."""

    stop_timestamp: str
    """Common stop timestamp shared by all groups."""

    group_stats: tuple[TimeSeriesGroupTimestampStats, ...]
    """Per-group timestamp statistics used to derive the result."""


def _resolve_group_column(data: pd.DataFrame, config: SafeSynthesizerParameters) -> tuple[pd.DataFrame, str]:
    group_by_col = config.data.group_training_examples_by
    working_df = data.copy()

    if group_by_col is None:
        try:
            check_no_pseudo_column_collision(working_df)
        except ParameterError as exc:
            raise TimeSeriesParameterValidationError("pseudo_column_collision", str(exc)) from exc
        except DataError as exc:
            raise TimeSeriesDataValidationError("pseudo_column_collision", str(exc)) from exc
        working_df[PSEUDO_GROUP_COLUMN] = 0
        return working_df, PSEUDO_GROUP_COLUMN

    try:
        check_groupby_column(working_df, group_by_col)
    except ParameterError as exc:
        raise TimeSeriesParameterValidationError("column_not_found", str(exc)) from exc
    except DataError as exc:
        raise TimeSeriesDataValidationError("column_nulls", str(exc)) from exc
    return working_df, group_by_col


def _add_elapsed_time_column(
    data: pd.DataFrame,
    ts_config: TimeSeriesParameters,
    group_by_col: str,
) -> tuple[pd.DataFrame, str]:
    if ts_config.timestamp_interval_seconds is None:
        raise TimeSeriesParameterValidationError(
            "timestamp_not_found",
            "Time-series mode requires either timestamp_column or timestamp_interval_seconds.",
        )

    timestamp_col = "elapsed_seconds"
    if timestamp_col in data.columns:
        timestamp_col = "_elapsed_seconds"

    working_df = data.copy()
    working_df[timestamp_col] = working_df.groupby(group_by_col).cumcount() * ts_config.timestamp_interval_seconds
    cols = [timestamp_col] + [c for c in working_df.columns if c != timestamp_col]
    return working_df.loc[:, cols], timestamp_col


def _detect_elapsed_seconds_format(data: pd.DataFrame, ts_config: TimeSeriesParameters, timestamp_col: str) -> bool:
    column = data[timestamp_col]

    if ts_config.timestamp_format == "elapsed_seconds":
        if not pd.api.types.is_numeric_dtype(column):
            raise TimeSeriesParameterValidationError(
                "timestamp_elapsed_non_numeric",
                f"timestamp_format='elapsed_seconds' requires timestamp column "
                f"'{timestamp_col}' to be numeric, but got dtype '{column.dtype}'.",
            )
        return True

    if ts_config.timestamp_format is None and pd.api.types.is_integer_dtype(column):
        return True

    return False


def _infer_and_convert_timestamp_format(df: pd.DataFrame, ts_config: TimeSeriesParameters) -> pd.DataFrame:
    """Infer or validate timestamp format and return a converted copy."""
    if len(df) == 0:
        raise TimeSeriesDataValidationError(
            "timestamp_parse_failed", "Cannot infer timestamp format from empty DataFrame"
        )

    timestamp_col = ts_config.timestamp_column
    if timestamp_col is None:
        raise TimeSeriesParameterValidationError(
            "timestamp_not_found",
            "timestamp_column must be set before inferring timestamp format.",
        )

    first_timestamp = df[timestamp_col].iloc[0]
    timestamp_format = ts_config.timestamp_format
    user_provided_format = timestamp_format is not None

    if timestamp_format is None:
        timestamp_format = guess_datetime_format(str(first_timestamp))
        if timestamp_format is None:
            raise TimeSeriesParameterValidationError(
                "timestamp_format_mismatch",
                f"Could not infer timestamp format from column '{timestamp_col}' "
                f"(first value: '{first_timestamp}'). "
                f"If the column contains numeric elapsed time values, set timestamp_format='elapsed_seconds'. "
                f"Otherwise, provide an explicit timestamp_format (e.g. '%Y-%m-%d %H:%M:%S').",
            )
        ts_config.timestamp_format = timestamp_format
    else:
        try:
            datetime.strptime(str(first_timestamp), timestamp_format)
        except ValueError as exc:
            inferred_format = guess_datetime_format(str(first_timestamp))
            suggestion = f" Did you mean: '{inferred_format}'?" if inferred_format is not None else ""
            raise TimeSeriesParameterValidationError(
                "timestamp_format_mismatch",
                f"Provided timestamp_format '{timestamp_format}' does not match the data. "
                f"First timestamp value: '{first_timestamp}'.{suggestion}",
            ) from exc

    converted = df.copy()
    converted[timestamp_col] = pd.to_datetime(converted[timestamp_col], format=timestamp_format, errors="coerce")

    nat_count = int(converted[timestamp_col].isna().sum())
    if nat_count > 0:
        format_source = "provided" if user_provided_format else "inferred"
        raise TimeSeriesDataValidationError(
            "timestamp_parse_failed",
            f"Failed to parse {nat_count} timestamp values using {format_source} format "
            f"'{timestamp_format}'. Please check your data or provide a valid timestamp_format.",
        )

    return converted


def _sort_by_group_and_timestamp(df: pd.DataFrame, group_by_col: str, timestamp_col: str) -> pd.DataFrame:
    return df.sort_values([group_by_col, timestamp_col]).reset_index(drop=True)


def _interval_seconds(timestamps: pd.Series, is_elapsed_time: bool) -> pd.Series:
    time_diffs = timestamps.diff().dropna()
    if time_diffs.empty:
        return time_diffs
    if is_elapsed_time:
        return time_diffs
    return time_diffs.dt.total_seconds()


def _validate_equal_group_lengths(df: pd.DataFrame, group_by_col: str) -> None:
    counts = df.groupby(group_by_col, sort=False).size()
    if counts.nunique() <= 1:
        return

    examples = ", ".join(f"{group}={count}" for group, count in counts.head(5).items())
    suffix = "..." if len(counts) > 5 else ""
    raise TimeSeriesDataValidationError(
        "timeseries_group_length_mismatch",
        f"Time-series groups must contain the same number of records. Found group sizes: {examples}{suffix}.",
    )


def _collect_group_timestamp_stats(
    df: pd.DataFrame,
    timestamp_col: str,
    group_by_col: str,
    is_elapsed_time: bool,
) -> tuple[TimeSeriesGroupTimestampStats, ...]:
    stats_list = []

    for group_name, group_df in df.groupby(group_by_col, sort=False):
        timestamps = group_df[timestamp_col]
        intervals = _interval_seconds(timestamps, is_elapsed_time)
        interval = None
        if not intervals.empty:
            unique_intervals = intervals.unique()
            if len(unique_intervals) == 1 or (unique_intervals.max() - unique_intervals.min()) < 0.1:
                interval = int(round(float(intervals.iloc[0])))

        stats_list.append(
            TimeSeriesGroupTimestampStats(
                group_name=group_name,
                start_timestamp=timestamps.iloc[0],
                stop_timestamp=timestamps.iloc[-1],
                interval_seconds=interval,
                record_count=len(group_df),
            )
        )

    return tuple(stats_list)


def _validate_interval_consistency(
    df: pd.DataFrame,
    timestamp_col: str,
    group_by_col: str,
    is_elapsed_time: bool,
    expected_interval_seconds: int | None,
    group_stats: tuple[TimeSeriesGroupTimestampStats, ...],
) -> int | None:
    tolerance = 0.1

    if expected_interval_seconds is not None:
        for group_name, group_df in df.groupby(group_by_col, sort=False):
            intervals = _interval_seconds(group_df[timestamp_col], is_elapsed_time)
            if not intervals.empty and not all(abs(intervals - expected_interval_seconds) <= tolerance):
                raise TimeSeriesDataValidationError(
                    "timestamp_interval_mismatch",
                    f"Provided timestamp_interval_seconds ({expected_interval_seconds}s) does not match "
                    f"actual intervals in group '{group_name}'.",
                )
        return expected_interval_seconds

    invalid_groups = [s.group_name for s in group_stats if s.record_count > 1 and s.interval_seconds is None]
    if invalid_groups:
        raise TimeSeriesDataValidationError(
            "timestamp_interval_mismatch",
            f"Timestamp intervals are inconsistent within group '{invalid_groups[0]}'.",
        )

    valid_intervals = [s.interval_seconds for s in group_stats if s.interval_seconds is not None]
    unique_intervals = set(valid_intervals)
    if len(unique_intervals) > 1:
        raise TimeSeriesDataValidationError(
            "timestamp_interval_mismatch",
            f"Timestamp intervals differ across groups. Found intervals: {sorted(unique_intervals)}.",
        )
    if valid_intervals:
        return valid_intervals[0]
    return None


def validate_start_stop_consistency(
    group_stats: tuple[TimeSeriesGroupTimestampStats, ...] | list[TimeSeriesGroupTimestampStats],
) -> tuple[str, str]:
    """Validate all groups have the same start/stop timestamps."""
    if not group_stats:
        raise TimeSeriesDataValidationError(
            "timeseries_empty",
            "Time-series data must contain at least one record.",
        )

    unique_starts = set(s.start_timestamp for s in group_stats)
    unique_stops = set(s.stop_timestamp for s in group_stats)

    if len(unique_starts) > 1:
        raise TimeSeriesDataValidationError(
            "timeseries_start_mismatch",
            f"Start timestamps differ across groups. Found {len(unique_starts)} different start timestamps: "
            f"{sorted([str(t) for t in list(unique_starts)[:5]])}{'...' if len(unique_starts) > 5 else ''}. "
            f"All groups must have the same start timestamp.",
        )

    if len(unique_stops) > 1:
        raise TimeSeriesDataValidationError(
            "timeseries_stop_mismatch",
            f"Stop timestamps differ across groups. Found {len(unique_stops)} different stop timestamps: "
            f"{sorted([str(t) for t in list(unique_stops)[:5]])}{'...' if len(unique_stops) > 5 else ''}. "
            f"All groups must have the same stop timestamp.",
        )

    return str(group_stats[0].start_timestamp), str(group_stats[0].stop_timestamp)


def validate_timeseries_data(data: pd.DataFrame, config: SafeSynthesizerParameters) -> TimeSeriesValidationResult:
    """Validate time-series data shape and infer timestamp metadata.

    The validator performs the same timestamp normalization checks needed by
    training preprocessing, but operates on copies so preflight can run it
    without mutating the caller's DataFrame or config.

    Args:
        data: Training DataFrame to validate.
        config: Safe Synthesizer parameters containing time-series settings.

    Returns:
        A validation result with a sorted DataFrame copy and resolved timestamp
        metadata suitable for updating the runtime config.

    Raises:
        TimeSeriesParameterValidationError: If configuration references missing
            columns or incompatible timestamp formats.
        TimeSeriesDataValidationError: If the data violates time-series shape
            invariants.
    """
    ts_config = config.time_series
    working_df, group_by_col = _resolve_group_column(data, config)

    timestamp_col = ts_config.timestamp_column
    if timestamp_col is None:
        working_df, timestamp_col = _add_elapsed_time_column(working_df, ts_config, group_by_col)
        timestamp_format = "elapsed_seconds"
        is_elapsed_time = True
    else:
        try:
            check_timestamp_column(working_df, timestamp_col)
        except ParameterError as exc:
            raise TimeSeriesParameterValidationError("timestamp_not_found", str(exc)) from exc
        except DataError as exc:
            raise TimeSeriesDataValidationError("timestamp_nulls", str(exc)) from exc

        is_elapsed_time = _detect_elapsed_seconds_format(working_df, ts_config, timestamp_col)
        timestamp_format = "elapsed_seconds" if is_elapsed_time else ts_config.timestamp_format

    if not is_elapsed_time:
        ts_config_copy = ts_config.model_copy(update={"timestamp_column": timestamp_col})
        working_df = _infer_and_convert_timestamp_format(working_df, ts_config_copy)
        timestamp_format = ts_config_copy.timestamp_format
        if timestamp_format is None:
            timestamp_format = guess_datetime_format(str(working_df[timestamp_col].iloc[0]))
    else:
        timestamp_format = "elapsed_seconds"

    if timestamp_format is None:
        raise TimeSeriesParameterValidationError(
            "timestamp_format_mismatch",
            f"Could not resolve timestamp format for column '{timestamp_col}'.",
        )

    working_df = _sort_by_group_and_timestamp(working_df, group_by_col, timestamp_col)
    _validate_equal_group_lengths(working_df, group_by_col)
    group_stats = _collect_group_timestamp_stats(working_df, timestamp_col, group_by_col, is_elapsed_time)
    start_ts, stop_ts = validate_start_stop_consistency(group_stats)
    interval_seconds = _validate_interval_consistency(
        working_df,
        timestamp_col,
        group_by_col,
        is_elapsed_time,
        ts_config.timestamp_interval_seconds,
        group_stats,
    )

    return TimeSeriesValidationResult(
        data=working_df,
        group_by_column=group_by_col,
        timestamp_column=timestamp_col,
        timestamp_format=timestamp_format,
        is_elapsed_time=is_elapsed_time,
        timestamp_interval_seconds=interval_seconds,
        start_timestamp=start_ts,
        stop_timestamp=stop_ts,
        group_stats=group_stats,
    )
