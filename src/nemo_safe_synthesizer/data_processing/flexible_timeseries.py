# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation and validation for automatically routed flexible time series."""

from __future__ import annotations

import pandas as pd

from ..config.parameters import SafeSynthesizerParameters
from ..defaults import PSEUDO_GROUP_COLUMN
from ..errors import DataError, ParameterError
from .actions.utils import guess_datetime_format
from .validation import (
    check_column_has_no_nulls,
    check_groupby_column,
    check_no_pseudo_column_collision,
    check_timestamp_column,
)

__all__ = [
    "prepare_flexible_timeseries_data",
    "validate_prepared_flexible_timeseries_data",
]


def _unused_column_name(preferred: str, columns: list[str]) -> str:
    """Return the preferred column name or a deterministic suffixed variant."""
    if preferred not in columns:
        return preferred
    suffix = 1
    while f"{preferred}_{suffix}" in columns:
        suffix += 1
    return f"{preferred}_{suffix}"


def _resolve_group(data: pd.DataFrame, config: SafeSynthesizerParameters) -> tuple[pd.DataFrame, str]:
    """Return a copy with a validated real or pseudo group column."""
    group_column = config.data.group_training_examples_by
    working = data.copy()
    if group_column is None:
        check_no_pseudo_column_collision(working)
        working[PSEUDO_GROUP_COLUMN] = 0
        return working, PSEUDO_GROUP_COLUMN
    check_groupby_column(working, group_column)
    return working, group_column


def _source_order_column(data: pd.DataFrame, config: SafeSynthesizerParameters) -> str | None:
    """Resolve the source column used for deterministic within-group ordering."""
    candidates = (
        config.data.order_training_examples_by,
        config.time_series.timestamp_column,
    )
    for candidate in candidates:
        if candidate is not None and candidate in data.columns:
            if candidate == config.data.order_training_examples_by:
                check_column_has_no_nulls(data, candidate, role="Order by")
            else:
                check_timestamp_column(data, candidate)
            return candidate
    return None


def _timestamp_sort_key(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
    timestamp_column: str,
) -> pd.Series:
    """Return normalized timestamp values for chronological source sorting."""
    timestamps = data[timestamp_column]
    timestamp_format = config.time_series.timestamp_format
    if timestamp_format == "elapsed_seconds" or (
        timestamp_format is None and pd.api.types.is_integer_dtype(timestamps)
    ):
        return timestamps

    if timestamp_format is None:
        timestamp_format = guess_datetime_format(str(timestamps.iloc[0]))
        if timestamp_format is None:
            raise ParameterError(
                f"Could not infer timestamp format from column '{timestamp_column}' "
                f"(first value: '{timestamps.iloc[0]}')."
            )

    parsed = pd.to_datetime(timestamps, format=timestamp_format, errors="coerce")
    invalid_count = int(parsed.isna().sum())
    if invalid_count:
        raise DataError(
            f"Failed to parse {invalid_count} timestamp values from column '{timestamp_column}' "
            f"using format '{timestamp_format}'."
        )
    return parsed


def _validate_common_columns(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
    group_column: str,
) -> None:
    index_column = config.time_series.sequence_index_column
    if index_column not in data.columns:
        raise ParameterError(f"Sequence index column {index_column!r} is not present in prepared data.")
    if data[index_column].isna().any():
        raise DataError(f"Sequence index column {index_column!r} must not contain null values.")
    if not pd.api.types.is_integer_dtype(data[index_column]) or pd.api.types.is_bool_dtype(data[index_column]):
        raise DataError(f"Sequence index column {index_column!r} must contain integer values.")

    for group_name, group in data.groupby(group_column, sort=False, dropna=False):
        expected = list(range(len(group)))
        actual = group[index_column].tolist()
        if actual != expected:
            raise DataError(
                f"Prepared sequence group {group_name!r} must have contiguous zero-based indices; got {actual!r}."
            )


def _validate_last_marker_data(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
    group_column: str,
) -> None:
    marker_column = config.time_series.sequence_marker_column
    if marker_column not in data.columns:
        raise ParameterError(f"Last-marker column {marker_column!r} is not present in prepared data.")
    if not pd.api.types.is_bool_dtype(data[marker_column]):
        raise DataError(f"Last-marker column {marker_column!r} must contain only boolean values.")

    for group_name, group in data.groupby(group_column, sort=False, dropna=False):
        true_positions = [position for position, value in enumerate(group[marker_column].tolist()) if value]
        if true_positions != [len(group) - 1]:
            raise DataError(f"Last-marker group {group_name!r} must contain exactly one true marker on its final row.")


def validate_prepared_flexible_timeseries_data(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
    group_column: str,
) -> None:
    """Fail closed when prepared flexible time-series rows violate invariants."""
    if data.empty:
        raise DataError("Prepared flexible time-series data must contain at least one row.")
    check_groupby_column(data, group_column)
    _validate_common_columns(data, config, group_column)

    if not config.time_series.flexible_timeseries:
        raise ParameterError("Prepared flexible time-series validation requires automatic flexible routing.")
    _validate_last_marker_data(data, config, group_column)

    max_records = config.time_series.sequence_max_records
    observed_max = int(data.groupby(group_column, sort=False).size().max())
    if max_records != observed_max:
        raise DataError(
            f"sequence_max_records must equal the dataset-level maximum group length {observed_max}; "
            f"got {max_records!r}."
        )


def _reorder_prepared_columns(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
    group_column: str,
) -> pd.DataFrame:
    ts_config = config.time_series
    source_columns = ts_config.sequence_source_columns or []
    payload_columns = [column for column in source_columns if column != group_column]
    ordered = [group_column, ts_config.sequence_index_column, *payload_columns, ts_config.sequence_marker_column]
    return data.loc[:, ordered]


def _prepare_raw_sequence_data(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
) -> tuple[pd.DataFrame, str]:
    ts_config = config.time_series
    source_columns = list(data.columns)
    working, group_column = _resolve_group(data, config)

    order_column = _source_order_column(working, config)
    temporary_columns: list[str] = []
    source_position_column = _unused_column_name("__nss_source_position", list(working.columns))
    working[source_position_column] = range(len(working))
    temporary_columns.append(source_position_column)

    sort_order_column = order_column
    if order_column is not None and order_column == ts_config.timestamp_column:
        sort_order_column = _unused_column_name("__nss_source_order", list(working.columns))
        working[sort_order_column] = _timestamp_sort_key(working, config, order_column)
        temporary_columns.append(sort_order_column)

    sort_columns = [group_column]
    if sort_order_column is not None and sort_order_column != group_column:
        sort_columns.append(sort_order_column)
    sort_columns.append(source_position_column)
    working = working.sort_values(sort_columns, kind="mergesort").drop(columns=temporary_columns)
    working = working.reset_index(drop=True)

    index_column = _unused_column_name(ts_config.sequence_index_column, list(working.columns))
    ts_config.sequence_index_column = index_column
    working[index_column] = working.groupby(group_column, sort=False).cumcount()

    observed_max = int(working.groupby(group_column, sort=False).size().max())
    if ts_config.sequence_max_records is not None and ts_config.sequence_max_records != observed_max:
        raise ParameterError(
            f"sequence_max_records must match the observed dataset maximum {observed_max}; "
            f"got {ts_config.sequence_max_records}."
        )
    ts_config.sequence_max_records = observed_max
    ts_config.sequence_source_columns = source_columns

    marker_column = _unused_column_name(ts_config.sequence_marker_column, list(working.columns))
    ts_config.sequence_marker_column = marker_column
    working[marker_column] = False
    final_indices = working.groupby(group_column, sort=False).tail(1).index
    working.loc[final_indices, marker_column] = True

    return _reorder_prepared_columns(working, config, group_column), group_column


def prepare_flexible_timeseries_data(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
) -> tuple[pd.DataFrame, str]:
    """Prepare raw data or normalize an already prepared flexible time-series table."""
    ts_config = config.time_series
    source_columns = ts_config.sequence_source_columns
    index_column = ts_config.sequence_index_column

    if index_column in data.columns and source_columns is not None:
        expected_columns = {*source_columns, index_column, ts_config.sequence_marker_column}
        if set(data.columns) != expected_columns:
            raise DataError(
                "Prepared flexible time-series columns must contain exactly the configured source columns and "
                "resolved control columns."
            )
        working, group_column = _resolve_group(data, config)
        observed_max = int(working.groupby(group_column, sort=False).size().max())
        if ts_config.sequence_max_records is None:
            ts_config.sequence_max_records = observed_max
        working = _reorder_prepared_columns(working, config, group_column)
    else:
        working, group_column = _prepare_raw_sequence_data(data, config)

    ts_config.timestamp_column = ts_config.sequence_index_column
    ts_config.timestamp_format = "elapsed_seconds"
    ts_config.timestamp_interval_seconds = 1
    ts_config.start_timestamp = 0
    ts_config.stop_timestamp = (ts_config.sequence_max_records or 1) - 1
    config.data.group_training_examples_by = group_column
    config.data.order_training_examples_by = ts_config.sequence_index_column
    validate_prepared_flexible_timeseries_data(working, config, group_column)
    return working, group_column
