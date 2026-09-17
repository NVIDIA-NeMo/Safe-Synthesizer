# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation and validation for automatically routed flexible time series."""

from __future__ import annotations

import pandas as pd

from ..config.parameters import SafeSynthesizerParameters
from ..config.time_series import FlexibleTimeseriesMetadata
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
    "resolve_flexible_timeseries_metadata",
]


def _unused_column_name(preferred: str, columns: list[str]) -> str:
    """Return the preferred column name or a deterministic suffixed variant."""
    if preferred not in columns:
        return preferred
    suffix = 1
    while f"{preferred}_{suffix}" in columns:
        suffix += 1
    return f"{preferred}_{suffix}"


def resolve_flexible_timeseries_metadata(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
    max_records: int,
) -> FlexibleTimeseriesMetadata:
    """Resolve internal control columns and source schema for flexible routing."""
    if data.columns.has_duplicates:
        duplicates = data.columns[data.columns.duplicated()].unique().tolist()
        raise DataError(
            f"Flexible time-series input contains duplicate column names {duplicates!r}. "
            "Rename or remove duplicate columns before running the pipeline."
        )

    columns = list(data.columns)
    if config.data.group_training_examples_by is None:
        check_no_pseudo_column_collision(data)
        columns.append(PSEUDO_GROUP_COLUMN)

    index_column = _unused_column_name(FlexibleTimeseriesMetadata.DEFAULT_INDEX_COLUMN, columns)
    marker_column = _unused_column_name(
        FlexibleTimeseriesMetadata.DEFAULT_MARKER_COLUMN,
        [*columns, index_column],
    )
    return FlexibleTimeseriesMetadata(
        index_column=index_column,
        marker_column=marker_column,
        max_records=max_records,
        source_columns=tuple(data.columns),
    )


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


def prepare_flexible_timeseries_data(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
) -> tuple[pd.DataFrame, str]:
    """Transform raw source data into the internal flexible time-series representation."""
    ts_config = config.time_series
    metadata = ts_config.flexible_timeseries_metadata
    if metadata is None:
        raise ParameterError("Flexible time-series preparation requires resolved internal metadata.")

    group_column = config.data.group_training_examples_by
    working = data.copy()
    if group_column is None:
        check_no_pseudo_column_collision(working)
        group_column = PSEUDO_GROUP_COLUMN
        working[group_column] = 0
    else:
        check_groupby_column(working, group_column)

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
    working = working.sort_values(sort_columns, kind="mergesort").drop(columns=temporary_columns).reset_index(drop=True)

    working[metadata.index_column] = working.groupby(group_column, sort=False).cumcount()
    working[metadata.marker_column] = False
    final_indices = working.groupby(group_column, sort=False).tail(1).index
    working.loc[final_indices, metadata.marker_column] = True

    payload_columns = [column for column in metadata.source_columns if column != group_column]
    ordered_columns = [group_column, metadata.index_column, *payload_columns, metadata.marker_column]
    working = working.loc[:, ordered_columns]

    ts_config.timestamp_column = metadata.index_column
    ts_config.timestamp_format = "elapsed_seconds"
    ts_config.timestamp_interval_seconds = 1
    ts_config.start_timestamp = 0
    ts_config.stop_timestamp = metadata.max_records - 1
    config.data.group_training_examples_by = group_column
    config.data.order_training_examples_by = metadata.index_column
    return working, group_column
