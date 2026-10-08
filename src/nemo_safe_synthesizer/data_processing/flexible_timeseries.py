# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation and validation for automatically routed flexible time series."""

from __future__ import annotations

import pandas as pd

from ..config.parameters import SafeSynthesizerParameters
from ..defaults import PSEUDO_GROUP_COLUMN
from ..llm.metadata import FlexibleTimeseriesMetadata
from .timeseries_utils import stable_sort_within_groups, unused_column_name
from .validation import (
    check_column_has_no_nulls,
    check_column_present,
    check_groupby_column,
    check_no_pseudo_column_collision,
    check_timestamp_column,
)

__all__ = [
    "finalize_flexible_timeseries_controls",
    "prepare_flexible_timeseries_data",
    "resolve_flexible_timeseries_metadata",
]


def resolve_flexible_timeseries_metadata(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
    max_records: int,
    source_timestamp_format: str,
) -> FlexibleTimeseriesMetadata:
    """Resolve internal control columns and timestamp checks for flexible routing.

    Args:
        data: Source time-series data.
        config: Parameters with the effective source timestamp column already resolved.
        max_records: Largest source-group length.
        source_timestamp_format: Validated or inferred format of the source timestamp column.

    Returns:
        Flexible controls persisted with the model and used by generation.
    """
    columns = list(data.columns)
    if config.data.group_training_examples_by is None:
        check_no_pseudo_column_collision(data)
        columns.append(PSEUDO_GROUP_COLUMN)

    index_column = unused_column_name(FlexibleTimeseriesMetadata.DEFAULT_INDEX_COLUMN, columns)
    marker_column = unused_column_name(
        FlexibleTimeseriesMetadata.DEFAULT_MARKER_COLUMN,
        [*columns, index_column],
    )
    source_timestamp_column = config.time_series.timestamp_column
    has_source_timestamp = source_timestamp_column is not None
    return FlexibleTimeseriesMetadata(
        index_column=index_column,
        marker_column=marker_column,
        max_records=max_records,
        source_timestamp_column=source_timestamp_column,
        source_timestamp_format=source_timestamp_format if has_source_timestamp else None,
        source_interval_seconds=config.time_series.timestamp_interval_seconds if has_source_timestamp else None,
    )


def _source_order_column(data: pd.DataFrame, config: SafeSynthesizerParameters) -> str | None:
    """Resolve the source column used for deterministic within-group ordering."""
    timestamp_column = config.time_series.timestamp_column
    if timestamp_column is not None:
        check_timestamp_column(data, timestamp_column)
        return timestamp_column

    order_column = config.data.order_training_examples_by
    if order_column is not None:
        check_column_present(data, order_column, role="Order by")
        check_column_has_no_nulls(data, order_column, role="Order by")
    return order_column


def prepare_flexible_timeseries_data(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
    metadata: FlexibleTimeseriesMetadata,
    source_timestamp_format: str,
) -> tuple[pd.DataFrame, str]:
    """Transform source data into the internal flexible representation.

    Rows are stably ordered within each group before the sequence index and
    final-row marker are added. The resolved configuration is updated to use
    the generated index as its generation-time timestamp.

    Args:
        data: Source time-series data.
        config: Parameters to update with the resolved internal columns.
        metadata: Resolved control-column names and sequence-length cap.
        source_timestamp_format: Format used to normalize timestamp ordering.

    Returns:
        The transformed data and effective group-column name.
    """
    ts_config = config.time_series

    group_column = config.data.group_training_examples_by
    working = data.copy()
    if group_column is None:
        check_no_pseudo_column_collision(working)
        group_column = PSEUDO_GROUP_COLUMN
        working[group_column] = 0
    else:
        check_groupby_column(working, group_column)

    order_column = _source_order_column(working, config)
    normalized_order = None
    if (
        order_column is not None
        and order_column == ts_config.timestamp_column
        and source_timestamp_format != "elapsed_seconds"
    ):
        normalized_order = pd.to_datetime(
            working[order_column],
            format=source_timestamp_format,
        )
    working = stable_sort_within_groups(
        working,
        group_column,
        order_column,
        normalized_order=normalized_order,
    )

    _assign_sequence_controls(working, group_column, metadata)

    payload_columns = [column for column in data.columns if column != group_column]
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


def _assign_sequence_controls(data: pd.DataFrame, group_column: str, metadata: FlexibleTimeseriesMetadata) -> None:
    """Number rows within each group in their current order and mark each group's final row, in place."""
    data[metadata.index_column] = data.groupby(group_column, sort=False).cumcount()
    data[metadata.marker_column] = False
    final_indices = data.groupby(group_column, sort=False).tail(1).index
    data.loc[final_indices, metadata.marker_column] = True


def finalize_flexible_timeseries_controls(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
    metadata: FlexibleTimeseriesMetadata,
) -> tuple[pd.DataFrame, FlexibleTimeseriesMetadata]:
    """Recompute flexible control columns after training-time preprocessing.

    Preprocessing actions can drop rows after the sequence index and final-row
    marker were assigned, which would leave index gaps or groups without a
    final-row marker. Indices and markers are reassigned in the current row
    order, and the record cap is updated to the longest remaining group.

    Args:
        data: Preprocessed training data in the flexible representation.
        config: Resolved parameters; the generation stop timestamp is updated.
        metadata: Flexible metadata resolved before preprocessing.

    Returns:
        The data with consistent control columns and the updated metadata.
    """
    group_column = config.data.group_training_examples_by
    if data.empty or group_column is None:
        return data, metadata
    working = data.reset_index(drop=True)
    _assign_sequence_controls(working, group_column, metadata)
    max_records = int(working.groupby(group_column, sort=False).size().max())
    config.time_series.stop_timestamp = max_records - 1
    return working, metadata.model_copy(update={"max_records": max_records})
