# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation and validation for automatically routed flexible time series."""

from __future__ import annotations

from typing import cast

import pandas as pd

from ..config.parameters import SafeSynthesizerParameters
from ..config.time_series import FlexibleTimeseriesMetadata
from ..defaults import PSEUDO_GROUP_COLUMN
from .timeseries_utils import stable_sort_within_groups, unused_column_name
from .validation import (
    check_column_has_no_nulls,
    check_column_present,
    check_groupby_column,
    check_no_pseudo_column_collision,
    check_timestamp_column,
)

__all__ = [
    "prepare_flexible_timeseries_data",
    "resolve_flexible_timeseries_metadata",
]


def resolve_flexible_timeseries_metadata(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
    max_records: int,
) -> FlexibleTimeseriesMetadata:
    """Resolve internal control columns and source schema for flexible routing."""
    columns = list(data.columns)
    if config.data.group_training_examples_by is None:
        check_no_pseudo_column_collision(data)
        columns.append(PSEUDO_GROUP_COLUMN)

    index_column = unused_column_name(FlexibleTimeseriesMetadata.DEFAULT_INDEX_COLUMN, columns)
    marker_column = unused_column_name(
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
    order_column = config.data.order_training_examples_by
    if order_column is not None:
        check_column_present(data, order_column, role="Order by")
        check_column_has_no_nulls(data, order_column, role="Order by")
        return order_column

    timestamp_column = config.time_series.timestamp_column
    if timestamp_column is not None:
        check_timestamp_column(data, timestamp_column)
        return timestamp_column
    return None


def prepare_flexible_timeseries_data(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
    metadata: FlexibleTimeseriesMetadata,
) -> tuple[pd.DataFrame, str]:
    """Transform raw source data into the internal flexible time-series representation."""
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
        and ts_config.timestamp_format != "elapsed_seconds"
    ):
        normalized_order = pd.to_datetime(
            working[order_column],
            format=cast(str, ts_config.timestamp_format),
        )
    working = stable_sort_within_groups(
        working,
        group_column,
        order_column,
        normalized_order=normalized_order,
    )

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
