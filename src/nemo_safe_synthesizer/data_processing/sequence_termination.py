# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation and validation for sequence-termination experiments."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from ..config.parameters import SafeSynthesizerParameters
from ..defaults import PSEUDO_GROUP_COLUMN
from ..errors import DataError, ParameterError
from .validation import check_groupby_column, check_no_pseudo_column_collision, check_timestamp_column

__all__ = [
    "is_padding_terminal",
    "prepare_sequence_termination_data",
    "sequence_schema_dataframe",
    "validate_prepared_sequence_data",
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
            check_timestamp_column(data, candidate)
            return candidate
    return None


def _payload_columns(config: SafeSynthesizerParameters, group_column: str) -> list[str]:
    source_columns = config.time_series.sequence_source_columns or []
    return [column for column in source_columns if column != group_column]


def is_padding_terminal(
    record: Mapping[str, object],
    *,
    group_column: str,
    index_column: str,
    source_columns: list[str],
) -> bool:
    """Return whether a record is the exact full-null padding terminal shape."""
    payload_columns = [column for column in source_columns if column != group_column]
    expected_columns = {group_column, index_column, *payload_columns}
    return (
        set(record) == expected_columns
        and record.get(group_column) is not None
        and record.get(index_column) is not None
        and all(record[column] is None for column in payload_columns)
    )


def _is_padding_row(row: pd.Series, payload_columns: list[str]) -> bool:
    """Return whether every source payload value in a prepared row is null."""
    return bool(payload_columns) and bool(row[payload_columns].isna().all())


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


def _validate_padding_data(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
    group_column: str,
) -> None:
    ts_config = config.time_series
    marker_column = ts_config.sequence_marker_column
    if marker_column in data.columns:
        raise DataError(f"Padding-mode prepared data must not contain marker column {marker_column!r}.")

    payload_columns = _payload_columns(config, group_column)
    if not payload_columns:
        raise DataError("Padding-mode prepared data requires at least one source payload column.")

    max_records = ts_config.sequence_max_records
    if max_records is None:
        raise ParameterError("sequence_max_records must be resolved before validating prepared data.")

    for group_name, group in data.groupby(group_column, sort=False, dropna=False):
        if len(group) != max_records:
            raise DataError(
                f"Padding-mode group {group_name!r} has {len(group)} rows; expected sequence_max_records={max_records}."
            )
        padding_started = False
        for _, row in group.iterrows():
            is_padding = _is_padding_row(row, payload_columns)
            if padding_started and not is_padding:
                raise DataError(f"Padding-mode group {group_name!r} contains a real row after terminal padding.")
            padding_started = padding_started or is_padding


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
            raise DataError(
                f"Last-marker group {group_name!r} must contain exactly one true marker on its final row."
            )


def validate_prepared_sequence_data(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
    group_column: str,
) -> None:
    """Fail closed when prepared experiment rows violate sequence invariants."""
    if data.empty:
        raise DataError("Prepared sequence-termination data must contain at least one row.")
    check_groupby_column(data, group_column)
    _validate_common_columns(data, config, group_column)

    mode = config.time_series.sequence_termination_mode
    if mode == "idx_padding":
        _validate_padding_data(data, config, group_column)
    elif mode == "idx_last":
        _validate_last_marker_data(data, config, group_column)
    else:
        raise ParameterError("Prepared sequence validation requires an experiment termination mode.")

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
    ordered = [group_column, ts_config.sequence_index_column, *payload_columns]
    if ts_config.sequence_termination_mode == "idx_last":
        ordered.append(ts_config.sequence_marker_column)
    return data.loc[:, ordered]


def _prepare_raw_sequence_data(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
) -> tuple[pd.DataFrame, str]:
    ts_config = config.time_series
    source_columns = list(data.columns)
    working, group_column = _resolve_group(data, config)

    if ts_config.sequence_termination_mode == "idx_padding":
        payload_columns = [column for column in source_columns if column != group_column]
        if payload_columns and working[payload_columns].isna().all(axis=1).any():
            raise DataError(
                "idx_padding source data contains a real row with a fully null non-group payload; "
                "it is indistinguishable from terminal padding."
            )

    order_column = _source_order_column(working, config)
    working["__nss_source_position"] = range(len(working))
    sort_columns = [group_column]
    if order_column is not None and order_column != group_column:
        sort_columns.append(order_column)
    sort_columns.append("__nss_source_position")
    working = working.sort_values(sort_columns, kind="mergesort").drop(columns=["__nss_source_position"])
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

    if ts_config.sequence_termination_mode == "idx_padding":
        payload_columns = [column for column in source_columns if column != group_column]
        prepared_groups: list[pd.DataFrame] = []
        for group_value, group in working.groupby(group_column, sort=False, dropna=False):
            group = group.copy()
            missing = observed_max - len(group)
            if missing:
                padding = pd.DataFrame(
                    [
                        {
                            group_column: group_value,
                            index_column: len(group) + offset,
                            **{column: None for column in payload_columns},
                        }
                        for offset in range(missing)
                    ]
                )
                group = pd.concat([group, padding], ignore_index=True)
            prepared_groups.append(group)
        working = pd.concat(prepared_groups, ignore_index=True)
    else:
        marker_column = _unused_column_name(ts_config.sequence_marker_column, list(working.columns))
        ts_config.sequence_marker_column = marker_column
        working[marker_column] = False
        final_indices = working.groupby(group_column, sort=False).tail(1).index
        working.loc[final_indices, marker_column] = True

    return _reorder_prepared_columns(working, config, group_column), group_column


def prepare_sequence_termination_data(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
) -> tuple[pd.DataFrame, str]:
    """Prepare raw data or normalize an already prepared experiment table."""
    ts_config = config.time_series
    source_columns = ts_config.sequence_source_columns
    index_column = ts_config.sequence_index_column
    is_prepared = index_column in data.columns and source_columns is not None

    if is_prepared:
        expected_columns = {*source_columns, index_column}
        if ts_config.sequence_termination_mode == "idx_last":
            expected_columns.add(ts_config.sequence_marker_column)
        if set(data.columns) != expected_columns:
            raise DataError(
                "Prepared sequence-termination columns must contain exactly the configured source columns and "
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
    validate_prepared_sequence_data(working, config, group_column)
    return working, group_column


def sequence_schema_dataframe(data: pd.DataFrame, config: SafeSynthesizerParameters) -> pd.DataFrame:
    """Return real prepared rows for strict schema inference."""
    if config.time_series.sequence_termination_mode != "idx_padding":
        return data
    group_column = config.data.group_training_examples_by
    if group_column is None:
        raise ParameterError("Padding schema inference requires a resolved group column.")
    payload_columns = _payload_columns(config, group_column)
    padding_mask = data[payload_columns].isna().all(axis=1)
    return data.loc[~padding_mask].copy()
