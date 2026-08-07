# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DataFrame-stage checks that inspect the training split's columns."""

from __future__ import annotations

from typing_extensions import override

from ...data_processing.timeseries_validation import (
    TimeSeriesDataValidationError,
    TimeSeriesParameterValidationError,
    TimeSeriesValidationReason,
    validate_timeseries_data,
)
from ...data_processing.validation import (
    check_column_has_no_nulls,
    check_column_present,
    check_no_pseudo_column_collision,
)
from ...errors import DataError, ParameterError
from ..base import DataFrameCheck, IssueCollector
from ..helpers import emit_on_raise
from ..types import DataFrameView, PreflightContext

__all__ = [
    "ConstantColumnCheck",
    "DatasetSizeCheck",
    "GroupbyColumnCheck",
    "OrderbyColumnCheck",
    "PseudoColumnCheck",
    "TimeSeriesDataShapeCheck",
    "TimestampColumnCheck",
]


class DatasetSizeCheck(DataFrameCheck):
    """Block training when the training split is unusably small.

    An empty / near-empty split would produce cascading failures in
    downstream token-budget and metadata checks; emitting the error here
    lets the orchestrator gate them cleanly via ``requires``.
    """

    name = "dataset.size"
    label = "Dataset size"
    category = "data quality"
    min_rows: int = 200

    @override
    def check(self, ctx: DataFrameView, collector: IssueCollector) -> None:
        n_rows = len(ctx.data)
        if n_rows < self.min_rows:
            collector.error(
                "dataset_too_small",
                f"Training split has {n_rows} rows; at least {self.min_rows} are needed for meaningful training.",
            )


class GroupbyColumnCheck(DataFrameCheck):
    """Validate group-by column existence and integrity."""

    name = "columns.groupby"
    label = "Group-by column"

    @override
    def check(self, ctx: DataFrameView, collector: IssueCollector) -> None:
        column = ctx.config.data.group_training_examples_by
        if column is None:
            return
        present = emit_on_raise(
            collector,
            lambda: check_column_present(ctx.data, column, role="Group by"),
            expect=ParameterError,
            code="column_not_found",
        )
        if not present:
            return
        emit_on_raise(
            collector,
            lambda: check_column_has_no_nulls(ctx.data, column, role="Group by"),
            expect=DataError,
            code="column_nulls",
        )


class OrderbyColumnCheck(DataFrameCheck):
    """Validate order-by column existence."""

    name = "columns.orderby"
    label = "Order-by column"

    @override
    def check(self, ctx: DataFrameView, collector: IssueCollector) -> None:
        config = ctx.config
        column = config.data.order_training_examples_by
        if column is None:
            return
        # Time-series mode without an explicit timestamp column defers
        # ordering until preprocessing synthesizes a timestamp, so there
        # is nothing to validate here yet.
        if config.time_series.is_timeseries and config.time_series.timestamp_column is None:
            return
        emit_on_raise(
            collector,
            lambda: check_column_present(ctx.data, column, role="Order by"),
            expect=ParameterError,
            code="column_not_found",
        )


class PseudoColumnCheck(DataFrameCheck):
    """Detect collision with the reserved pseudo group column."""

    name = "columns.pseudo"
    label = "Pseudo column"

    @override
    def check(self, ctx: DataFrameView, collector: IssueCollector) -> None:
        emit_on_raise(
            collector,
            lambda: check_no_pseudo_column_collision(ctx.data),
            expect=(DataError, ParameterError),
            code="pseudo_column_collision",
        )


class ConstantColumnCheck(DataFrameCheck):
    """Warn about columns with a single unique value."""

    name = "columns.constant"
    label = "Constant columns"

    @override
    def check(self, ctx: DataFrameView, collector: IssueCollector) -> None:
        data = ctx.data
        for col in data.columns:
            if data[col].dropna().nunique() == 1:
                collector.warning(
                    "constant_column",
                    f"Column '{col}' has only 1 unique value — consider dropping it or verifying the data.",
                )


class TimestampColumnCheck(DataFrameCheck):
    """Validate time-series timestamp column presence and integrity."""

    name = "timeseries.timestamp"
    label = "Timestamp column"

    @override
    def enabled(self, ctx: PreflightContext) -> bool:
        if not super().enabled(ctx):
            return False
        return bool(ctx.config.time_series.is_timeseries)

    @override
    def check(self, ctx: DataFrameView, collector: IssueCollector) -> None:
        timestamp_column = ctx.config.time_series.timestamp_column
        if timestamp_column is None:
            return
        present = emit_on_raise(
            collector,
            lambda: check_column_present(ctx.data, timestamp_column, role="Timestamp"),
            expect=ParameterError,
            code="timestamp_not_found",
        )
        if not present:
            return
        emit_on_raise(
            collector,
            lambda: check_column_has_no_nulls(ctx.data, timestamp_column, role="Timestamp"),
            expect=DataError,
            code="timestamp_nulls",
        )


class TimeSeriesDataShapeCheck(DataFrameCheck):
    """Validate time-series timestamp format and per-group shape invariants."""

    name = "timeseries.shape"
    label = "Time-series data shape"
    requires = ("columns.groupby", "columns.pseudo")
    issue_codes = {
        TimeSeriesValidationReason.COLUMN_NOT_FOUND: "column_not_found",
        TimeSeriesValidationReason.COLUMN_NULLS: "column_nulls",
        TimeSeriesValidationReason.PSEUDO_COLUMN_COLLISION: "pseudo_column_collision",
        TimeSeriesValidationReason.TIMESTAMP_NOT_FOUND: "timestamp_not_found",
        TimeSeriesValidationReason.TIMESTAMP_NULLS: "timestamp_nulls",
        TimeSeriesValidationReason.TIMESTAMP_FORMAT_MISMATCH: "timestamp_format_mismatch",
        TimeSeriesValidationReason.TIMESTAMP_PARSE_FAILED: "timestamp_parse_failed",
        TimeSeriesValidationReason.TIMESTAMP_ELAPSED_NON_NUMERIC: "timestamp_elapsed_non_numeric",
        TimeSeriesValidationReason.TIMESTAMP_ELAPSED_INVALID: "timestamp_elapsed_invalid",
        TimeSeriesValidationReason.TIMESTAMP_INTERVAL_MISMATCH: "timestamp_interval_mismatch",
        TimeSeriesValidationReason.TIMESERIES_EMPTY: "timeseries_empty",
        TimeSeriesValidationReason.TIMESERIES_NO_VALUE_COLUMNS: "timeseries_no_value_columns",
        TimeSeriesValidationReason.TIMESERIES_GROUP_LENGTH_MISMATCH: "timeseries_group_length_mismatch",
        TimeSeriesValidationReason.TIMESERIES_START_MISMATCH: "timeseries_start_mismatch",
        TimeSeriesValidationReason.TIMESERIES_STOP_MISMATCH: "timeseries_stop_mismatch",
    }

    @override
    def enabled(self, ctx: PreflightContext) -> bool:
        if not super().enabled(ctx):
            return False
        if not ctx.config.time_series.is_timeseries:
            return False
        timestamp_column = ctx.config.time_series.timestamp_column
        if timestamp_column is not None:
            try:
                check_column_present(ctx.data, timestamp_column, role="Timestamp")
                check_column_has_no_nulls(ctx.data, timestamp_column, role="Timestamp")
            except (DataError, ParameterError):
                return False
        return True

    @override
    def check(self, ctx: DataFrameView, collector: IssueCollector) -> None:
        timestamp_column = ctx.config.time_series.timestamp_column
        if timestamp_column is not None:
            try:
                check_column_present(ctx.data, timestamp_column, role="Timestamp")
                check_column_has_no_nulls(ctx.data, timestamp_column, role="Timestamp")
            except (DataError, ParameterError):
                return

        try:
            validate_timeseries_data(ctx.data, ctx.config)
        except (TimeSeriesDataValidationError, TimeSeriesParameterValidationError) as exc:
            collector.error(self.issue_codes[exc.reason], str(exc))
