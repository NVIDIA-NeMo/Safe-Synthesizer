# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DataFrame-stage checks that inspect the training split's columns."""

from __future__ import annotations

from ...data_processing.validation import (
    check_column_has_no_nulls,
    check_column_present,
    check_no_pseudo_column_collision,
)
from ...errors import DataError, ParameterError
from ..base import DataFrameCheck
from ..helpers import emit_on_raise
from ..base import IssueCollector
from ..types import PreflightContext, DataFrameView

__all__ = [
    "ConstantColumnCheck",
    "GroupbyColumnCheck",
    "OrderbyColumnCheck",
    "PseudoColumnCheck",
    "TimestampColumnCheck",
]


class GroupbyColumnCheck(DataFrameCheck):
    """Validate group-by column existence and integrity."""

    name = "columns.groupby"
    label = "Group-by column"

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

    def check(self, ctx: DataFrameView, collector: IssueCollector) -> None:
        emit_on_raise(
            collector,
            lambda: check_no_pseudo_column_collision(ctx.data),
            expect=DataError,
            code="pseudo_column_collision",
        )


class ConstantColumnCheck(DataFrameCheck):
    """Warn about columns with a single unique value."""

    name = "columns.constant"
    label = "Constant columns"

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

    def enabled(self, ctx: PreflightContext) -> bool:
        if not super().enabled(ctx):
            return False
        return bool(ctx.config.time_series.is_timeseries)

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
