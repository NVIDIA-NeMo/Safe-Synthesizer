# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared column-validation primitives and compound checks.

Two layers:

* Single-purpose primitives (``check_column_present``,
  ``check_column_has_no_nulls``, ``check_no_pseudo_column_collision``)
  raise one specific error for one specific failure mode. Preflight
  checks call these directly so each ``collector.error("code", ...)``
  sits next to the check that owns the issue code -- no shared
  exception-to-code mapping.
* Compound checks (``check_groupby_column``, ``check_orderby_column``,
  ``check_timestamp_column``) are 2-3 line orchestrations of the
  primitives. They give non-preflight callers (SDK pipeline, holdout,
  assembler, time-series preprocessing) a single fail-fast gate per
  column.
"""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd

from ..defaults import PSEUDO_GROUP_COLUMN
from ..errors import DataError, ParameterError


def _get_column_names(data: pd.DataFrame | Collection[str]) -> Collection[str]:
    if isinstance(data, pd.DataFrame):
        _reject_multiindex_columns(data)
        return data.columns
    return data


def _reject_multiindex_columns(data: pd.DataFrame) -> None:
    """Fail fast on ``MultiIndex`` columns.

    Flat-string primitives can't meaningfully answer "is this column
    present?" or "does this column have nulls?" for a MultiIndex schema
    -- ``"a" in MultiIndex[("a","b"),("a","c")]`` is False even though
    level 0 contains ``"a"``. Rather than silently miss collisions or
    mis-report missing columns, require the caller to flatten the
    schema first.
    """
    if isinstance(data.columns, pd.MultiIndex):
        raise ParameterError(
            "MultiIndex column DataFrames are not supported by column validators; "
            "flatten the column schema (e.g. join levels with '_') before calling."
        )


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def check_column_present(
    data: pd.DataFrame | Collection[str],
    column: str,
    *,
    role: str,
    hint: str | None = None,
) -> None:
    """Raise ``ParameterError`` if ``column`` is not in ``data``'s columns.

    ``role`` is an English label for the column's purpose (e.g.
    ``"Group by"``, ``"Order by"``, ``"Timestamp"``) used in the error
    message. ``hint`` is an optional trailing sentence callers can use
    to tell the user how to resolve the error (e.g. "Please set X to
    null to disable Y."). The ``"Group by"`` role has a built-in hint
    for comma-in-name that always takes precedence over ``hint``.
    """
    columns = _get_column_names(data)
    if column in columns:
        return

    message = f"{role} column '{column}' not found in input dataset columns."
    if role == "Group by" and "," in column:
        message += (
            " The column name contains a comma -- multi-column grouping is not supported. Use a single column name."
        )
    elif hint:
        message += f" {hint}"
    raise ParameterError(message)


def check_column_has_no_nulls(
    data: pd.DataFrame | Collection[str],
    column: str,
    *,
    role: str,
) -> None:
    """Raise ``DataError`` if ``column`` contains any null values.

    Only inspects row contents when ``data`` is a DataFrame; pure
    column-name collections are treated as a no-op.
    """
    if not isinstance(data, pd.DataFrame):
        return
    _reject_multiindex_columns(data)
    if data[column].isna().any():
        raise DataError(f"{role} column '{column}' has missing values. Please remove/replace them.")


def check_no_pseudo_column_collision(data: pd.DataFrame | Collection[str]) -> None:
    """Raise ``DataError`` if the reserved pseudo-group column is in use."""
    columns = _get_column_names(data)
    if PSEUDO_GROUP_COLUMN in columns:
        raise DataError(
            f"Column '{PSEUDO_GROUP_COLUMN}' is reserved for internal use. Please rename this column in your data."
        )


# ---------------------------------------------------------------------------
# Compound checks -- thin orchestrations for non-preflight callers.
# ---------------------------------------------------------------------------


def check_groupby_column(data: pd.DataFrame | Collection[str], group_by: str | None) -> None:
    """Validate the configured group-by column exists and has no missing values.

    Raises:
        ParameterError: If ``group_by`` is configured but not present in ``data``.
        DataError: If ``data`` is a DataFrame and ``group_by`` contains missing values.
    """
    if group_by is None:
        return
    check_column_present(
        data,
        group_by,
        role="Group by",
        hint="Please set `data.group_training_examples_by` to an existing column or to `null`/`None` to disable grouping.",
    )
    check_column_has_no_nulls(data, group_by, role="Group by")


def check_orderby_column(
    data: pd.DataFrame | Collection[str],
    order_by: str | None,
    *,
    is_timeseries: bool = False,
    timestamp_column: str | None = None,
) -> None:
    """Validate the configured order-by column exists.

    In time-series mode without an explicit timestamp column, ordering
    is deferred until preprocessing synthesizes a timestamp, so this
    check is skipped.

    Raises:
        ParameterError: If ``order_by`` is configured but not present in ``data``.
    """
    if order_by is None:
        return
    if is_timeseries and timestamp_column is None:
        return
    check_column_present(data, order_by, role="Order by")


def check_timestamp_column(data: pd.DataFrame | Collection[str], timestamp_column: str) -> None:
    """Validate the configured timestamp column exists and has no missing values.

    Raises:
        ParameterError: If ``timestamp_column`` is not present in ``data``.
        DataError: If ``data`` is a DataFrame and ``timestamp_column`` contains missing values.
    """
    check_column_present(data, timestamp_column, role="Timestamp")
    check_column_has_no_nulls(data, timestamp_column, role="Timestamp")
