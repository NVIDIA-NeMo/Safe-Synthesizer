# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data validation helpers shared across pipeline stages."""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd

from ..errors import DataError, ParameterError

MISSING_GROUP_BY_COLUMN_ERROR = "Group by column '{group_by}' not found in input dataset columns. "
MISSING_GROUP_BY_VALUES_ERROR = "Group by column '{group_by}' has missing values. Please remove/replace them."
MISSING_ORDER_BY_COLUMN_ERROR = "Order by column '{order_by}' not found in the input data."


def _get_column_names(data: pd.DataFrame |Collection[str]) -> Collection[str]:
    if isinstance(data, pd.DataFrame):
        return data.columns
    return data


def validate_groupby_column(data: pd.DataFrame |Collection[str], group_by: str | None) -> None:
    """Validate that the configured group-by column exists and has no missing values.

    Args:
        data: A DataFrame or collection of column names to validate against.
        group_by: Name of the configured grouping column.

    Raises:
        ParameterError: If ``group_by`` is configured but not present in ``data``.
        DataError: If ``data`` is a DataFrame and ``group_by`` contains missing values.
    """
    if group_by is None:
        return

    columns = _get_column_names(data)

    if group_by not in columns:
        message = MISSING_GROUP_BY_COLUMN_ERROR.format(group_by=group_by)
        if "," in group_by:
            message += (
                " The column name contains a comma -- multi-column grouping is not supported. Use a single column name."
            )
        else:
            message += (
                " Please set `data.group_training_examples_by` to an existing column or to `null`/`None` to disable grouping."
            )
        raise ParameterError(message)

    if isinstance(data, pd.DataFrame) and data[group_by].isna().any():
        raise DataError(MISSING_GROUP_BY_VALUES_ERROR.format(group_by=group_by))


def validate_orderby_column(data: pd.DataFrame |Collection[str], order_by: str | None) -> None:
    """Validate that the configured order-by column exists.

    Args:
        data: A DataFrame or collection of column names to validate against.
        order_by: Name of the configured ordering column.

    Raises:
        ParameterError: If ``order_by`` is configured but not present in ``data``.
    """
    if order_by is None:
        return

    columns = _get_column_names(data)

    if order_by not in columns:
        raise ParameterError(MISSING_ORDER_BY_COLUMN_ERROR.format(order_by=order_by))
