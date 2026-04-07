# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data validation helpers shared across pipeline stages."""

from __future__ import annotations

import pandas as pd

from ..errors import DataError, ParameterError

MISSING_GROUP_BY_COLUMN_ERROR = (
    "Group by column '{group_by}' not found in input dataset columns. "
    "Please set `data.group_training_examples_by` to an existing column or to `null`/`None` to disable grouping."
)
MISSING_GROUP_BY_VALUES_ERROR = "Group by column '{group_by}' has missing values. Please remove/replace them."
MISSING_ORDER_BY_COLUMN_ERROR = "Order by column '{order_by}' not found in the input data."


def validate_groupby_column(df: pd.DataFrame, group_by: str | None) -> None:
    """Validate that the configured group-by column exists and has no missing values.

    Args:
        df: Dataframe to validate.
        group_by: Name of the configured grouping column.

    Raises:
        ParameterError: If ``group_by`` is configured but not present in ``df``.
        DataError: If ``group_by`` contains missing values.
    """
    if group_by is None:
        return

    if group_by not in df.columns:
        message = MISSING_GROUP_BY_COLUMN_ERROR.format(group_by=group_by)
        if "," in group_by:
            message += " The column name contains a comma -- multi-column grouping is not supported. Use a single column name."
        raise ParameterError(message)

    if df[group_by].isna().any():
        raise DataError(MISSING_GROUP_BY_VALUES_ERROR.format(group_by=group_by))


def validate_orderby_column(df: pd.DataFrame, order_by: str | None) -> None:
    """Validate that the configured order-by column exists.

    Args:
        df: Dataframe to validate.
        order_by: Name of the configured ordering column.

    Raises:
        ParameterError: If ``order_by`` is configured but not present in ``df``.
    """
    if order_by is None:
        return

    if order_by not in df.columns:
        raise ParameterError(MISSING_ORDER_BY_COLUMN_ERROR.format(order_by=order_by))
