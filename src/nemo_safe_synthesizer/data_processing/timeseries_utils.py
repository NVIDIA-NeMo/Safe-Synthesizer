# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared internal helpers for deterministic time-series row ordering."""

from __future__ import annotations

import pandas as pd


def unused_column_name(preferred: str, columns: list[str]) -> str:
    """Return the preferred column name or a deterministic suffixed variant."""
    if preferred not in columns:
        return preferred
    suffix = 1
    while f"{preferred}_{suffix}" in columns:
        suffix += 1
    return f"{preferred}_{suffix}"


def stable_sort_within_groups(
    data: pd.DataFrame,
    group_column: str,
    order_column: str | None,
    *,
    normalized_order: pd.Series | None = None,
) -> pd.DataFrame:
    """Sort groups deterministically while retaining source position as the final tie-breaker."""
    working = data.copy()
    temporary_columns: list[str] = []

    source_position_column = unused_column_name("__nss_source_position", list(working.columns))
    working[source_position_column] = range(len(working))
    temporary_columns.append(source_position_column)

    sort_order_column = order_column
    if normalized_order is not None:
        sort_order_column = unused_column_name("__nss_source_order", list(working.columns))
        working[sort_order_column] = normalized_order
        temporary_columns.append(sort_order_column)

    sort_columns = [group_column]
    if sort_order_column is not None and sort_order_column != group_column:
        sort_columns.append(sort_order_column)
    sort_columns.append(source_position_column)
    return (
        working.sort_values(sort_columns, kind="mergesort")
        .drop(columns=temporary_columns)
        .reset_index(drop=True)
    )
