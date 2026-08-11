# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-column descriptive statistics and group/record scope classification."""

from __future__ import annotations

import pandas as pd


def _sample_values(series: pd.Series, k: int = 8) -> list[str]:
    vals = series.dropna().unique().tolist()
    return [str(v) for v in vals[:k]]


def column_stats(df: pd.DataFrame) -> dict[str, dict]:
    """Compute per-column descriptive statistics for detection gates.

    Args:
        df: Input dataframe whose columns are summarized.

    Returns:
        Mapping of column name to stats dict with ``dtype``, ``n_unique``,
        ``unique_ratio``, ``null_rate``, and ``samples`` keys.
    """
    stats: dict[str, dict] = {}
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        nun = int(non_null.nunique())
        stats[col] = {
            "dtype": str(s.dtype),
            "n_unique": nun,
            "unique_ratio": round(nun / len(non_null), 4) if len(non_null) else 0.0,
            "null_rate": round(float(s.isna().mean()), 4),
            "samples": _sample_values(non_null),
        }
    return stats


def within_group_constancy(df: pd.DataFrame, key: str, col: str) -> float:
    """Return the fraction of ``key`` groups where ``col`` has one distinct value.

    Args:
        df: Input dataframe.
        key: Grouping column name.
        col: Column whose within-group constancy is measured.

    Returns:
        Fraction in ``[0.0, 1.0]``; ``1.0`` when ``key == col``.
    """
    if key == col:
        return 1.0
    g = df.groupby(key, dropna=True)[col].nunique(dropna=True)
    return float((g <= 1).mean()) if len(g) else 0.0


def classify_columns_by_scope(df: pd.DataFrame, group_key: str, threshold: float) -> tuple[list[str], list[str]]:
    """Split non-key columns into group-constant and record-varying lists.

    Args:
        df: Input dataframe.
        group_key: Column used to define groups.
        threshold: Minimum within-group constancy to classify a column as group-constant.

    Returns:
        Tuple ``(group_constant_columns, record_varying_columns)``.
    """
    const_cols, vary_cols = [], []
    for c in df.columns:
        if c == group_key:
            continue
        if within_group_constancy(df, group_key, c) >= threshold:
            const_cols.append(c)
        else:
            vary_cols.append(c)
    return const_cols, vary_cols


def scoped_column_stats(
    df: pd.DataFrame, group_key: str | None, group_constancy_threshold: float = 0.95
) -> dict[str, dict]:
    """Return ``column_stats`` with scope-aware cardinality denominators.

    For group-constant columns, ``unique_ratio`` is recomputed as
    ``n_unique / n_groups`` so a per-group attribute (e.g. one MRN per patient
    that repeats on every visit row) is not mistaken for low-variety free text.
    Record-level columns keep the per-row denominator. Each entry also gets a
    ``scope`` tag (``key``, ``group``, or ``record``).

    Args:
        df: Input dataframe.
        group_key: Grouping column name, or ``None`` to treat all columns as record-scoped.
        group_constancy_threshold: Constancy threshold passed to
            ``classify_columns_by_scope``.

    Returns:
        Mapping of column name to stats dict (same keys as ``column_stats`` plus
        ``scope``).
    """
    stats = column_stats(df)
    if not (group_key and group_key in df.columns):
        for c in stats:
            stats[c]["scope"] = "record"
        return stats
    n_groups = int(df[group_key].nunique(dropna=True)) or len(df)
    const_cols, _ = classify_columns_by_scope(df, group_key, group_constancy_threshold)
    const_set = set(const_cols)
    for c in stats:
        if c == group_key:
            # The group key is group-constant by definition (n_unique == n_groups),
            # so its cardinality is also measured per group -> unique_ratio == 1.0.
            nun = stats[c]["n_unique"]
            stats[c]["unique_ratio"] = round(nun / n_groups, 4) if n_groups else 0.0
            stats[c]["scope"] = "key"
        elif c in const_set:
            nun = stats[c]["n_unique"]
            stats[c]["unique_ratio"] = round(nun / n_groups, 4) if n_groups else 0.0
            stats[c]["scope"] = "group"
        else:
            stats[c]["scope"] = "record"
    return stats
