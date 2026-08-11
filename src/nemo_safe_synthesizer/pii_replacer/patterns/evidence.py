# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared evidence slice and ranking for every pattern flavour.

Temporal formats, value templates, and persona conventions all read the same
seeded sample and name formats by the same share of the column. Discovery and
plan validation must use this module so a secondary format found in one sample
cannot fail validation that only inspects another.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from ..entities import is_missing_value, sval

# Rows sampled when inferring which formats a column writes.
PATTERN_SAMPLE_SIZE = 2000
# A format has to describe this much of a column to be worth naming, and a column
# names at most this many; a value no named format describes keeps its own shape.
PATTERN_MIN_SHARE = 5.0
PATTERN_MAX_PATTERNS = 5


def pattern_evidence_values(values: pd.Series, *, sample_size: int = PATTERN_SAMPLE_SIZE) -> list[str]:
    """Return the shared value slice for pattern inference and plan-pattern validation.

    Uses a seeded row sample (not first-seen distincts) so common secondary
    formats that appear late in ``unique()`` order are still visible to both
    discovery and ``validate_plan``.

    Args:
        values: Column values to sample.
        sample_size: Maximum number of non-null rows to include.

    Returns:
        Non-null string values from the sample, excluding missing-value markers.
        A 10k-row column yields at most ``PATTERN_SAMPLE_SIZE`` (2000) strings
        with ``random_state=0``.
    """
    non_null = values.dropna()
    if non_null.empty:
        return []
    if len(non_null) > sample_size:
        non_null = non_null.sample(sample_size, random_state=0)
    out: list[str] = []
    for v in non_null:
        s = sval(v)
        if s and not is_missing_value(s):
            out.append(s)
    return out


def ranked_formats(counts: Mapping[str, int], total: int) -> list[str]:
    """Return formats worth naming, most common first.

    A long tail of one-off shapes says nothing about the column, and each of those
    values is replaced in its own shape anyway, so only formats above
    ``PATTERN_MIN_SHARE`` are named and at most ``PATTERN_MAX_PATTERNS`` of them.

    Args:
        counts: Format label to occurrence count.
        total: Total values counted.

    Returns:
        Up to ``PATTERN_MAX_PATTERNS`` format labels whose share is at least
        ``PATTERN_MIN_SHARE`` percent.

    Example:
        ``{"%m/%d/%Y": 90, "%Y-%m-%d": 8, "odd": 2}`` with ``total=100`` ->
        ``["%m/%d/%Y"]`` (8% and 2% fall below the 5% share floor).
    """
    if not total:
        return []
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [fmt for fmt, count in ranked if count / total * 100 >= PATTERN_MIN_SHARE][:PATTERN_MAX_PATTERNS]
