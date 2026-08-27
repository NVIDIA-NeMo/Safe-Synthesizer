# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared evidence slice and ranking for every pattern flavour.

Temporal formats, value templates, and name/email conventions all read the same
seeded sample. Plan emission names at most one dominant format (≥ 85% coverage
of that sample); internal shape grouping may still rank formats for inference.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from ..entities import is_missing_value, sval

# Rows sampled when inferring which formats a column writes.
PATTERN_SAMPLE_SIZE = 2000
# Legacy multi-pattern floor / cap — kept for internal ranked_formats callers
# (e.g. handle-shape enumeration). Plan emission uses ``dominant_format`` instead.
PATTERN_MIN_SHARE = 5.0
PATTERN_MAX_PATTERNS = 5
# Plan emission: write ``pattern`` only when the top format covers this share.
DOMINANT_PATTERN_MIN_COVERAGE = 85.0


def pattern_evidence_values(values: pd.Series, *, sample_size: int = PATTERN_SAMPLE_SIZE) -> list[str]:
    """Return the shared value slice for pattern inference.

    Uses a seeded row sample (not first-seen distincts) so common secondary
    formats that appear late in ``unique()`` order are still visible.

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
    """Return formats above ``PATTERN_MIN_SHARE``, most common first (internal use).

    Plan emission should call ``dominant_format`` instead. This helper remains for
    inference internals that need a short ranked list (e.g. email handle shapes).
    """
    if not total:
        return []
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [fmt for fmt, count in ranked if count / total * 100 >= PATTERN_MIN_SHARE][:PATTERN_MAX_PATTERNS]


def dominant_format(
    counts: Mapping[str, int],
    total: int,
    *,
    min_coverage: float = DOMINANT_PATTERN_MIN_COVERAGE,
) -> str | None:
    """Return the top format when it covers ``min_coverage`` percent of ``total``.

    Args:
        counts: Format label to occurrence count.
        total: Size of the evidence sample (denominator).
        min_coverage: Minimum percent of ``total`` the top format must cover.

    Returns:
        The dominant format string, or ``None`` when none reaches the threshold.
    """
    if not total or not counts:
        return None
    # Highest count, then lexicographically smallest label on ties.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    fmt, count = ranked[0]
    if count / total * 100 >= min_coverage:
        return fmt
    return None
