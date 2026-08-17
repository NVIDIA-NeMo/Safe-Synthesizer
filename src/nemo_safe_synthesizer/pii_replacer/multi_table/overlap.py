# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Value-overlap bundling for orphan standalone columns.

Schema PK/FK links always create key domains. Among standalone non-person
columns that share an ``entity_type`` and are not already in the same schema
domain, strong value overlap proposes an additional shared domain.

Constants (unique non-null values):
  - ``OVERLAP_MIN_SHARED_COUNT``: minimum shared distinct values
  - ``OVERLAP_MIN_FRACTION``: shared / min(|A|, |B|) must be at least this
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ...observability import get_logger

logger = get_logger(__name__)

# Guardrails for value-overlap domain bundling (documented at implement time).
OVERLAP_MIN_SHARED_COUNT = 2
OVERLAP_MIN_FRACTION = 0.5

# Standalone entity types eligible for overlap bundling (not persona attributes).
OVERLAP_ELIGIBLE_ENTITIES = frozenset(
    {
        "unique_identifier",
        "api_key",
        "ipv4",
        "ipv6",
        "ssn",
        "national_id",
        "credit_debit_card",
    }
)

__all__ = [
    "OVERLAP_ELIGIBLE_ENTITIES",
    "OVERLAP_MIN_FRACTION",
    "OVERLAP_MIN_SHARED_COUNT",
    "column_value_set",
    "overlap_score",
    "should_bundle_by_overlap",
]


def column_value_set(series: pd.Series) -> set[str]:
    """Return unique non-null string values from ``series``."""
    return {str(v) for v in series.dropna().unique()}


def overlap_score(a: set[str], b: set[str]) -> tuple[int, float]:
    """Return ``(shared_count, shared / min(|A|,|B|))`` (0.0 if either empty)."""
    if not a or not b:
        return 0, 0.0
    shared = len(a & b)
    frac = shared / min(len(a), len(b))
    return shared, frac


def should_bundle_by_overlap(a: set[str], b: set[str]) -> bool:
    """Whether two value sets meet the overlap bundling threshold."""
    shared, frac = overlap_score(a, b)
    return shared >= OVERLAP_MIN_SHARED_COUNT and frac >= OVERLAP_MIN_FRACTION


def warn_if_schema_domain_disjoint(
    domain_id: str,
    qualified_columns: Iterable[str],
    value_sets: dict[str, set[str]],
) -> None:
    """Warn when schema-linked columns have pairwise empty overlap in this extract."""
    cols = list(qualified_columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            a = value_sets.get(left) or set()
            b = value_sets.get(right) or set()
            if a and b and a.isdisjoint(b):
                logger.user.warning(
                    f"[PII Replacement] Schema key domain {domain_id!r} links {left!r} and "
                    f"{right!r}, but this extract has no overlapping values; keeping the "
                    "schema domain anyway."
                )
