# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column and value classification for PII discovery."""

from __future__ import annotations

from .column_grouping import detect_structured_columns, names_agree_for_link
from .column_names import fuzzy_match_label
from .free_text import select_free_text_columns
from .stats import (
    classify_columns_by_grain,
    column_stats,
    scoped_column_stats,
    within_group_constancy,
)
from .value_recognizers import (
    API_PREFIXES,
    UUID_RE,
    EntityCoverage,
    analyze_column_patterns,
    candidate_entities,
    card_brand,
    collect_value_entities,
    entity_coverage,
    looks_like_api_key_value,
    looks_like_person_name,
    looks_like_street_address,
)

__all__ = [
    "API_PREFIXES",
    "UUID_RE",
    "EntityCoverage",
    "analyze_column_patterns",
    "candidate_entities",
    "card_brand",
    "classify_columns_by_grain",
    "collect_value_entities",
    "column_stats",
    "detect_structured_columns",
    "entity_coverage",
    "fuzzy_match_label",
    "looks_like_api_key_value",
    "looks_like_person_name",
    "looks_like_street_address",
    "names_agree_for_link",
    "scoped_column_stats",
    "select_free_text_columns",
    "within_group_constancy",
]
