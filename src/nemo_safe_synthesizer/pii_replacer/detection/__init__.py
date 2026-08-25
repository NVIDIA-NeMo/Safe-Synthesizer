# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column and value classification for PII discovery."""

from __future__ import annotations

from .free_text import select_free_text_columns
from .structured import detect_structured_columns
from .value_recognizers import (
    UUID_RE,
    analyze_column_patterns,
    match_value_entity,
    match_value_pattern,
)

__all__ = [
    "UUID_RE",
    "analyze_column_patterns",
    "detect_structured_columns",
    "match_value_entity",
    "match_value_pattern",
    "select_free_text_columns",
]
