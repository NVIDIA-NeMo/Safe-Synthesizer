# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Value and name format templates: infer, match, generate, and attach to plans.

The formats a column writes are read in three flavours, one module each:
``temporal`` for strftime dates, times and durations, ``value_templates`` for the
character templates identifiers, phones and cards wear, and
``name_templates`` for the conventions name and email columns write a person
in. Shared sample size, evidence slicing, and dominant-format selection live in
``evidence``.
"""

from __future__ import annotations

from .evidence import (
    DOMINANT_PATTERN_MIN_COVERAGE,
    PATTERN_MAX_PATTERNS,
    PATTERN_MIN_SHARE,
    PATTERN_SAMPLE_SIZE,
    dominant_format,
    pattern_evidence_values,
    ranked_formats,
)
from .name_templates import (
    DIGIT_PLACEHOLDER,
    DOMAIN_PLACEHOLDER,
    EMAIL_DOMAIN_ONLY_PATTERN,
    NAME_PLACEHOLDERS,
    attach_name_patterns,
    handle_email_pattern,
    infer_email_pattern,
    infer_name_pattern,
    name_column_pattern,
    name_parts,
    placeholder_tokens,
    render_email_pattern,
    render_name_pattern,
    split_email,
    split_full_name,
    split_title,
)
from .temporal import (
    date_pattern,
    date_patterns,
    detect_date_format,
    match_date_format,
    match_datetime_format,
    match_duration_format,
    match_time_format,
    try_strftime_formats,
)
from .value_templates import (
    attach_value_patterns,
    generate_from_pattern,
    infer_value_pattern,
    luhn_valid,
    matching_template,
    ranked_value_patterns,
    synth_card_value,
    value_matches_template,
    value_pattern,
    value_patterns,
    value_shape_template,
    value_structure_template,
    value_template_has_unbalanced_brackets,
    value_template_is_constant,
)

__all__ = [
    "DIGIT_PLACEHOLDER",
    "DOMAIN_PLACEHOLDER",
    "DOMINANT_PATTERN_MIN_COVERAGE",
    "EMAIL_DOMAIN_ONLY_PATTERN",
    "NAME_PLACEHOLDERS",
    "PATTERN_MAX_PATTERNS",
    "PATTERN_MIN_SHARE",
    "PATTERN_SAMPLE_SIZE",
    "attach_name_patterns",
    "attach_value_patterns",
    "date_pattern",
    "date_patterns",
    "detect_date_format",
    "dominant_format",
    "generate_from_pattern",
    "handle_email_pattern",
    "infer_email_pattern",
    "infer_name_pattern",
    "infer_value_pattern",
    "luhn_valid",
    "match_date_format",
    "match_datetime_format",
    "match_duration_format",
    "match_time_format",
    "matching_template",
    "name_column_pattern",
    "name_parts",
    "pattern_evidence_values",
    "placeholder_tokens",
    "ranked_formats",
    "ranked_value_patterns",
    "render_email_pattern",
    "render_name_pattern",
    "split_email",
    "split_full_name",
    "split_title",
    "synth_card_value",
    "try_strftime_formats",
    "value_matches_template",
    "value_pattern",
    "value_patterns",
    "value_shape_template",
    "value_structure_template",
    "value_template_has_unbalanced_brackets",
    "value_template_is_constant",
]
