# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Value and persona format templates: infer, match, generate, and attach to plans.

The formats a column writes are read in three flavours, one module each:
``temporal`` for strftime dates, times and durations, ``value_templates`` for the
character templates identifiers, phones and cards wear, and
``persona_templates`` for the conventions name and email columns write a person
in. Shared sample size, share caps, evidence slicing, and ranking live in
``evidence``.
"""

from __future__ import annotations

from .evidence import (
    PATTERN_MAX_PATTERNS,
    PATTERN_MIN_SHARE,
    PATTERN_SAMPLE_SIZE,
    pattern_evidence_values,
    ranked_formats,
)
from .persona_templates import (
    DIGIT_PLACEHOLDER,
    DOMAIN_PLACEHOLDER,
    EMAIL_DOMAIN_ONLY_PATTERN,
    PERSONA_PLACEHOLDERS,
    attach_persona_patterns,
    handle_email_pattern,
    infer_email_pattern,
    infer_persona_pattern,
    name_parts,
    persona_column_patterns,
    placeholder_tokens,
    render_email_pattern,
    render_persona_pattern,
    split_email,
    split_full_name,
    split_title,
)
from .temporal import (
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
    conform_to_template,
    generate_from_pattern,
    infer_value_pattern,
    luhn_valid,
    matching_template,
    pattern_preserving_token,
    synth_card_value,
    value_matches_template,
    value_patterns,
    value_shape_template,
    value_structure_template,
    value_template_is_constant,
)

__all__ = [
    "DIGIT_PLACEHOLDER",
    "DOMAIN_PLACEHOLDER",
    "EMAIL_DOMAIN_ONLY_PATTERN",
    "PATTERN_MAX_PATTERNS",
    "PATTERN_MIN_SHARE",
    "PATTERN_SAMPLE_SIZE",
    "PERSONA_PLACEHOLDERS",
    "attach_persona_patterns",
    "attach_value_patterns",
    "conform_to_template",
    "date_patterns",
    "detect_date_format",
    "generate_from_pattern",
    "handle_email_pattern",
    "infer_email_pattern",
    "infer_persona_pattern",
    "infer_value_pattern",
    "luhn_valid",
    "match_date_format",
    "match_datetime_format",
    "match_duration_format",
    "match_time_format",
    "matching_template",
    "name_parts",
    "pattern_evidence_values",
    "pattern_preserving_token",
    "persona_column_patterns",
    "placeholder_tokens",
    "ranked_formats",
    "render_email_pattern",
    "render_persona_pattern",
    "split_email",
    "split_full_name",
    "split_title",
    "synth_card_value",
    "try_strftime_formats",
    "value_matches_template",
    "value_patterns",
    "value_shape_template",
    "value_structure_template",
    "value_template_is_constant",
]
