# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a replacement plan is arrived at: discovered, validated, and persisted.

``discovery`` turns the evidence a dataframe carries into a plan, ``validation``
says whether a plan (auto-discovered or hand-written) can run on that dataframe,
and ``io`` reads and writes plan YAML and resolves the configured source into
the plan the replacement finally runs.
"""

from __future__ import annotations

from .discovery import discover_plan
from .io import (
    PII_REPLACEMENT_PLAN_FILENAME,
    PLAN_YAML_HEADER,
    PLAN_YAML_SECTION_COMMENTS,
    load_plan_from_path,
    plan_to_commented_yaml,
    resolve_plan,
    save_plan_to_path,
)
from .validation import (
    DATE_PATTERN_ENTITIES,
    PATTERNED_ENTITIES,
    PERSONA_PATTERN_ENTITIES,
    TEMPLATE_PATTERN_ENTITIES,
    PlanIssue,
    iter_plan_advisories,
    iter_plan_issues,
    protected_columns,
    strip_protected_columns_from_plan,
    validate_plan,
)

__all__ = [
    "DATE_PATTERN_ENTITIES",
    "PATTERNED_ENTITIES",
    "PERSONA_PATTERN_ENTITIES",
    "PII_REPLACEMENT_PLAN_FILENAME",
    "PLAN_YAML_HEADER",
    "PLAN_YAML_SECTION_COMMENTS",
    "TEMPLATE_PATTERN_ENTITIES",
    "PlanIssue",
    "discover_plan",
    "iter_plan_advisories",
    "iter_plan_issues",
    "load_plan_from_path",
    "plan_to_commented_yaml",
    "protected_columns",
    "resolve_plan",
    "save_plan_to_path",
    "strip_protected_columns_from_plan",
    "validate_plan",
]
