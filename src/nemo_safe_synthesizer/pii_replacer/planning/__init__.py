# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a replacement plan is arrived at: discovered from a dataframe and persisted.

Full plan validation lands in a follow-up PR. Replacement execution is still a
no-op on this branch (``process_data`` writes the plan YAML only).
"""

from __future__ import annotations

from .discovery import build_depends_on_hints, discover_plan, discover_plan_with_hints
from .io import (
    PII_REPLACEMENT_PLAN_FILENAME,
    PLAN_YAML_HEADER,
    PLAN_YAML_SECTION_COMMENTS,
    load_plan_from_path,
    plan_to_commented_yaml,
    resolve_plan,
    save_plan_to_path,
)

__all__ = [
    "PII_REPLACEMENT_PLAN_FILENAME",
    "PLAN_YAML_HEADER",
    "PLAN_YAML_SECTION_COMMENTS",
    "build_depends_on_hints",
    "discover_plan",
    "discover_plan_with_hints",
    "load_plan_from_path",
    "plan_to_commented_yaml",
    "resolve_plan",
    "save_plan_to_path",
]
