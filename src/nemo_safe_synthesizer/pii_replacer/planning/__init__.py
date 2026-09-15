# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal planning module for dataset-specific PII replacement plans."""

from __future__ import annotations

from .io import load_plan, save_plan
from .llm import LLMPlanEnhancer
from .patterns import pattern_grammar_catalog
from .plan_builder import (
    ColumnClassification,
    DependencyCandidate,
    apply_dependencies,
    derive_dependency_candidates,
    plan_from_classifications,
)
from .resolver import (
    ColumnProfile,
    HeuristicPlanDiscoverer,
    PlanDiscoverer,
    PlanDiscoveryInput,
    PlanEnhancer,
    resolve_plan,
    resolve_replacement_config,
)
from .validation import get_protected_columns, validate_plan

__all__ = [
    "ColumnClassification",
    "ColumnProfile",
    "DependencyCandidate",
    "HeuristicPlanDiscoverer",
    "LLMPlanEnhancer",
    "PlanDiscoverer",
    "PlanDiscoveryInput",
    "PlanEnhancer",
    "apply_dependencies",
    "derive_dependency_candidates",
    "load_plan",
    "pattern_grammar_catalog",
    "plan_from_classifications",
    "get_protected_columns",
    "resolve_plan",
    "resolve_replacement_config",
    "save_plan",
    "validate_plan",
]
