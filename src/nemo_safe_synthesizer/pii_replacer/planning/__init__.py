# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal planning module for dataset-specific PII replacement plans."""

from __future__ import annotations

from .assembly import (
    ColumnClassification,
    DependencyCandidate,
    apply_dependencies,
    derive_dependency_candidates,
    plan_from_classifications,
)
from .io import load_plan, save_plan
from .llm import InferenceSettings, LLMPlanEnhancer, OpenAICompatibleTransport, resolve_inference_settings
from .patterns import pattern_grammar_catalog
from .resolver import (
    ColumnProfile,
    HeuristicPlanDiscoverer,
    PlanDiscoverer,
    PlanDiscoveryInput,
    PlanEnhancer,
    resolve_plan,
)
from .validation import protected_columns, validate_plan

__all__ = [
    "ColumnClassification",
    "ColumnProfile",
    "DependencyCandidate",
    "HeuristicPlanDiscoverer",
    "InferenceSettings",
    "LLMPlanEnhancer",
    "OpenAICompatibleTransport",
    "PlanDiscoverer",
    "PlanDiscoveryInput",
    "PlanEnhancer",
    "apply_dependencies",
    "derive_dependency_candidates",
    "load_plan",
    "pattern_grammar_catalog",
    "plan_from_classifications",
    "protected_columns",
    "resolve_plan",
    "resolve_inference_settings",
    "save_plan",
    "validate_plan",
]
