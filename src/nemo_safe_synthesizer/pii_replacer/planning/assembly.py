# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic assembly shared by PII plan discoverers and enhancers."""

from __future__ import annotations

from collections.abc import Sequence, Set
from dataclasses import dataclass
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from ...config.replace_pii import (
    ALLOWED_DEPENDS_ON,
    ENTITY_BY_TYPE,
    ConditioningColumn,
    EntityType,
    PiiColumnPlan,
    PiiReplacementPlan,
    PiiReplacementScope,
    is_columns_to_replace_type,
)
from ...errors import ParameterError

__all__ = [
    "ColumnClassification",
    "DependencyCandidate",
    "apply_dependencies",
    "derive_dependency_candidates",
    "plan_from_classifications",
]


class ColumnClassification(BaseModel):
    """Semantic classification produced by a heuristic or LLM planner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column_name: str
    entity_type: EntityType | None
    pattern: str | None = None

    @model_validator(mode="after")
    def _validate_pattern_eligibility(self) -> Self:
        if self.pattern is None:
            return self
        if not self.pattern.strip():
            raise ValueError("patterns must be non-empty when provided")
        if self.entity_type is None:
            raise ValueError("unclassified columns cannot include a pattern")
        if ENTITY_BY_TYPE[self.entity_type].pattern_syntax is None:
            raise ValueError("the classified entity_type does not support patterns")
        return self


@dataclass(frozen=True, slots=True)
class DependencyCandidate:
    """One catalog-permitted dependency that a planner may select."""

    target_column: str
    target_entity_type: EntityType
    source_column: str
    source_entity_type: EntityType

    def __post_init__(self) -> None:
        if self.target_column == self.source_column:
            raise ParameterError("a dependency candidate cannot target and source the same column")
        allowed_sources = ALLOWED_DEPENDS_ON.get(self.target_entity_type, frozenset())
        if self.source_entity_type not in allowed_sources:
            raise ParameterError(
                f"entity_type {self.source_entity_type.value!r} cannot condition "
                f"entity_type {self.target_entity_type.value!r}"
            )


def plan_from_classifications(
    scope: PiiReplacementScope,
    classifications: Sequence[ColumnClassification],
    *,
    protected_columns: Set[str] = frozenset(),
) -> PiiReplacementPlan:
    """Build replacement membership deterministically from semantic classifications."""
    columns_to_replace: list[PiiColumnPlan] = []
    for classification in classifications:
        entity_type = classification.entity_type
        if (
            entity_type is None
            or not is_columns_to_replace_type(entity_type)
            or classification.column_name in protected_columns
        ):
            continue
        columns_to_replace.append(
            PiiColumnPlan(
                column_name=classification.column_name,
                entity_type=entity_type,
                pattern=classification.pattern,
            )
        )
    return PiiReplacementPlan(scope=scope, columns_to_replace=columns_to_replace)


def derive_dependency_candidates(
    plan: PiiReplacementPlan,
    classifications: Sequence[ColumnClassification],
) -> list[DependencyCandidate]:
    """Return every dependency permitted by the entity relationship catalog."""
    candidates: list[DependencyCandidate] = []
    for target in plan.columns_to_replace:
        allowed_sources = ALLOWED_DEPENDS_ON.get(target.entity_type, frozenset())
        for source in classifications:
            if (
                source.entity_type is None
                or source.entity_type not in allowed_sources
                or source.column_name == target.column_name
            ):
                continue
            candidates.append(
                DependencyCandidate(
                    target_column=target.column_name,
                    target_entity_type=target.entity_type,
                    source_column=source.column_name,
                    source_entity_type=source.entity_type,
                )
            )
    return candidates


def apply_dependencies(
    plan: PiiReplacementPlan,
    dependencies: Sequence[DependencyCandidate],
) -> PiiReplacementPlan:
    """Return ``plan`` with the selected catalog-derived dependencies applied."""
    dependencies_by_target: dict[str, list[ConditioningColumn]] = {}
    targets = {spec.column_name: spec for spec in plan.columns_to_replace}
    for dependency in dependencies:
        target = targets.get(dependency.target_column)
        if target is None:
            raise ParameterError(f"dependency targets unknown replacement column {dependency.target_column!r}")
        if target.entity_type is not dependency.target_entity_type:
            raise ParameterError(
                f"dependency target {dependency.target_column!r} is classified as "
                f"{dependency.target_entity_type.value!r}, but the plan uses {target.entity_type.value!r}"
            )
        dependencies_by_target.setdefault(dependency.target_column, []).append(
            ConditioningColumn(
                column_name=dependency.source_column,
                entity_type=dependency.source_entity_type,
            )
        )

    return PiiReplacementPlan(
        scope=plan.scope,
        columns_to_replace=[
            PiiColumnPlan(
                column_name=spec.column_name,
                entity_type=spec.entity_type,
                pattern=spec.pattern,
                depends_on=dependencies_by_target.get(spec.column_name, []),
            )
            for spec in plan.columns_to_replace
        ],
    )
