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
    ConditioningColumn,
    EntityType,
    PiiColumnPlan,
    PiiReplacementPlan,
    PiiReplacementScope,
    is_columns_to_replace_type,
    validate_dependency_relationship,
    validate_pattern_eligibility,
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
        validate_pattern_eligibility(self.entity_type, self.pattern)
        return self


@dataclass(frozen=True, slots=True)
class DependencyCandidate:
    """One proposed dependency edge identified only by dataframe columns.

    Entity types deliberately remain in ``ColumnClassification`` as the single
    source of truth. ``apply_dependencies`` validates this edge against those
    classifications and the target plan before constructing the final plan.
    """

    target_column: str
    source_column: str

    def __post_init__(self) -> None:
        if self.target_column == self.source_column:
            raise ParameterError("a dependency candidate cannot target and source the same column")


def _classifications_by_column(
    classifications: Sequence[ColumnClassification],
) -> dict[str, ColumnClassification]:
    """Index classifications and reject ambiguous duplicate column entries."""
    by_column: dict[str, ColumnClassification] = {}
    duplicates: list[str] = []
    for classification in classifications:
        if classification.column_name in by_column:
            duplicates.append(classification.column_name)
        by_column[classification.column_name] = classification
    if duplicates:
        raise ParameterError(
            "classifications has duplicate column_name values: " + ", ".join(repr(column) for column in duplicates)
        )
    return by_column


def _validate_plan_classifications(
    plan: PiiReplacementPlan,
    classifications_by_column: dict[str, ColumnClassification],
) -> dict[str, PiiColumnPlan]:
    """Ensure replacement targets agree with their semantic classifications."""
    targets = {spec.column_name: spec for spec in plan.columns_to_replace}
    for target in targets.values():
        classification = classifications_by_column.get(target.column_name)
        if classification is None:
            raise ParameterError(f"replacement column {target.column_name!r} is missing from classifications")
        if classification.entity_type is not target.entity_type:
            classified_as = classification.entity_type.value if classification.entity_type is not None else None
            raise ParameterError(
                f"replacement column {target.column_name!r} is classified as {classified_as!r}, "
                f"but the plan uses {target.entity_type.value!r}"
            )
    return targets


def plan_from_classifications(
    scope: PiiReplacementScope,
    classifications: Sequence[ColumnClassification],
    *,
    protected_columns: Set[str] = frozenset(),
) -> PiiReplacementPlan:
    """Build replacement membership deterministically from semantic classifications."""
    _classifications_by_column(classifications)
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
    classifications_by_column = _classifications_by_column(classifications)
    _validate_plan_classifications(plan, classifications_by_column)
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
                    source_column=source.column_name,
                )
            )
    return candidates


def apply_dependencies(
    plan: PiiReplacementPlan,
    dependencies: Sequence[DependencyCandidate],
    *,
    classifications: Sequence[ColumnClassification],
) -> PiiReplacementPlan:
    """Validate and apply selected dependency edges using classified entity types."""
    dependencies_by_target: dict[str, list[ConditioningColumn]] = {}
    classifications_by_column = _classifications_by_column(classifications)
    targets = _validate_plan_classifications(plan, classifications_by_column)
    for dependency in dependencies:
        target = targets.get(dependency.target_column)
        if target is None:
            raise ParameterError(f"dependency targets unknown replacement column {dependency.target_column!r}")
        source = classifications_by_column.get(dependency.source_column)
        if source is None:
            raise ParameterError(f"dependency sources unknown classified column {dependency.source_column!r}")
        if source.entity_type is None:
            raise ParameterError(f"dependency source {dependency.source_column!r} is unclassified")
        validate_dependency_relationship(
            target.entity_type,
            source.entity_type,
            target_column=target.column_name,
        )
        # Replacement sources omit entity_type so PiiReplacementPlan infers it
        # from the source node. Read-only sources retain their explicit type.
        conditioner = (
            ConditioningColumn(column_name=source.column_name)
            if source.column_name in targets
            else ConditioningColumn(column_name=source.column_name, entity_type=source.entity_type)
        )
        dependencies_by_target.setdefault(dependency.target_column, []).append(conditioner)

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
