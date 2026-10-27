# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset context that helps a person review a drafted replacement plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from ...config.replace_pii import EntityType, PiiReplacementPlan, ReplacePiiConfig
from .dependency_mappings import conditioner_columns, distinct_source_values

__all__ = ["DependencyValueReview", "PlanReviewNotes", "plan_review_notes"]


@dataclass(frozen=True, slots=True)
class DependencyValueReview:
    """How one depends_on column's values reach the sampler."""

    column_name: str
    """Dependency column the values come from."""

    entity_type: EntityType
    """Entity type the plan assigns to that column."""

    used_as_is: tuple[str, ...]
    """Distinct dataset values without a mapping entry that match a sampler value, ignoring case."""

    sampler_values: tuple[str, ...]
    """Values the sampler accepts for ``entity_type``."""


@dataclass(frozen=True, slots=True)
class PlanReviewNotes:
    """Dataset context written as comments into a saved plan, describing the plan as drafted.

    For example, a dataset with columns ``name``, ``sex``, and ``admission_id``
    and a plan that replaces only ``name`` conditioned on ``sex`` gives
    ``unchanged_columns=("sex", "admission_id")`` and one
    ``DependencyValueReview`` for ``sex``.
    """

    unchanged_columns: tuple[str, ...]
    """Dataset columns the plan leaves unchanged, in dataset order."""

    dependency_values: tuple[DependencyValueReview, ...] = ()
    """One entry per depends_on column whose entity type the sampler has values for."""


def plan_review_notes(
    dataframe: pd.DataFrame,
    config: ReplacePiiConfig,
    *,
    sampler_value_catalog: Mapping[EntityType, tuple[str, ...]] | None = None,
) -> PlanReviewNotes | None:
    """Return review notes for ``config``'s inline plan, or ``None`` when it has no inline plan.

    Args:
        dataframe: Data the plan was resolved against.
        config: Resolved replacement configuration.
        sampler_value_catalog: Values the sampler accepts per entity type; loaded from
            the plan's ``sampler`` in ``config.locale`` when omitted and the plan has depends_on columns.

    Raises:
        ParameterError: If the sampler catalog must be loaded and cannot be.
    """
    plan = config.inline_plan
    if plan is None:
        return None
    dependencies = conditioner_columns(plan)
    if sampler_value_catalog is None and dependencies:
        from ..sampler_catalog import load_sampler_value_catalog

        sampler_value_catalog = load_sampler_value_catalog(plan.sampler, config.locale)
    catalog = sampler_value_catalog or {}
    replaced = {spec.column_name for spec in plan.columns_to_replace}
    return PlanReviewNotes(
        unchanged_columns=tuple(str(column) for column in dataframe.columns if column not in replaced),
        dependency_values=tuple(
            review
            for column_name, entity_type in dependencies.items()
            if (review := _dependency_value_review(dataframe, plan, column_name, entity_type, catalog)) is not None
        ),
    )


def _dependency_value_review(
    dataframe: pd.DataFrame,
    plan: PiiReplacementPlan,
    column_name: str,
    entity_type: EntityType,
    catalog: Mapping[EntityType, tuple[str, ...]],
) -> DependencyValueReview | None:
    sampler_values = catalog.get(entity_type)
    if not sampler_values or column_name not in dataframe.columns:
        return None
    matches = {value.casefold() for value in sampler_values}
    mapped = {source.casefold() for source in plan.data_to_sampler_value_mapping.get(column_name, {})}
    used_as_is = tuple(
        value
        for value in distinct_source_values(dataframe[column_name])
        if value.casefold() in matches and value.casefold() not in mapped
    )
    return DependencyValueReview(column_name, entity_type, used_as_is, sampler_values)
