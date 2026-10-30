# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve dataset dependency columns against sampler value catalogs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from ...config.replace_pii import DataToSamplerValueMapping, EntityType, PiiReplacementPlan
from ...errors import ParameterError
from ...observability import get_logger
from ..replacement.canonicalization import canonicalize_scalar, readable_value
from ..sampler_catalog import SamplerValueCatalog

logger = get_logger(__name__)

MAX_MAPPING_SOURCE_VALUE_LENGTH = 128
MAX_CONDITIONER_DISTINCT_VALUES = 1000

BUILT_IN_VALUE_MAPPINGS: Mapping[EntityType, Mapping[str, str]] = MappingProxyType(
    {
        EntityType.GENDER: MappingProxyType(
            {
                "f": "female",
                "female": "female",
                "woman": "female",
                "girl": "female",
                "m": "male",
                "male": "male",
                "man": "male",
                "boy": "male",
            }
        ),
    }
)
"""Common dataset spellings and the sampler value each maps to, matched ignoring case.

Resolution writes matching entries into the plan's ``data_to_sampler_value_mapping``
for every sampler, so they are visible and editable in the saved plan. Every
sampler accepts ``female`` and ``male`` (see ``PERSONA_SEX_VALUES``), so the
entries are the same for every sampler and locale.
"""


@dataclass(frozen=True, slots=True)
class DependencyMappingInput:
    """Unmatched values for one dependency column and sampler value catalog."""

    column_name: str
    """Dependency column the values come from."""

    entity_type: EntityType
    """Entity type the plan assigns to that column."""

    source_values: tuple[str, ...]
    """Distinct dataset values with neither a case-insensitive sampler match nor a mapping entry."""

    sampler_values: tuple[str, ...]
    """Values the sampler accepts for ``entity_type``."""


def conditioner_columns(plan: PiiReplacementPlan) -> dict[str, EntityType]:
    """Return each dependency source column and its single resolved entity type."""
    columns: dict[str, EntityType] = {}
    for spec in plan.columns_to_replace:
        for dependency in spec.depends_on:
            entity_type = dependency.entity_type
            if entity_type is None:
                raise ParameterError(f"dependency column {dependency.column_name!r} is missing its entity type")
            prior = columns.get(dependency.column_name)
            if prior is not None and prior is not entity_type:
                raise ParameterError(
                    f"dependency column {dependency.column_name!r} is used as both "
                    f"{prior.value!r} and {entity_type.value!r}"
                )
            columns[dependency.column_name] = entity_type
    return columns


def mapping_inputs(
    dataframe: pd.DataFrame,
    plan: PiiReplacementPlan,
    catalog: SamplerValueCatalog,
) -> tuple[DependencyMappingInput, ...]:
    """Return conditioner values with neither a case-insensitive sampler match nor a mapping entry."""
    inputs: list[DependencyMappingInput] = []
    for column_name, entity_type in conditioner_columns(plan).items():
        sampler_values = catalog.get(entity_type)
        if not sampler_values:
            continue
        covered = {sampler_value.casefold() for sampler_value in sampler_values} | {
            source.casefold() for source in plan.data_to_sampler_value_mapping.get(column_name, {})
        }
        source_values = distinct_source_values(dataframe[column_name])
        unmatched = tuple(value for value in source_values if value.casefold() not in covered)
        if unmatched:
            inputs.append(
                DependencyMappingInput(
                    column_name=column_name,
                    entity_type=entity_type,
                    source_values=unmatched,
                    sampler_values=sampler_values,
                )
            )
    return tuple(inputs)


def validate_data_to_sampler_value_mapping(
    plan: PiiReplacementPlan, catalog: SamplerValueCatalog, *, sampler_name: str
) -> None:
    """Validate the plan's value mappings against its conditioner columns and the sampler catalog.

    Every entry the sampler cannot use is reported in one error: a target that
    is not a sampler value, and any mapping for a column whose entity type the
    sampler has no values for, such as ethnicity with Faker. A plan is checked
    against the sampler of the run that uses it, so a plan written for another
    sampler or locale fails here.

    Args:
        plan: Replacement plan whose ``data_to_sampler_value_mapping`` is checked.
        catalog: Sampler values available for each conditioner entity type.
        sampler_name: The sampler the catalog belongs to, as error messages name it.

    Raises:
        ParameterError: If a mapped column is not a conditioner column, or a
            mapping entry does not match the sampler.

    Example:
        With a plan where ``first_name`` depends on ``sex`` (``gender``), the
        catalog ``{GENDER: ("female", "male")}`` accepts
        ``{"sex": {"Woman": ["female"]}}`` and rejects
        ``{"sex": {"Woman": ["woman"]}}``.
    """
    columns = conditioner_columns(plan)
    for column_name in plan.data_to_sampler_value_mapping:
        if column_name not in columns:
            raise ParameterError(
                f"data_to_sampler_value_mapping column {column_name!r} is not used as a dependency "
                "in the replacement plan"
            )
    _, problems = _usable_value_mapping(plan, catalog)
    if problems:
        raise ParameterError(
            f"data_to_sampler_value_mapping does not match the {sampler_name} sampler: {'; '.join(problems)}. "
            "The plan may have been written for another sampler or locale; change these entries, rerun discovery "
            "with replace_pii.replacement_plan set to auto_discovery, or set replace_pii.sampler and "
            "replace_pii.locale to the ones the plan was written for"
        )


def usable_value_mapping(
    plan: PiiReplacementPlan, catalog: SamplerValueCatalog, *, sampler_name: str
) -> DataToSamplerValueMapping:
    """Return the plan's value mapping without the entries the sampler cannot use, warning about them.

    This is the lenient form of ``validate_data_to_sampler_value_mapping`` for a plan the user supplied:
    entries for a column that is not a dependency, for an entity type the sampler has no values for, and
    targets the sampler does not accept are dropped. A dataset value left without targets is then
    unmatched; see ``entries_for_unmatched_values``.

    Example:
        With the catalog ``{GENDER: ("female", "male")}``, ``{"sex": {"Woman": ["woman", "female"]}}``
        becomes ``{"sex": {"Woman": ["female"]}}``.
    """
    usable, problems = _usable_value_mapping(plan, catalog)
    if problems:
        logger.user.warning(
            f"Ignoring data_to_sampler_value_mapping entries the {sampler_name} sampler cannot use: "
            f"{'; '.join(problems)}."
        )
    return usable


def _usable_value_mapping(
    plan: PiiReplacementPlan, catalog: SamplerValueCatalog
) -> tuple[DataToSamplerValueMapping, list[str]]:
    """Return the plan's value mapping restricted to what the sampler can use, and what was left out."""
    columns = conditioner_columns(plan)
    usable: DataToSamplerValueMapping = {}
    problems: list[str] = []
    for column_name, value_mappings in plan.data_to_sampler_value_mapping.items():
        entity_type = columns.get(column_name)
        if entity_type is None:
            problems.append(f"column {column_name!r} is not used as a dependency in the replacement plan")
            continue
        supported = catalog.get(entity_type)
        if not supported:
            problems.append(f"column {column_name!r} maps {entity_type.value} values, which this sampler does not have")
            continue
        supported_keys = {sampler_value.casefold() for sampler_value in supported}
        unknown = sorted(
            {
                target
                for targets in value_mappings.values()
                for target in targets
                if target.casefold() not in supported_keys
            }
        )
        if unknown:
            problems.append(
                f"column {column_name!r} maps to {', '.join(unknown)}, which this sampler does not accept "
                f"(it accepts {', '.join(supported)})"
            )
        kept = {
            source: [target for target in targets if target.casefold() in supported_keys]
            for source, targets in value_mappings.items()
        }
        if kept := {source: targets for source, targets in kept.items() if targets}:
            usable[column_name] = kept
    return usable, problems


def entries_for_unmatched_values(
    dataframe: pd.DataFrame, plan: PiiReplacementPlan, catalog: SamplerValueCatalog, *, sampler_name: str
) -> DataToSamplerValueMapping:
    """Return entries that let each unmatched dataset value match any sampler value, warning about them.

    This is the lenient form of the unmatched-value check for a plan the user supplied: a dataset value
    with neither a sampler match nor a mapping entry no longer conditions the replacement.

    Example:
        With the catalog ``{GENDER: ("female", "male")}`` and a ``sex`` value ``"X"`` without an entry,
        the result is ``{"sex": {"X": ["female", "male"]}}``.
    """
    inputs = mapping_inputs(dataframe, plan, catalog)
    if not inputs:
        return {}
    counts = ", ".join(f"{item.column_name!r} ({len(item.source_values)})" for item in inputs)
    logger.user.warning(
        f"data_to_sampler_value_mapping has no entry for dataset values that match no value of the {sampler_name} "
        f"sampler, counted by depends_on column: {counts}. Rows with these values are not conditioned on them."
    )
    return {item.column_name: {value: list(item.sampler_values) for value in item.source_values} for item in inputs}


def without_non_categorical_dependencies(
    dataframe: pd.DataFrame, plan: PiiReplacementPlan, catalog: SamplerValueCatalog
) -> PiiReplacementPlan:
    """Return ``plan`` without sampler-conditioned dependencies on non-categorical columns, warning about each.

    This is the lenient form of the categorical check in ``distinct_source_values`` for a plan the user
    supplied. A column with too many distinct values or values too long to map is dropped from every
    ``depends_on`` and from ``data_to_sampler_value_mapping``, so replacements that depended on it are
    drawn without it.
    """
    dropped = {
        column_name: issue
        for column_name, entity_type in conditioner_columns(plan).items()
        if catalog.get(entity_type) and (issue := _non_categorical_issue(dataframe[column_name])) is not None
    }
    if not dropped:
        return plan
    for column_name, issue in dropped.items():
        logger.user.warning(f"Ignoring the dependency on column {column_name!r}: {issue}.")
    columns_to_replace = [
        spec.model_copy(
            update={
                "depends_on": [dependency for dependency in spec.depends_on if dependency.column_name not in dropped]
            }
        )
        for spec in plan.columns_to_replace
    ]
    mapping = {
        column_name: entries
        for column_name, entries in plan.data_to_sampler_value_mapping.items()
        if column_name not in dropped
    }
    fields = {name: getattr(plan, name) for name in PiiReplacementPlan.model_fields}
    return PiiReplacementPlan.model_validate(
        {**fields, "columns_to_replace": columns_to_replace, "data_to_sampler_value_mapping": mapping}
    )


def unmatched_dependency_values(
    dataframe: pd.DataFrame,
    plan: PiiReplacementPlan,
    catalog: SamplerValueCatalog,
) -> dict[str, int]:
    """Count dataset values per conditioner column that have no mapping entry and no sampler match.

    Values are compared ignoring case. Columns whose entity type has no sampler catalog are skipped.

    Args:
        dataframe: Data whose conditioner-column values are checked.
        plan: Replacement plan whose ``data_to_sampler_value_mapping`` supplies the entries.
        catalog: Sampler values available for each conditioner entity type.

    Returns:
        The count of uncovered distinct values for each column that has any.

    Example:
        With the catalog ``{GENDER: ("female", "male")}``, a ``sex`` column
        holding ``"Female"``, ``"Woman"`` and ``"X"``, and the mapping
        ``{"sex": {"Woman": ["female"]}}``, the result is ``{"sex": 1}``:
        ``"Female"`` matches the catalog, ``"Woman"`` has an entry, and ``"X"``
        has neither.
    """
    return {item.column_name: len(item.source_values) for item in mapping_inputs(dataframe, plan, catalog)}


def built_in_value_mappings(
    dataframe: pd.DataFrame,
    plan: PiiReplacementPlan,
    catalog: SamplerValueCatalog,
) -> DataToSamplerValueMapping:
    """Return ``BUILT_IN_VALUE_MAPPINGS`` entries for dataset values the plan's mapping does not cover.

    An entry is added only when the dataset value has no case-insensitive
    sampler match, the plan's mapping has no entry for it, and its built-in
    target is in the sampler catalog.

    Example:
        With the catalog ``{GENDER: ("female", "male")}`` and a ``sex`` column
        holding ``"F"``, ``"Female"`` and ``"Man"``, the result is
        ``{"sex": {"F": ["female"], "Man": ["male"]}}``. The fr_FR Nemotron-Personas
        catalog ``{GENDER: ("female", "femme", "homme", "male")}`` gives the same result.
    """
    entries: DataToSamplerValueMapping = {}
    for column_name, entity_type in conditioner_columns(plan).items():
        variants = BUILT_IN_VALUE_MAPPINGS.get(entity_type)
        sampler_values = {sampler_value.casefold(): sampler_value for sampler_value in catalog.get(entity_type, ())}
        if not variants or not sampler_values:
            continue
        covered = {source.casefold() for source in plan.data_to_sampler_value_mapping.get(column_name, {})}
        for value in distinct_source_values(dataframe[column_name]):
            key = value.casefold()
            target = variants.get(key)
            if key in sampler_values or key in covered or target is None or target not in sampler_values:
                continue
            entries.setdefault(column_name, {})[value] = [sampler_values[target]]
    return entries


def distinct_source_values(values: pd.Series) -> tuple[str, ...]:
    """Return distinct non-missing values in the form replacement uses to look up mapping keys.

    Raises:
        ParameterError: If the column has too many distinct values or values too long to map.
    """
    if (issue := _non_categorical_issue(values)) is not None:
        raise ParameterError(
            f"dependency column {values.name!r} {issue}; sampler-conditioned dependencies such as gender or "
            "ethnic background must be categorical"
        )
    distinct: dict[str, str] = {}
    for raw in values.dropna().unique().tolist():
        value = readable_value(canonicalize_scalar(raw))
        distinct.setdefault(value.casefold(), value)
    return tuple(distinct.values())


def _non_categorical_issue(values: pd.Series) -> str | None:
    """Return why ``values`` cannot condition the sampler, or ``None`` if they can."""
    unique = values.dropna().unique()
    if len(unique) > MAX_CONDITIONER_DISTINCT_VALUES:
        return f"has more than {MAX_CONDITIONER_DISTINCT_VALUES} distinct values"
    if any(len(readable_value(canonicalize_scalar(raw))) > MAX_MAPPING_SOURCE_VALUE_LENGTH for raw in unique.tolist()):
        return f"has values longer than {MAX_MAPPING_SOURCE_VALUE_LENGTH} characters"
    return None
