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
    problems: list[str] = []
    for column_name, value_mappings in plan.data_to_sampler_value_mapping.items():
        entity_type = columns.get(column_name)
        if entity_type is None:
            raise ParameterError(
                f"data_to_sampler_value_mapping column {column_name!r} is not used as a dependency "
                "in the replacement plan"
            )
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
    if problems:
        raise ParameterError(
            f"data_to_sampler_value_mapping does not match the {sampler_name} sampler: {'; '.join(problems)}. "
            "The plan may have been written for another sampler or locale; change these entries, rerun discovery "
            "with replace_pii.replacement_plan set to auto_discovery, or set replace_pii.sampler and "
            "replace_pii.locale to the ones the plan was written for"
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
    """Return distinct non-missing values in the form replacement uses to look up mapping keys."""
    unique = values.dropna().unique()
    if len(unique) > MAX_CONDITIONER_DISTINCT_VALUES:
        raise ParameterError(
            f"dependency column {values.name!r} has more than {MAX_CONDITIONER_DISTINCT_VALUES} distinct values; "
            "sampler-conditioned dependencies such as gender or ethnic background must be categorical"
        )
    distinct: dict[str, str] = {}
    for raw in unique.tolist():
        value = readable_value(canonicalize_scalar(raw))
        if len(value) > MAX_MAPPING_SOURCE_VALUE_LENGTH:
            raise ParameterError(
                f"dependency column {values.name!r} has values longer than {MAX_MAPPING_SOURCE_VALUE_LENGTH} "
                "characters; sampler-conditioned dependencies such as gender or ethnic background must be categorical"
            )
        distinct.setdefault(value.casefold(), value)
    return tuple(distinct.values())
