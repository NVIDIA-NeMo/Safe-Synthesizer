# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve dataset dependency columns against sampler value catalogs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ...config.replace_pii import EntityType, PiiReplacementPlan
from ...errors import ParameterError
from ...observability import get_logger
from ..replacement.canonicalization import canonicalize_scalar, readable_value
from ..sampler_catalog import SamplerValueCatalog

logger = get_logger(__name__)

MAX_MAPPING_SOURCE_VALUE_LENGTH = 128


@dataclass(frozen=True, slots=True)
class DependencyMappingInput:
    """Unmatched values for one dependency column and sampler value catalog."""

    column_name: str
    entity_type: EntityType
    source_values: tuple[str, ...]
    sampler_values: tuple[str, ...]


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
    """Return non-identity dependency values that require automatic mapping."""
    inputs: list[DependencyMappingInput] = []
    for column_name, entity_type in conditioner_columns(plan).items():
        sampler_values = catalog.get(entity_type)
        if not sampler_values:
            continue
        sampler_keys = {sampler_value.casefold() for sampler_value in sampler_values}
        source_values = _distinct_source_values(dataframe[column_name])
        unmatched = tuple(value for value in source_values if value.casefold() not in sampler_keys)
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


def validate_data_to_sampler_value_mapping(plan: PiiReplacementPlan, catalog: SamplerValueCatalog) -> None:
    """Validate the plan's value mappings against its conditioner columns and the sampler catalog.

    A mapping for a column whose entity type the sampler has no values for is
    kept but logged as ignored: the configured sampler cannot condition on it,
    but the same plan may later run with a sampler that can.

    Args:
        plan: Replacement plan whose ``data_to_sampler_value_mapping`` is checked.
        catalog: Sampler values available for each conditioner entity type.

    Raises:
        ParameterError: If a mapped column is not a conditioner column, or a
            mapped sampler value is not in the catalog for that column.

    Example:
        With a plan where ``first_name`` depends on ``sex`` (``gender``), the
        catalog ``{GENDER: ("female", "male")}`` accepts
        ``{"sex": {"Woman": ["female"]}}`` and rejects
        ``{"sex": {"Woman": ["woman"]}}``.
    """
    columns = conditioner_columns(plan)
    for column_name, value_mappings in plan.data_to_sampler_value_mapping.items():
        entity_type = columns.get(column_name)
        if entity_type is None:
            raise ParameterError(
                f"data_to_sampler_value_mapping column {column_name!r} is not used as a dependency "
                "in the replacement plan"
            )
        supported = catalog.get(entity_type)
        if not supported:
            logger.user.warning(
                "Ignoring data_to_sampler_value_mapping for a column the configured sampler cannot condition on",
                extra={"column": column_name, "entity_type": entity_type.value},
            )
            continue
        supported_keys = {sampler_value.casefold() for sampler_value in supported}
        for targets in value_mappings.values():
            unknown = sorted(target for target in targets if target.casefold() not in supported_keys)
            if unknown:
                raise ParameterError(
                    f"data_to_sampler_value_mapping for column {column_name!r} contains values not supported "
                    f"by the configured sampler: {unknown}"
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
    mappings = plan.data_to_sampler_value_mapping
    unmatched: dict[str, int] = {}
    for item in mapping_inputs(dataframe, plan, catalog):
        mapped = {source.casefold() for source in mappings.get(item.column_name, {})}
        missing = sum(1 for value in item.source_values if value.casefold() not in mapped)
        if missing:
            unmatched[item.column_name] = missing
    return unmatched


def _distinct_source_values(values: pd.Series) -> tuple[str, ...]:
    """Return distinct non-missing values in the form replacement uses to look up mapping keys."""
    distinct: dict[str, str] = {}
    for raw in values.dropna().unique().tolist():
        value = readable_value(canonicalize_scalar(raw))
        if len(value) > MAX_MAPPING_SOURCE_VALUE_LENGTH:
            raise ParameterError(
                "automatic dependency mapping requires categorical values no longer than "
                f"{MAX_MAPPING_SOURCE_VALUE_LENGTH} characters"
            )
        distinct.setdefault(value.casefold(), value)
    return tuple(distinct.values())
