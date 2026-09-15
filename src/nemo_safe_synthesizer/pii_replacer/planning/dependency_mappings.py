# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve dataset dependency columns against sampler value catalogs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ...config.replace_pii import DataToSamplerValueMapping, EntityType, PiiReplacementPlan
from ...errors import ParameterError
from ..sampler_catalog import SamplerValueCatalog

MAX_MAPPING_SOURCE_VALUE_LENGTH = 128


@dataclass(frozen=True, slots=True)
class DependencyMappingInput:
    """Unmatched values for one dependency column and sampler value catalog."""

    column_name: str
    entity_type: EntityType
    source_values: tuple[str, ...]
    sampler_values: tuple[str, ...]


def dependency_columns(plan: PiiReplacementPlan) -> dict[str, EntityType]:
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
    for column_name, entity_type in dependency_columns(plan).items():
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


def validate_data_to_sampler_value_mapping(
    plan: PiiReplacementPlan,
    mappings: DataToSamplerValueMapping,
    catalog: SamplerValueCatalog,
) -> None:
    """Validate authoritative mappings against plan columns and supported sampler values."""
    columns = dependency_columns(plan)
    for column_name, value_mappings in mappings.items():
        entity_type = columns.get(column_name)
        if entity_type is None:
            raise ParameterError(
                f"data_to_sampler_value_mapping column {column_name!r} is not used as a dependency in the replacement plan"
            )
        supported = catalog.get(entity_type)
        if not supported:
            continue
        supported_keys = {sampler_value.casefold() for sampler_value in supported}
        for targets in value_mappings.values():
            if targets is None:
                continue
            unknown = sorted(target for target in targets if target.casefold() not in supported_keys)
            if unknown:
                raise ParameterError(
                    f"data_to_sampler_value_mapping for column {column_name!r} contains values not supported "
                    f"by the configured sampler: {unknown}"
                )


def _distinct_source_values(values: pd.Series) -> tuple[str, ...]:
    distinct: dict[str, str] = {}
    for raw in values.dropna().unique().tolist():
        value = str(raw)
        if len(value) > MAX_MAPPING_SOURCE_VALUE_LENGTH:
            raise ParameterError(
                "automatic dependency mapping requires categorical values no longer than "
                f"{MAX_MAPPING_SOURCE_VALUE_LENGTH} characters"
            )
        distinct.setdefault(value.casefold(), value)
    return tuple(distinct.values())
