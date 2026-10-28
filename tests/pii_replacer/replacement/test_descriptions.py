# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.config.replace_pii import (
    ENTITIES,
    ConditioningColumn,
    EntityAction,
    EntityType,
    PatternSyntax,
    PiiColumnPlan,
    PiiSamplerBackend,
    PiiSamplerConfig,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.pii_replacer.replacement.descriptions import describe_replacement

_EXAMPLE_PATTERNS = {
    PatternSyntax.NAME_PARTS: "{First} {Last}",
    PatternSyntax.CHARACTER_MASK: "4###-####",
    PatternSyntax.STRFTIME: "%Y-%m-%d",
}
_REPLACEABLE = [entity for entity in ENTITIES if entity.action is not EntityAction.IDENTIFY_ONLY]


def _settings(backend: PiiSamplerBackend = PiiSamplerBackend.NEMOTRON_PERSONAS, locale: str = "en_US"):
    return ReplacePiiConfig(locale=locale, sampler=PiiSamplerConfig(backend=backend))


def _pattern(entity_type: EntityType, syntax: PatternSyntax) -> str:
    return "{first}.{last}@{domain}" if entity_type is EntityType.EMAIL else _EXAMPLE_PATTERNS[syntax]


@pytest.mark.unit
@pytest.mark.parametrize("backend", list(PiiSamplerBackend))
@pytest.mark.parametrize("entity", _REPLACEABLE, ids=lambda entity: entity.entity_type.value)
def test_every_replaceable_type_is_described(entity, backend) -> None:
    columns = [PiiColumnPlan(column_name="value", entity_type=entity.entity_type)]
    if entity.pattern_syntax is not None:
        pattern = _pattern(entity.entity_type, entity.pattern_syntax)
        columns.append(PiiColumnPlan(column_name="value", entity_type=entity.entity_type, pattern=pattern))

    for column in columns:
        assert describe_replacement(column, _settings(backend))


@pytest.mark.unit
def test_email_names_the_columns_it_is_built_from() -> None:
    column = PiiColumnPlan(
        column_name="email",
        entity_type=EntityType.EMAIL,
        depends_on=[ConditioningColumn(column_name="full_name", entity_type=EntityType.FULL_NAME)],
    )

    assert describe_replacement(column, _settings()) == "Built from full_name, keeping the original domain"


@pytest.mark.unit
def test_patterned_identifier_is_allocated_from_the_pattern() -> None:
    column = PiiColumnPlan(column_name="id", entity_type=EntityType.UNIQUE_IDENTIFIER, pattern="ID-####")

    assert describe_replacement(column, _settings()) == "New values that fit the pattern"


@pytest.mark.unit
def test_street_uses_faker_outside_en_us() -> None:
    column = PiiColumnPlan(column_name="address", entity_type=EntityType.STREET_ADDRESS)

    assert describe_replacement(column, _settings(locale="en_US")).startswith("Street sampled from Nemotron Personas")
    assert describe_replacement(column, _settings(locale="fr_FR")).startswith("Street sampled from Faker")
