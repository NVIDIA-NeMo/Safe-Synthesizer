# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.replace_pii import (
    EntityType,
    PiiColumnPlan,
    PiiReplacementPlan,
    PiiSamplerBackend,
)
from nemo_safe_synthesizer.pii_replacer.replacement.collisions import CandidateChooser, ReplacementPool
from nemo_safe_synthesizer.pii_replacer.replacement.executor import StructuredReplacementExecutor
from nemo_safe_synthesizer.pii_replacer.replacement.generation import ReplacementGenerationRequest


class _ScriptedGenerator:
    backend = PiiSamplerBackend.FAKER

    def __init__(self, *values: str) -> None:
        self._values = list(values)
        self.requests: list[ReplacementGenerationRequest] = []

    def generate(self, request: ReplacementGenerationRequest) -> str:
        self.requests.append(request)
        return self._values.pop(0)


def _request(entity_type: EntityType, original: str) -> ReplacementGenerationRequest:
    return ReplacementGenerationRequest(
        entity_type=entity_type, original_value=original, effective_dependency_tuple=(), pattern=None, seed=0
    )


def _execute(dataframe: pd.DataFrame, *specs: PiiColumnPlan, generator: _ScriptedGenerator) -> pd.DataFrame:
    executor = StructuredReplacementExecutor(
        PiiReplacementPlan(columns_to_replace=list(specs)), generator, group_column=None, base_seed=42
    )
    return executor.execute(dataframe).dataframe


@pytest.mark.unit
class TestReplacementPool:
    def test_identifier_values_are_compared_ignoring_case(self) -> None:
        pool = ReplacementPool()
        pool.reserve_originals(EntityType.EMAIL, ["Ada@x.com"])
        pool.record_used(EntityType.EMAIL, "KATE@y.com")

        assert pool.is_original(EntityType.EMAIL, "ada@X.COM")
        assert pool.is_used(EntityType.EMAIL, "kate@Y.com")
        assert not pool.is_taken(EntityType.PHONE_NUMBER, "ada@x.com")

    def test_names_are_not_tracked(self) -> None:
        pool = ReplacementPool()
        pool.reserve_originals(EntityType.FIRST_NAME, ["Ada"])
        pool.record_used(EntityType.FIRST_NAME, "Kathleen")

        assert not pool.is_taken(EntityType.FIRST_NAME, "Ada")
        assert not pool.is_taken(EntityType.FIRST_NAME, "Kathleen")


@pytest.mark.unit
class TestCandidateChooser:
    def test_fallbacks_prefer_a_repeat_then_another_original(self) -> None:
        pool = ReplacementPool()
        pool.reserve_originals(EntityType.SSN, ["111-11-1111", "222-22-2222"])
        pool.record_used(EntityType.SSN, "333-33-3333")
        chooser = CandidateChooser(pool, EntityType.SSN, "111-11-1111")
        request = _request(EntityType.SSN, "111-11-1111")

        assert not chooser.offer(request, "111-11-1111")
        assert not chooser.offer(request, "222-22-2222")
        assert chooser.fallback() == ("222-22-2222", "original")
        assert not chooser.offer(request, "333-33-3333")
        assert chooser.fallback() == ("333-33-3333", "repeat")
        assert chooser.offer(request, "444-44-4444")

    def test_one_to_one_target_rejects_taken_values_without_a_fallback(self) -> None:
        pool = ReplacementPool()
        chooser = CandidateChooser(pool, EntityType.FIRST_NAME, "Ada", one_to_one={"grace"})
        request = _request(EntityType.FIRST_NAME, "Ada")

        assert not chooser.offer(request, "Grace")
        assert chooser.fallback() is None
        assert chooser.offer(request, "Kathleen")


@pytest.mark.unit
class TestCrossColumnIdentifierCollisions:
    def test_replacements_avoid_originals_and_replacements_of_other_columns_of_the_same_type(self) -> None:
        dataframe = pd.DataFrame({"ssn": ["111-11-1111"], "spouse_ssn": ["222-22-2222"]})
        # spouse_ssn is first offered ssn's original, then ssn's replacement, then a new value.
        generator = _ScriptedGenerator("333-33-3333", "111-11-1111", "333-33-3333", "444-44-4444")

        result = _execute(
            dataframe,
            PiiColumnPlan(column_name="ssn", entity_type=EntityType.SSN),
            PiiColumnPlan(column_name="spouse_ssn", entity_type=EntityType.SSN),
            generator=generator,
        )

        assert result.iloc[0].tolist() == ["333-33-3333", "444-44-4444"]

    def test_mask_allocation_skips_values_used_by_another_column(self) -> None:
        dataframe = pd.DataFrame({"user": ["U-1", "U-2"], "manager": ["U-3", "U-4"]})
        specs = (
            PiiColumnPlan(column_name="user", entity_type=EntityType.UNIQUE_IDENTIFIER, pattern="U-#"),
            PiiColumnPlan(column_name="manager", entity_type=EntityType.UNIQUE_IDENTIFIER, pattern="U-#"),
        )

        result = _execute(dataframe, *specs, generator=_ScriptedGenerator())

        replaced = result["user"].tolist() + result["manager"].tolist()
        assert len(set(replaced)) == 4
        assert not set(replaced) & {"U-1", "U-2", "U-3", "U-4"}


@pytest.mark.unit
class TestCrossColumnSharing:
    def test_the_same_value_in_two_columns_of_one_type_and_format_shares_a_replacement(self) -> None:
        dataframe = pd.DataFrame({"email": ["ada@x.com", "bo@x.com"], "backup_email": ["ADA@X.COM", "cy@x.com"]})
        generator = _ScriptedGenerator("kate@y.com", "jo@y.com", "max@y.com")

        result = _execute(
            dataframe,
            PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL),
            PiiColumnPlan(column_name="backup_email", entity_type=EntityType.EMAIL),
            generator=generator,
        )

        assert result.to_dict("list") == {
            "email": ["kate@y.com", "jo@y.com"],
            "backup_email": ["KATE@Y.COM", "max@y.com"],
        }
        assert len(generator.requests) == 3

    def test_columns_with_different_patterns_are_replaced_separately(self) -> None:
        dataframe = pd.DataFrame({"phone": ["(555) 123-4567"], "work_phone": ["(555) 123-4567"]})
        specs = (
            PiiColumnPlan(column_name="phone", entity_type=EntityType.PHONE_NUMBER, pattern="(###) ###-####"),
            PiiColumnPlan(column_name="work_phone", entity_type=EntityType.PHONE_NUMBER),
        )

        result = _execute(dataframe, *specs, generator=_ScriptedGenerator("555.987.6543"))

        assert result["work_phone"].tolist() == ["555.987.6543"]
        assert result["phone"].iloc[0] != "555.987.6543"
