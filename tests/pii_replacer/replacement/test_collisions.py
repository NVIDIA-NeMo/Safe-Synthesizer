# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.replace_pii import (
    EntityType,
    PiiColumnPlan,
    PiiReplacementPlan,
    PiiSamplerBackend,
)
from nemo_safe_synthesizer.errors import GenerationError, ParameterError
from nemo_safe_synthesizer.pii_replacer.replacement.collisions import CandidateChooser, all_distinct
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
class TestAllDistinct:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [(["A-1", "a-2"], True), (["A-1", "a-1"], False), ([], False)],
    )
    def test_values_are_compared_ignoring_case(self, values: list[str], expected: bool) -> None:
        assert all_distinct(values) is expected


@pytest.mark.unit
class TestCandidateChooser:
    def test_another_original_is_accepted_and_a_used_value_only_repeats(self) -> None:
        chooser = CandidateChooser(EntityType.SSN, "111-11-1111", used={"333-33-3333"})
        request = _request(EntityType.SSN, "111-11-1111")

        assert not chooser.offer(request, "111-11-1111")
        assert not chooser.offer(request, "333-33-3333")
        assert chooser.fallback() == ("333-33-3333", "repeat")
        assert chooser.offer(request, "222-22-2222")

    def test_without_a_used_set_any_value_but_the_own_original_is_accepted(self) -> None:
        chooser = CandidateChooser(EntityType.SSN, "111-11-1111")
        request = _request(EntityType.SSN, "111-11-1111")

        assert not chooser.offer(request, "111-11-1111")
        assert chooser.offer(request, "222-22-2222")

    def test_one_to_one_target_rejects_taken_values_without_a_fallback(self) -> None:
        chooser = CandidateChooser(EntityType.FIRST_NAME, "Ada", one_to_one={"grace"})
        request = _request(EntityType.FIRST_NAME, "Ada")

        assert not chooser.offer(request, "Grace")
        assert chooser.fallback() is None
        assert chooser.offer(request, "Kathleen")


@pytest.mark.unit
class TestDistinctColumns:
    def test_a_column_of_distinct_values_keeps_its_replacements_distinct(self) -> None:
        dataframe = pd.DataFrame({"ssn": ["111-11-1111", "222-22-2222"]})
        generator = _ScriptedGenerator("333-33-3333", "333-33-3333", "444-44-4444")

        result = _execute(dataframe, PiiColumnPlan(column_name="ssn", entity_type=EntityType.SSN), generator=generator)

        assert result["ssn"].tolist() == ["333-33-3333", "444-44-4444"]

    def test_a_column_with_repeated_values_may_repeat_replacements(self) -> None:
        dataframe = pd.DataFrame({"ssn": ["111-11-1111", "111-11-1111", "222-22-2222"]})
        generator = _ScriptedGenerator("333-33-3333", "333-33-3333", "333-33-3333")

        result = _execute(dataframe, PiiColumnPlan(column_name="ssn", entity_type=EntityType.SSN), generator=generator)

        assert result["ssn"].tolist() == ["333-33-3333"] * 3

    def test_a_replacement_may_equal_another_rows_or_columns_original(self) -> None:
        dataframe = pd.DataFrame({"ssn": ["111-11-1111"], "spouse_ssn": ["222-22-2222"]})
        generator = _ScriptedGenerator("222-22-2222", "111-11-1111")

        result = _execute(
            dataframe,
            PiiColumnPlan(column_name="ssn", entity_type=EntityType.SSN),
            PiiColumnPlan(column_name="spouse_ssn", entity_type=EntityType.SSN),
            generator=generator,
        )

        assert result.iloc[0].tolist() == ["222-22-2222", "111-11-1111"]

    def test_mask_allocation_keeps_a_distinct_column_distinct(self) -> None:
        dataframe = pd.DataFrame({"user": [f"U-{value}" for value in range(9)]})
        spec = PiiColumnPlan(column_name="user", entity_type=EntityType.UNIQUE_IDENTIFIER, pattern="U-#")

        result = _execute(dataframe, spec, generator=_ScriptedGenerator())

        assert result["user"].nunique() == 9
        assert all(new != old for new, old in zip(result["user"], dataframe["user"], strict=True))


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

    @pytest.mark.parametrize(
        ("account", "parent"),
        [
            # Row 0 shares account_id's value into parent_id; later rows must not get it again.
            (
                [f"{row:03d}" for row in range(200)],
                [f"{row:03d}" if row % 2 else f"{500 + row:03d}" for row in range(200)],
            ),
            # Rows 100-199 share values parent_id could already have given rows 0-99.
            (
                [f"{row:03d}" for row in range(200)],
                [f"{500 + row:03d}" for row in range(100)] + [f"{row:03d}" for row in range(100, 200)],
            ),
        ],
    )
    def test_shared_values_do_not_repeat_in_a_distinct_column(self, account: list[str], parent: list[str]) -> None:
        dataframe = pd.DataFrame({"account_id": account, "parent_id": parent})
        specs = [
            PiiColumnPlan(column_name=column, entity_type=EntityType.UNIQUE_IDENTIFIER, pattern="###")
            for column in ("account_id", "parent_id")
        ]

        result = _execute(dataframe, *specs, generator=_ScriptedGenerator())

        assert result["account_id"].is_unique
        assert result["parent_id"].is_unique
        shared_rows = dataframe["account_id"] == dataframe["parent_id"]
        assert (result.loc[shared_rows, "account_id"] == result.loc[shared_rows, "parent_id"]).all()

    def test_a_distinct_column_does_not_take_a_shared_value_it_already_has(self) -> None:
        # ssn repeats a value, so it may repeat replacements: rows 0 and 1 both get 444. alt_ssn must stay
        # distinct, so it shares 444 in row 0 but generates a new value in row 1.
        dataframe = pd.DataFrame(
            {
                "ssn": ["111-11-1111", "222-22-2222", "111-11-1111"],
                "alt_ssn": ["111-11-1111", "222-22-2222", "333-33-3333"],
            }
        )
        generator = _ScriptedGenerator("444-44-4444", "444-44-4444", "777-77-7777", "888-88-8888", "999-99-9999")
        specs = [PiiColumnPlan(column_name=column, entity_type=EntityType.SSN) for column in ("ssn", "alt_ssn")]

        result = _execute(dataframe, *specs, generator=generator)

        assert result.to_dict("list") == {
            "ssn": ["444-44-4444", "444-44-4444", "777-77-7777"],
            "alt_ssn": ["444-44-4444", "888-88-8888", "999-99-9999"],
        }

    def test_columns_with_different_patterns_are_replaced_separately(self) -> None:
        dataframe = pd.DataFrame({"phone": ["(555) 123-4567"], "work_phone": ["(555) 123-4567"]})
        specs = (
            PiiColumnPlan(column_name="phone", entity_type=EntityType.PHONE_NUMBER, pattern="(###) ###-####"),
            PiiColumnPlan(column_name="work_phone", entity_type=EntityType.PHONE_NUMBER),
        )

        result = _execute(dataframe, *specs, generator=_ScriptedGenerator("555.987.6543"))

        assert result["work_phone"].tolist() == ["555.987.6543"]
        assert result["phone"].iloc[0] != "555.987.6543"


@pytest.mark.unit
class TestGroupingColumnIsOneToOne:
    @staticmethod
    def _execute_grouped(dataframe: pd.DataFrame, spec: PiiColumnPlan, generator: _ScriptedGenerator) -> pd.DataFrame:
        executor = StructuredReplacementExecutor(
            PiiReplacementPlan(columns_to_replace=[spec]), generator, group_column=spec.column_name, base_seed=42
        )
        return executor.execute(dataframe).dataframe

    def test_a_name_another_group_already_got_is_rejected(self) -> None:
        dataframe = pd.DataFrame({"owner": ["Ada Lovelace", "Ada Lovelace", "Bo Diddley"]})
        generator = _ScriptedGenerator("Kate Moss", "KATE MOSS", "Grace Hopper")

        result = self._execute_grouped(
            dataframe, PiiColumnPlan(column_name="owner", entity_type=EntityType.FULL_NAME), generator
        )

        assert result["owner"].tolist() == ["Kate Moss", "Kate Moss", "Grace Hopper"]

    def test_running_out_of_distinct_values_fails_instead_of_merging_groups(self) -> None:
        dataframe = pd.DataFrame({"owner": ["Ada", "Bo"]})
        generator = _ScriptedGenerator(*["Kate"] * 60)

        with pytest.raises(GenerationError, match="two groups would merge"):
            self._execute_grouped(
                dataframe, PiiColumnPlan(column_name="owner", entity_type=EntityType.FIRST_NAME), generator
            )

    def test_a_mask_too_small_for_every_group_is_widened(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING)
        dataframe = pd.DataFrame({"gid": [f"ID-{group}" for group in range(10) for _ in range(3)]})
        spec = PiiColumnPlan(column_name="gid", entity_type=EntityType.UNIQUE_IDENTIFIER, pattern="ID-#")

        result = self._execute_grouped(dataframe, spec, _ScriptedGenerator())

        assert result["gid"].nunique() == 10
        assert result["gid"].str.fullmatch(r"ID-\d\d").all()
        assert any("Widened grouping-column identifiers" in record.getMessage() for record in caplog.records)


@pytest.mark.unit
class TestMaskWidening:
    def test_a_mask_without_a_multi_choice_position_fails_instead_of_widening_forever(self) -> None:
        dataframe = pd.DataFrame({"gid": ["ID-0", "ID-1"]})
        spec = PiiColumnPlan(column_name="gid", entity_type=EntityType.UNIQUE_IDENTIFIER, pattern="ID-[0]")
        executor = StructuredReplacementExecutor(
            PiiReplacementPlan(columns_to_replace=[spec]), _ScriptedGenerator(), group_column="gid", base_seed=42
        )

        with pytest.raises(ParameterError, match="use a pattern with more generated positions"):
            executor.execute(dataframe)


@pytest.mark.unit
class TestPatternsSurviveLetterCase:
    def test_occurrence_case_is_dropped_when_it_would_break_the_mask(self) -> None:
        dataframe = pd.DataFrame({"code": ["B"]})
        spec = PiiColumnPlan(column_name="code", entity_type=EntityType.UNIQUE_IDENTIFIER, pattern="[aB]")

        result = _execute(dataframe, spec, generator=_ScriptedGenerator())

        assert result["code"].tolist() == ["a"]

    def test_a_mask_allowing_both_cases_never_restores_the_original(self) -> None:
        dataframe = pd.DataFrame({"code": ["a"] * 20})
        spec = PiiColumnPlan(column_name="code", entity_type=EntityType.UNIQUE_IDENTIFIER, pattern="[aAb]")

        result = _execute(dataframe, spec, generator=_ScriptedGenerator())

        assert result["code"].tolist() == ["b"] * 20

    def test_a_distinct_column_stays_distinct_after_letter_case_is_restored(self) -> None:
        # Lowercase originals get every replacement in lowercase, so "Ab" and "aB" would both become "ab".
        dataframe = pd.DataFrame({"key": [f"{first}{second}" for first in "abcdefghij" for second in "abcdefghij"]})
        spec = PiiColumnPlan(column_name="key", entity_type=EntityType.API_KEY, pattern="**")

        result = _execute(dataframe, spec, generator=_ScriptedGenerator())

        assert result["key"].str.casefold().nunique() == 100
        assert (result["key"].str.casefold() != dataframe["key"]).all()
