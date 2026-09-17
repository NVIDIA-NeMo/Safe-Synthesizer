# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import (
    ENTITY_BY_TYPE,
    ConditioningColumn,
    EntityType,
    PiiColumnPlan,
    PiiReplacementPlan,
)
from nemo_safe_synthesizer.errors import InternalError, ParameterError
from nemo_safe_synthesizer.pii_replacer.planning import validate_plan
from nemo_safe_synthesizer.pii_replacer.planning.patterns import NAME_PART_PLACEHOLDERS


@pytest.fixture
def fixture_pii_df() -> pd.DataFrame:
    """Return representative structured PII columns for plan validation."""
    return pd.DataFrame(
        {
            "patient_id": [1, 2],
            "name": ["Ada Lovelace", "Grace Hopper"],
            "email": ["ada@example.com", "grace@example.com"],
            "phone": ["+1-202-555-0101", "+1-303-555-0102"],
            "dob": ["12/10/1815", "12/09/1906"],
        }
    )


@pytest.mark.unit
class TestValidatePlan:
    def test_accepts_valid_dependencies_and_patterns(self, fixture_pii_df: pd.DataFrame) -> None:
        plan = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(
                    column_name="name",
                    entity_type=EntityType.FULL_NAME,
                    pattern="{First} {Last}",
                ),
                PiiColumnPlan(
                    column_name="email",
                    entity_type=EntityType.EMAIL,
                    pattern="{first}@{domain}",
                    depends_on=[ConditioningColumn(column_name="name")],
                ),
                PiiColumnPlan(
                    column_name="phone",
                    entity_type=EntityType.PHONE_NUMBER,
                    pattern="+1-###-555-####",
                ),
                PiiColumnPlan(
                    column_name="dob",
                    entity_type=EntityType.DATE_OF_BIRTH,
                    pattern="%m/%d/%Y",
                ),
            ]
        )

        validate_plan(fixture_pii_df, plan, data_config=DataParameters())

    def test_accepts_timezone_bearing_strftime_pattern(self) -> None:
        dataframe = pd.DataFrame({"timestamp": ["2001-02-03+0000", "1999-12-31-0500"]})
        plan = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(
                    column_name="timestamp",
                    entity_type=EntityType.DATE_OF_BIRTH,
                    pattern="%Y-%m-%d%z",
                )
            ]
        )

        validate_plan(dataframe, plan, data_config=DataParameters())

    @pytest.mark.parametrize(
        ("pattern", "value"),
        [
            ("#", "7"),
            ("^", "A"),
            ("@", "a"),
            ("&", "7"),
            ("%", "a"),
            ("*", "Z"),
            ("[abc]", "b"),
        ],
    )
    def test_accepts_every_documented_character_mask_token(self, pattern: str, value: str) -> None:
        dataframe = pd.DataFrame({"identifier": [value]})
        plan = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(
                    column_name="identifier",
                    entity_type=EntityType.UNIQUE_IDENTIFIER,
                    pattern=pattern,
                )
            ]
        )

        validate_plan(dataframe, plan, data_config=DataParameters())

    @pytest.mark.parametrize("escaped", ["#", "^", "@", "%", "&", "*", "[", "]", "\\"])
    def test_accepts_every_documented_character_mask_escape(self, escaped: str) -> None:
        dataframe = pd.DataFrame({"identifier": [f"{escaped}7"]})
        plan = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(
                    column_name="identifier",
                    entity_type=EntityType.UNIQUE_IDENTIFIER,
                    pattern=f"\\{escaped}#",
                )
            ]
        )

        validate_plan(dataframe, plan, data_config=DataParameters())

    @pytest.mark.parametrize("pattern", [r"\d#", r"\s#", "#\\"])
    def test_rejects_unsupported_or_incomplete_character_mask_escape(self, pattern: str) -> None:
        dataframe = pd.DataFrame({"identifier": ["d7"]})
        plan = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(
                    column_name="identifier",
                    entity_type=EntityType.UNIQUE_IDENTIFIER,
                    pattern=pattern,
                )
            ]
        )

        with pytest.raises(ParameterError, match="unsupported character|trailing"):
            validate_plan(dataframe, plan, data_config=DataParameters())

    @pytest.mark.parametrize("placeholder", NAME_PART_PLACEHOLDERS)
    def test_accepts_every_documented_name_part_placeholder(self, placeholder: str) -> None:
        definition = NAME_PART_PLACEHOLDERS[placeholder]
        is_email_only = definition.part in {"domain", "organization"}
        pattern = f"x@{placeholder}" if is_email_only else placeholder
        value = "x@example.com" if is_email_only else ("A" if definition.initial else "Ada")
        dataframe = pd.DataFrame({"value": [value]})
        plan = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(
                    column_name="value",
                    entity_type=EntityType.EMAIL if is_email_only else EntityType.FULL_NAME,
                    pattern=pattern,
                )
            ]
        )

        validate_plan(dataframe, plan, data_config=DataParameters())

    def test_rejects_undocumented_name_part_case_variant(self) -> None:
        dataframe = pd.DataFrame({"name": ["Ada"]})
        plan = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(
                    column_name="name",
                    entity_type=EntityType.FULL_NAME,
                    pattern="{fIrSt}",
                )
            ]
        )

        with pytest.raises(ParameterError, match="unknown placeholder"):
            validate_plan(dataframe, plan, data_config=DataParameters())

    def test_accepts_email_name_parts_pattern_with_digit_token(self) -> None:
        dataframe = pd.DataFrame({"email": ["ada1@example.com", "grace2@example.com"]})
        plan = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(
                    column_name="email",
                    entity_type=EntityType.EMAIL,
                    pattern="{first}#@{domain}",
                )
            ]
        )

        validate_plan(dataframe, plan, data_config=DataParameters())

    def test_accepts_email_organization_placeholder(self) -> None:
        dataframe = pd.DataFrame({"email": ["ada@mail.example.org"]})
        plan = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(
                    column_name="email",
                    entity_type=EntityType.EMAIL,
                    pattern="{first}@mail.{organization}.org",
                )
            ]
        )

        validate_plan(dataframe, plan, data_config=DataParameters())

    def test_reports_missing_replacement_and_dependency_columns(self, fixture_pii_df: pd.DataFrame) -> None:
        plan = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(column_name="missing", entity_type=EntityType.EMAIL),
                PiiColumnPlan(
                    column_name="name",
                    entity_type=EntityType.FULL_NAME,
                    depends_on=[ConditioningColumn(column_name="gender", entity_type=EntityType.GENDER)],
                ),
            ]
        )

        with pytest.raises(
            ParameterError,
            match="(?s)replacement column 'missing'.*depends_on column 'gender'",
        ):
            validate_plan(fixture_pii_df, plan, data_config=DataParameters())

    def test_allows_replacing_the_group_column(self, fixture_pii_df: pd.DataFrame) -> None:
        plan = PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="patient_id", entity_type=EntityType.UNIQUE_IDENTIFIER)],
        )

        validate_plan(
            fixture_pii_df,
            plan,
            data_config=DataParameters(group_training_examples_by="patient_id"),
        )

    def test_rejects_replacing_an_ordering_column(self, fixture_pii_df: pd.DataFrame) -> None:
        dataframe = fixture_pii_df.assign(event_index=[0, 0])
        plan = PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="event_index", entity_type=EntityType.UNIQUE_IDENTIFIER)],
        )

        with pytest.raises(ParameterError, match="protected column 'event_index' cannot be replaced"):
            validate_plan(
                dataframe,
                plan,
                data_config=DataParameters(
                    group_training_examples_by="patient_id",
                    order_training_examples_by="event_index",
                ),
            )

    def test_configured_group_column_must_exist(self, fixture_pii_df: pd.DataFrame) -> None:
        plan = PiiReplacementPlan()

        # With no group column, record-consistent replacement needs no group validation.
        validate_plan(fixture_pii_df, plan, data_config=DataParameters())

        with pytest.raises(ParameterError, match="group column 'missing_group' is not present"):
            validate_plan(
                fixture_pii_df,
                plan,
                data_config=DataParameters(group_training_examples_by="missing_group"),
            )

    def test_rejects_pattern_below_coverage_threshold(self, fixture_pii_df: pd.DataFrame) -> None:
        plan = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(
                    column_name="phone",
                    entity_type=EntityType.PHONE_NUMBER,
                    pattern="###-###-####",
                )
            ]
        )

        with pytest.raises(ParameterError, match="covers 0.0%.*at least 85%"):
            validate_plan(fixture_pii_df, plan, data_config=DataParameters())

    def test_missing_matcher_is_an_internal_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entity = ENTITY_BY_TYPE[EntityType.UNIQUE_IDENTIFIER]
        monkeypatch.setitem(ENTITY_BY_TYPE, EntityType.UNIQUE_IDENTIFIER, replace(entity, pattern_syntax=None))
        dataframe = pd.DataFrame({"identifier": ["7"]})
        plan = PiiReplacementPlan.model_construct(
            columns_to_replace=[
                PiiColumnPlan.model_construct(
                    column_name="identifier",
                    entity_type=EntityType.UNIQUE_IDENTIFIER,
                    pattern="#",
                    depends_on=[],
                )
            ],
        )

        with pytest.raises(InternalError, match="has no matcher"):
            validate_plan(dataframe, plan, data_config=DataParameters())
