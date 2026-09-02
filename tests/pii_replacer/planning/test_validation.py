# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import (
    ConditioningColumn,
    EntityType,
    PiiColumnPlan,
    PiiReplacementPlan,
    PiiReplacementScope,
)
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.planning import validate_plan


@pytest.fixture
def pii_df() -> pd.DataFrame:
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
    def test_accepts_valid_dependencies_and_patterns(self, pii_df: pd.DataFrame) -> None:
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

        validate_plan(pii_df, plan, data_config=DataParameters())

    def test_accepts_all_documented_character_mask_classes(self) -> None:
        dataframe = pd.DataFrame({"identifier": ["A7-a8", "Z0-z9"]})
        plan = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(
                    column_name="identifier",
                    entity_type=EntityType.UNIQUE_IDENTIFIER,
                    pattern="&&-%%",
                )
            ]
        )

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

    def test_reports_missing_replacement_and_dependency_columns(self, pii_df: pd.DataFrame) -> None:
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
            validate_plan(pii_df, plan, data_config=DataParameters())

    def test_allows_replacing_the_group_column(self, pii_df: pd.DataFrame) -> None:
        plan = PiiReplacementPlan(
            scope=PiiReplacementScope.GROUP,
            columns_to_replace=[PiiColumnPlan(column_name="patient_id", entity_type=EntityType.UNIQUE_IDENTIFIER)],
        )

        validate_plan(
            pii_df,
            plan,
            data_config=DataParameters(group_training_examples_by="patient_id"),
        )

    def test_rejects_replacing_an_ordering_column(self, pii_df: pd.DataFrame) -> None:
        dataframe = pii_df.assign(event_index=[0, 0])
        plan = PiiReplacementPlan(
            scope=PiiReplacementScope.GROUP,
            columns_to_replace=[PiiColumnPlan(column_name="event_index", entity_type=EntityType.UNIQUE_IDENTIFIER)],
        )

        with pytest.raises(ParameterError, match="structural column 'event_index' cannot be replaced"):
            validate_plan(
                dataframe,
                plan,
                data_config=DataParameters(
                    group_training_examples_by="patient_id",
                    order_training_examples_by="event_index",
                ),
            )

    def test_group_scope_requires_a_configured_existing_group_column(self, pii_df: pd.DataFrame) -> None:
        plan = PiiReplacementPlan(scope=PiiReplacementScope.GROUP)

        with pytest.raises(ParameterError, match="group_training_examples_by is not configured"):
            validate_plan(pii_df, plan, data_config=DataParameters())

        with pytest.raises(ParameterError, match="group column 'missing_group' is not present"):
            validate_plan(
                pii_df,
                plan,
                data_config=DataParameters(group_training_examples_by="missing_group"),
            )

    def test_rejects_self_dependency(self, pii_df: pd.DataFrame) -> None:
        plan = PiiReplacementPlan.model_construct(
            scope=PiiReplacementScope.DATAFRAME,
            columns_to_replace=[
                PiiColumnPlan.model_construct(
                    column_name="name",
                    entity_type=EntityType.FULL_NAME,
                    pattern=None,
                    depends_on=[
                        ConditioningColumn.model_construct(column_name="name", entity_type=EntityType.FULL_NAME)
                    ],
                )
            ],
        )

        with pytest.raises(ParameterError, match="cannot depend on itself"):
            validate_plan(pii_df, plan, data_config=DataParameters())

    def test_rejects_dependency_cycles(self, pii_df: pd.DataFrame) -> None:
        plan = PiiReplacementPlan.model_construct(
            scope=PiiReplacementScope.DATAFRAME,
            columns_to_replace=[
                PiiColumnPlan.model_construct(
                    column_name="name",
                    entity_type=EntityType.FULL_NAME,
                    pattern=None,
                    depends_on=[ConditioningColumn.model_construct(column_name="email", entity_type=EntityType.EMAIL)],
                ),
                PiiColumnPlan.model_construct(
                    column_name="email",
                    entity_type=EntityType.EMAIL,
                    pattern=None,
                    depends_on=[
                        ConditioningColumn.model_construct(column_name="name", entity_type=EntityType.FULL_NAME)
                    ],
                ),
            ],
        )

        with pytest.raises(ParameterError, match="dependencies contain a cycle"):
            validate_plan(pii_df, plan, data_config=DataParameters())

    def test_rejects_pattern_below_coverage_threshold(self, pii_df: pd.DataFrame) -> None:
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
            validate_plan(pii_df, plan, data_config=DataParameters())
