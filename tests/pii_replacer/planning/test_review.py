# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.replace_pii import (
    ConditioningColumn,
    EntityType,
    PiiColumnPlan,
    PiiReplacementPlan,
    PiiReplacementSettings,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.pii_replacer.planning.review import DependencyValueReview, plan_review_notes


@pytest.mark.unit
class TestPlanReviewNotes:
    def test_notes_list_unchanged_columns_and_values_used_as_is(self) -> None:
        dataframe = pd.DataFrame(
            {
                "first": ["Ada", "Grace", "Alan", "Mae"],
                "sex": ["Female", "Woman", "male", None],
                "race": ["White", "Black", "White", "Asian"],
                "admission_id": ["A1", "A2", "A3", "A4"],
            }
        )
        plan = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(
                    column_name="first",
                    entity_type=EntityType.FIRST_NAME,
                    depends_on=[
                        ConditioningColumn(column_name="sex", entity_type=EntityType.GENDER),
                        ConditioningColumn(column_name="race", entity_type=EntityType.ETHNIC_BACKGROUND),
                    ],
                )
            ],
            data_to_sampler_value_mapping={"sex": {"Woman": ["female"]}},
        )

        notes = plan_review_notes(
            dataframe,
            ReplacePiiConfig(replacement=PiiReplacementSettings(plan=plan)),
            sampler_value_catalog={EntityType.GENDER: ("female", "male")},
        )

        assert notes is not None
        assert notes.unchanged_columns == ("sex", "race", "admission_id")
        assert notes.dependency_values == (
            DependencyValueReview("sex", EntityType.GENDER, ("Female", "male"), ("female", "male")),
        )

    def test_auto_discovery_has_no_notes(self) -> None:
        assert plan_review_notes(pd.DataFrame({"a": [1]}), ReplacePiiConfig()) is None
