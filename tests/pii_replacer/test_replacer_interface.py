# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import pandas as pd
import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.replace_pii import EntityType, PiiReplacementPlan, ReplacePiiConfig
from nemo_safe_synthesizer.pii_replacer import (
    FreeTextReplacementRecord,
    ReplacementGenerationStatistics,
    ReplacementMap,
    StructuredReplacementRecord,
    TabularPiiReplacer,
)
from nemo_safe_synthesizer.pii_replacer.transform_result import TransformResult


@pytest.mark.unit
class TestTabularPiiReplacerInterface:
    def test_constructor_has_only_the_public_configuration_inputs(self) -> None:
        signature = inspect.signature(TabularPiiReplacer)

        assert list(signature.parameters) == ["config", "data_config", "time_series"]
        assert signature.parameters["data_config"].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["time_series"].default is None

    def test_replacement_map_capture_is_keyword_only_and_off_by_default(self) -> None:
        parameter = inspect.signature(TabularPiiReplacer.replace).parameters["capture_replacement_map"]

        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is False


@pytest.mark.unit
class TestTransformResult:
    def test_result_includes_plan_and_replacement_timing_statistics(self) -> None:
        dataframe = pd.DataFrame({"email": ["synthetic@example.com"]})
        plan = PiiReplacementPlan()
        generation_statistics = ReplacementGenerationStatistics(
            generated_replacement_count=2,
            elapsed_time_seconds=0.1,
        )

        result = TransformResult(
            transformed_df=dataframe,
            column_statistics={},
            replacement_plan=plan,
            resolved_config=ReplacePiiConfig(replacement_plan=plan),
            generation_statistics=generation_statistics,
            elapsed_time_seconds=0.25,
        )

        assert result.transformed_df is dataframe
        assert result.replacement_plan is plan
        assert result.generation_statistics is generation_statistics
        assert result.elapsed_time_seconds == 0.25
        assert result.replacement_map is None

    @pytest.mark.parametrize(
        "field_overrides",
        [
            {"elapsed_time_seconds": -0.1},
            {
                "generation_statistics": {
                    "generated_replacement_count": 1,
                    "elapsed_time_seconds": -0.1,
                }
            },
            {
                "generation_statistics": {
                    "generated_replacement_count": -1,
                    "elapsed_time_seconds": 0.1,
                }
            },
        ],
    )
    def test_elapsed_times_and_generation_count_cannot_be_negative(self, field_overrides: dict[str, object]) -> None:
        values: dict[str, object] = {
            "transformed_df": pd.DataFrame(),
            "column_statistics": {},
            "replacement_plan": PiiReplacementPlan(),
            "resolved_config": ReplacePiiConfig(replacement_plan=PiiReplacementPlan()),
            "generation_statistics": {
                "generated_replacement_count": 1,
                "elapsed_time_seconds": 0.1,
            },
            "elapsed_time_seconds": 0.2,
        }
        values.update(field_overrides)

        with pytest.raises(ValidationError) as exc_info:
            TransformResult.model_validate(values)

        assert exc_info.value.errors()[0]["type"] == "greater_than_equal"


@pytest.mark.unit
class TestReplacementMap:
    def test_records_hide_sensitive_values_from_repr(self) -> None:
        structured = StructuredReplacementRecord(
            row_position=0,
            column_name="email",
            entity_type=EntityType.EMAIL,
            scope="record",
            original_value="ada@example.com",
            replacement_value="grace@example.org",
        )
        free_text = FreeTextReplacementRecord(
            row_position=0,
            column_name="notes",
            start=0,
            end=3,
            entity_type=EntityType.FIRST_NAME,
            detection_source="gliner",
            score=0.9,
            scope="record",
            original_value="Ada",
            replacement_value="Eve",
        )
        replacement_map = ReplacementMap(structured=(structured,), free_text=(free_text,))

        assert "ada@example.com" not in repr(replacement_map)
        assert "grace@example.org" not in repr(replacement_map)
        assert "Ada" not in repr(replacement_map)
        assert replacement_map.free_text[0].original_value == "Ada"
