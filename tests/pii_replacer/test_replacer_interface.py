# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import pandas as pd
import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.replace_pii import EntityType, PiiColumnPlan, PiiReplacementPlan, ReplacePiiConfig
from nemo_safe_synthesizer.errors import InternalError
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
            generator_time_seconds=0.1,
        )

        result = TransformResult(
            transformed_df=dataframe,
            column_statistics={},
            resolved_config=ReplacePiiConfig(replacement_plan=plan),
            generation_statistics=generation_statistics,
            replacement_time_seconds=0.25,
        )

        assert result.transformed_df is dataframe
        assert result.replacement_plan == plan
        assert result.generation_statistics is generation_statistics
        assert result.replacement_time_seconds == 0.25
        assert result.replacement_map is None

    @pytest.mark.parametrize(
        "field_overrides",
        [
            {"replacement_time_seconds": -0.1},
            {
                "generation_statistics": {
                    "generated_replacement_count": 1,
                    "generator_time_seconds": -0.1,
                }
            },
            {
                "generation_statistics": {
                    "generated_replacement_count": -1,
                    "generator_time_seconds": 0.1,
                }
            },
        ],
    )
    def test_elapsed_times_and_generation_count_cannot_be_negative(self, field_overrides: dict[str, object]) -> None:
        values: dict[str, object] = {
            "transformed_df": pd.DataFrame(),
            "column_statistics": {},
            "resolved_config": ReplacePiiConfig(replacement_plan=PiiReplacementPlan()),
            "generation_statistics": {
                "generated_replacement_count": 1,
                "generator_time_seconds": 0.1,
            },
            "replacement_time_seconds": 0.2,
        }
        values.update(field_overrides)

        with pytest.raises(ValidationError) as exc_info:
            TransformResult.model_validate(values)

        assert exc_info.value.errors()[0]["type"] == "greater_than_equal"

    def test_result_keeps_its_own_copy_of_the_executed_config(self) -> None:
        email = PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)
        config = ReplacePiiConfig(replacement_plan=PiiReplacementPlan(columns_to_replace=[email]))

        result = _result(config)
        assert config.inline_plan is not None
        config.inline_plan.columns_to_replace.clear()

        assert result.replacement_plan.columns_to_replace == [email]
        assert result.resolved_config.inline_plan == result.replacement_plan

    def test_result_requires_a_concrete_executed_plan(self) -> None:
        with pytest.raises(InternalError, match="must contain the executed replacement plan"):
            _result(ReplacePiiConfig())

    def test_result_rejects_a_separate_replacement_plan(self) -> None:
        config = ReplacePiiConfig(replacement_plan=PiiReplacementPlan())

        with pytest.raises(ValidationError, match="replacement_plan"):
            TransformResult.model_validate(
                {
                    "transformed_df": pd.DataFrame(),
                    "column_statistics": {},
                    "replacement_plan": PiiReplacementPlan(),
                    "resolved_config": config,
                    "generation_statistics": {"generated_replacement_count": 0, "generator_time_seconds": 0.0},
                    "replacement_time_seconds": 0.0,
                }
            )


def _result(config: ReplacePiiConfig) -> TransformResult:
    return TransformResult(
        transformed_df=pd.DataFrame(),
        column_statistics={},
        resolved_config=config,
        generation_statistics=ReplacementGenerationStatistics(
            generated_replacement_count=0, generator_time_seconds=0.0
        ),
        replacement_time_seconds=0.0,
    )


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

    @pytest.mark.parametrize(
        ("overrides", "error"),
        [
            ({"start": 10, "end": 2}, InternalError),
            ({"score": 5.0}, ValidationError),
            ({"score": float("nan")}, ValidationError),
        ],
    )
    def test_free_text_record_rejects_impossible_spans(
        self,
        overrides: dict[str, object],
        error: type[Exception],
    ) -> None:
        values: dict[str, object] = {
            "row_position": 0,
            "column_name": "notes",
            "start": 0,
            "end": 3,
            "entity_type": EntityType.FIRST_NAME,
            "detection_source": "gliner",
            "score": 0.9,
            "scope": "record",
            "original_value": "Ada",
            "replacement_value": "Eve",
        } | overrides

        with pytest.raises(error):
            FreeTextReplacementRecord.model_validate(values)
