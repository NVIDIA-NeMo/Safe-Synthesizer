# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import (
    EntityType,
    PiiColumnPlan,
    PiiReplacementPlan,
    PiiSamplerBackend,
    PiiSamplerConfig,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.pii_replacer import ReplacementGenerationStatistics, TabularPiiReplacer
from nemo_safe_synthesizer.pii_replacer.transform_result import TransformResult


class _FakeGlinerModel:
    def batch_extract_entities(self, texts: list[str], labels: list[str], **kwargs: object) -> list[object]:
        return [{"entities": {"first_name": [{"start": 0, "end": 3, "confidence": 0.9}]}} for _ in texts]


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

    def test_replacement_map_capture_is_explicitly_deferred(self) -> None:
        replacer = TabularPiiReplacer(ReplacePiiConfig(), data_config=DataParameters())

        with pytest.raises(NotImplementedError, match="replacement map capture is not implemented"):
            replacer.replace(pd.DataFrame({"email": ["ada@example.com"]}), capture_replacement_map=True)

    def test_empty_plan_returns_a_copy_without_mutating_the_caller_dataframe(self) -> None:
        dataframe = pd.DataFrame({"email": ["ada@example.com"]}, index=[7])
        original = dataframe.copy(deep=True)
        replacer = TabularPiiReplacer(ReplacePiiConfig(), data_config=DataParameters())

        result = replacer.replace(dataframe)

        pd.testing.assert_frame_equal(dataframe, original)
        pd.testing.assert_frame_equal(result.transformed_df, original)
        assert result.transformed_df is not dataframe
        assert result.generation_statistics.generated_replacement_count == 0

    def test_replace_executes_an_explicit_structured_plan(self) -> None:
        dataframe = pd.DataFrame({"identifier": ["USER-001", "USER-002"]}, index=[3, 3])
        config = ReplacePiiConfig(
            replacement_plan=PiiReplacementPlan(
                columns_to_replace=[
                    PiiColumnPlan(
                        column_name="identifier",
                        entity_type=EntityType.UNIQUE_IDENTIFIER,
                        pattern="USER-###",
                    )
                ]
            ),
            sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
        )

        first = TabularPiiReplacer(config, data_config=DataParameters()).replace(dataframe)
        second = TabularPiiReplacer(config, data_config=DataParameters()).replace(dataframe)

        assert first.transformed_df.equals(second.transformed_df)
        assert first.transformed_df.index.tolist() == [3, 3]
        assert first.transformed_df["identifier"].str.fullmatch(r"USER-\d{3}").all()
        assert first.generation_statistics.generated_replacement_count == 2
        assert first.resolved_config.inline_plan == config.inline_plan
        assert first.resolved_config.inline_plan is not None
        assert first.resolved_config.inline_plan.data_to_sampler_value_mapping == {}

    def test_replace_executes_an_explicit_free_text_plan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dataframe = pd.DataFrame({"notes": ["Ada called"]})
        original = dataframe.copy(deep=True)
        config = ReplacePiiConfig(
            replacement_plan=PiiReplacementPlan(
                columns_to_replace=[PiiColumnPlan(column_name="notes", entity_type=EntityType.FREE_TEXT)]
            ),
            sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
        )

        class FakeGliner2:
            @classmethod
            def from_pretrained(cls, model_id: str, **kwargs: object) -> _FakeGlinerModel:
                return _FakeGlinerModel()

        monkeypatch.setitem(sys.modules, "gliner2", SimpleNamespace(GLiNER2=FakeGliner2))

        result = TabularPiiReplacer(config, data_config=DataParameters()).replace(dataframe)

        pd.testing.assert_frame_equal(dataframe, original)
        assert result.transformed_df["notes"].iloc[0] != "Ada called"
        assert result.transformed_df["notes"].iloc[0].endswith(" called")


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
            replacement_plan=plan,
            resolved_config=ReplacePiiConfig(replacement_plan=plan),
            generation_statistics=generation_statistics,
            replacement_time_seconds=0.25,
        )

        assert result.transformed_df is dataframe
        assert result.replacement_plan is plan
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
            "replacement_plan": PiiReplacementPlan(),
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

    def test_resolved_config_must_contain_the_executed_plan(self) -> None:
        executed = PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)]
        )

        with pytest.raises(InternalError, match="must contain the executed replacement_plan"):
            TransformResult(
                transformed_df=pd.DataFrame(),
                column_statistics={},
                replacement_plan=executed,
                resolved_config=ReplacePiiConfig(),
                generation_statistics=ReplacementGenerationStatistics(
                    generated_replacement_count=0,
                    generator_time_seconds=0.0,
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
