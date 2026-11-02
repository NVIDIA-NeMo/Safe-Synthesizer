# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging
import re
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
from nemo_safe_synthesizer.errors import InternalError
from nemo_safe_synthesizer.pii_replacer import (
    FreeTextReplacementRecord,
    ReplacementGenerationStatistics,
    ReplacementMap,
    StructuredReplacementRecord,
    TabularPiiReplacer,
)
from nemo_safe_synthesizer.pii_replacer.transform_result import TransformResult


class _FakeGlinerModel:
    def batch_extract_entities(
        self,
        texts: list[str],
        labels: dict[str, dict[str, float]],
        **kwargs: object,
    ) -> list[object]:
        return [{"entities": {"first_name": [{"start": 0, "end": 3, "confidence": 0.99}]}} for _ in texts]


class _NoSpanGlinerModel:
    def batch_extract_entities(
        self,
        texts: list[str],
        labels: dict[str, dict[str, float]],
        **kwargs: object,
    ) -> list[object]:
        return [{"entities": {}} for _ in texts]


def _use_fake_gliner2(monkeypatch: pytest.MonkeyPatch, model: object) -> None:
    """Load ``model`` through the real GLiNER2 loader, as if a GPU were available."""

    class FakeGliner2:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
            return model

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, "gliner2", SimpleNamespace(GLiNER2=FakeGliner2))


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

    def test_empty_plan_returns_a_copy_without_mutating_the_caller_dataframe(self) -> None:
        dataframe = pd.DataFrame({"email": ["ada@example.com"]}, index=[7])
        original = dataframe.copy(deep=True)
        config = ReplacePiiConfig(
            sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER), replacement_plan=PiiReplacementPlan()
        )
        replacer = TabularPiiReplacer(config, data_config=DataParameters())

        result = replacer.replace(dataframe)

        pd.testing.assert_frame_equal(dataframe, original)
        pd.testing.assert_frame_equal(result.transformed_df, original)
        assert result.transformed_df is not dataframe
        assert result.generation_statistics.generated_replacement_count == 0

    def test_locale_and_seed_come_from_replace_pii(self) -> None:
        dataframe = pd.DataFrame({"name": [f"Person {row}" for row in range(20)]})
        plan = PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="name", entity_type=EntityType.LAST_NAME)],
        )

        def replaced(**settings: object) -> list[str]:
            config = ReplacePiiConfig.model_validate(
                {"sampler": {"backend": "faker"}, "replacement_plan": plan, **settings}
            )
            return (
                TabularPiiReplacer(config, data_config=DataParameters())
                .replace(dataframe)
                .transformed_df["name"]
                .tolist()
            )

        assert replaced(seed=7) == replaced(seed=7)
        assert replaced(seed=7) != replaced(seed=8)
        assert replaced(seed=7, locale="ru_RU") != replaced(seed=7)
        assert any(not name.isascii() for name in replaced(seed=7, locale="ru_RU"))

    def test_generators_receive_the_text_form_that_patterns_were_validated_against(self) -> None:
        dataframe = pd.DataFrame({"dob": pd.to_datetime(["2000-01-01", "1990-05-17"]), "id": [1.5, 2.25]})
        config = ReplacePiiConfig(
            sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
            replacement_plan=PiiReplacementPlan(
                columns_to_replace=[
                    PiiColumnPlan(column_name="dob", entity_type=EntityType.DATE_OF_BIRTH, pattern="%Y-%m-%d"),
                    PiiColumnPlan(column_name="id", entity_type=EntityType.UNIQUE_IDENTIFIER),
                ],
            ),
        )

        result = TabularPiiReplacer(config, data_config=DataParameters()).replace(dataframe)

        assert result.transformed_df["dob"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
        assert result.transformed_df["id"].tolist()[0].count(".") == 1
        assert result.transformed_df["id"].str.fullmatch(r"\d\.\d|\d\.\d{2}").all()

    def test_supplied_plan_below_pattern_coverage_warns_and_replaces_every_value(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        phones = [f"555-{row:04d}" for row in range(90)] + [f"(555) {row:03d}-0000" for row in range(10)]
        config = ReplacePiiConfig(
            sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
            replacement_plan=PiiReplacementPlan(
                columns_to_replace=[
                    PiiColumnPlan(column_name="phone", entity_type=EntityType.PHONE_NUMBER, pattern="###-####")
                ],
            ),
        )

        with caplog.at_level(logging.WARNING):
            result = TabularPiiReplacer(config, data_config=DataParameters()).replace(pd.DataFrame({"phone": phones}))

        assert "pattern '###-####' covers only 90.0% of non-null values" in caplog.text
        replaced = result.transformed_df["phone"].tolist()
        assert all(re.fullmatch(r"[0-9]{3}-[0-9]{4}", value) for value in replaced)
        assert all(new != old for new, old in zip(replaced, phones, strict=True))

    def test_replace_executes_an_explicit_structured_plan(self) -> None:
        dataframe = pd.DataFrame({"identifier": ["USER-001", "USER-002"]}, index=[3, 3])
        config = ReplacePiiConfig(
            sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
            replacement_plan=PiiReplacementPlan(
                columns_to_replace=[
                    PiiColumnPlan(
                        column_name="identifier",
                        entity_type=EntityType.UNIQUE_IDENTIFIER,
                        pattern="USER-###",
                    )
                ],
            ),
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
        assert first.replacement_map is None

    def test_replace_can_capture_sensitive_replacement_provenance(self) -> None:
        dataframe = pd.DataFrame({"identifier": ["USER-001"]})
        config = ReplacePiiConfig(
            sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
            replacement_plan=PiiReplacementPlan(
                columns_to_replace=[
                    PiiColumnPlan(
                        column_name="identifier",
                        entity_type=EntityType.UNIQUE_IDENTIFIER,
                        pattern="USER-###",
                    )
                ],
            ),
        )

        result = TabularPiiReplacer(config, data_config=DataParameters()).replace(
            dataframe,
            capture_replacement_map=True,
        )

        assert result.replacement_map is not None
        assert result.replacement_map.structured[0].original_value == "USER-001"
        assert result.replacement_map.structured[0].replacement_value == result.transformed_df.at[0, "identifier"]

    def test_replace_executes_an_explicit_free_text_plan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dataframe = pd.DataFrame({"notes": ["Ada called"]})
        original = dataframe.copy(deep=True)
        config = ReplacePiiConfig(
            sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
            replacement_plan=PiiReplacementPlan(
                columns_to_replace=[PiiColumnPlan(column_name="notes", entity_type=EntityType.FREE_TEXT)],
            ),
        )

        _use_fake_gliner2(monkeypatch, _FakeGlinerModel())

        result = TabularPiiReplacer(config, data_config=DataParameters()).replace(dataframe)

        pd.testing.assert_frame_equal(dataframe, original)
        assert result.transformed_df["notes"].iloc[0] != "Ada called"
        assert result.transformed_df["notes"].iloc[0].endswith(" called")

    @pytest.mark.parametrize(("locale", "searched"), [("en_US", True), ("fr_FR", False)])
    def test_find_and_replace_runs_only_for_en_us(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, locale: str, searched: bool
    ) -> None:
        dataframe = pd.DataFrame({"first_name": ["Adeline"], "notes": ["Adeline called"]})
        config = ReplacePiiConfig(
            locale=locale,
            sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
            replacement_plan=PiiReplacementPlan(
                columns_to_replace=[
                    PiiColumnPlan(column_name="first_name", entity_type=EntityType.FIRST_NAME),
                    PiiColumnPlan(column_name="notes", entity_type=EntityType.FREE_TEXT),
                ],
            ),
        )
        _use_fake_gliner2(monkeypatch, _NoSpanGlinerModel())

        with caplog.at_level(logging.WARNING):
            result = TabularPiiReplacer(config, data_config=DataParameters()).replace(dataframe)

        assert ("Adeline" not in result.transformed_df["notes"].iloc[0]) is searched
        assert ("Skipping free_text_detection.find_and_replace" in caplog.text) is not searched

    def test_detect_reports_replace_statistics_without_changing_the_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The fake detector finds the leading "Ada"; "Adeline" is found only by matching row 0's first name.
        dataframe = pd.DataFrame({"first_name": ["Adeline", "Ada"], "notes": ["Ada met Adeline", "Ada left"]})
        original = dataframe.copy(deep=True)
        config = ReplacePiiConfig(
            sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
            replacement_plan=PiiReplacementPlan(
                columns_to_replace=[
                    PiiColumnPlan(column_name="first_name", entity_type=EntityType.FIRST_NAME),
                    PiiColumnPlan(column_name="notes", entity_type=EntityType.FREE_TEXT),
                ],
            ),
        )
        _use_fake_gliner2(monkeypatch, _FakeGlinerModel())
        replacer = TabularPiiReplacer(config, data_config=DataParameters())

        detected = replacer.detect(dataframe)
        replaced = replacer.replace(dataframe)

        pd.testing.assert_frame_equal(dataframe, original)
        assert detected.resolved_config.inline_plan == replaced.resolved_config.inline_plan
        for column_name, statistics in detected.column_statistics.items():
            assert statistics.detected_entity_counts == replaced.column_statistics[column_name].detected_entity_counts
            assert statistics.detected_entity_values == replaced.column_statistics[column_name].detected_entity_values
            assert not statistics.is_transformed
        assert detected.column_statistics["notes"].detected_entity_values == {"first_name": {"Ada", "Adeline"}}


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
