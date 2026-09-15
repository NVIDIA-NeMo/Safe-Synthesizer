# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

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
from nemo_safe_synthesizer.errors import GenerationError
from nemo_safe_synthesizer.pii_replacer import ReplacementGenerationStatistics, TabularPiiReplacer
from nemo_safe_synthesizer.pii_replacer.transform_result import TransformResult


@pytest.mark.unit
class TestTabularPiiReplacerInterface:
    def test_constructor_has_only_the_public_configuration_inputs(self) -> None:
        signature = inspect.signature(TabularPiiReplacer)

        assert list(signature.parameters) == ["config", "data_config", "time_series"]
        assert signature.parameters["data_config"].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["time_series"].default is None

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
        assert first.resolved_config.inline_plan.dependency_value_mappings == {}

    def test_free_text_replacement_is_explicitly_deferred(self) -> None:
        dataframe = pd.DataFrame({"notes": ["Ada called"]})
        original = dataframe.copy(deep=True)
        config = ReplacePiiConfig(
            replacement_plan=PiiReplacementPlan(
                columns_to_replace=[PiiColumnPlan(column_name="notes", entity_type=EntityType.FREE_TEXT)]
            ),
            sampler=PiiSamplerConfig(backend=PiiSamplerBackend.FAKER),
        )

        with pytest.raises(GenerationError, match="detector and span-resolution"):
            TabularPiiReplacer(config, data_config=DataParameters()).replace(dataframe)

        pd.testing.assert_frame_equal(dataframe, original)


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

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            TransformResult.model_validate(values)
