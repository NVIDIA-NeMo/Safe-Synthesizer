# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import pandas as pd
import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import PiiReplacementPlan, ReplacePiiConfig
from nemo_safe_synthesizer.pii_replacer import ReplacementGenerationStatistics, TabularPiiReplacer
from nemo_safe_synthesizer.pii_replacer.transform_result import TransformResult


@pytest.mark.unit
class TestTabularPiiReplacerInterface:
    def test_constructor_has_only_the_public_configuration_inputs(self) -> None:
        signature = inspect.signature(TabularPiiReplacer)

        assert list(signature.parameters) == ["config", "data_config", "time_series"]
        assert signature.parameters["data_config"].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["time_series"].default is None

    def test_interface_stub_does_not_mutate_the_caller_dataframe(self) -> None:
        dataframe = pd.DataFrame({"email": ["ada@example.com"]}, index=[7])
        original = dataframe.copy(deep=True)
        replacer = TabularPiiReplacer(ReplacePiiConfig(), data_config=DataParameters())

        with pytest.raises(NotImplementedError, match="execution is not implemented"):
            replacer.replace(dataframe)

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
            "generation_statistics": {
                "generated_replacement_count": 1,
                "elapsed_time_seconds": 0.1,
            },
            "elapsed_time_seconds": 0.2,
        }
        values.update(field_overrides)

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            TransformResult.model_validate(values)
