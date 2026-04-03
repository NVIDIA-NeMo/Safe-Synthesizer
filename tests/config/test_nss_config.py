# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated, Literal

import pytest
from pydantic import Field, ValidationError

from nemo_safe_synthesizer.config import (
    DataParameters,
    DifferentialPrivacyHyperparams,
    PiiReplacerConfig,
    SafeSynthesizerParameters,
    TimeSeriesParameters,
)
from nemo_safe_synthesizer.configurator.parameters import Parameters
from nemo_safe_synthesizer.configurator.validators import ValueValidator


class SubGroup(Parameters):
    basic_int_param: Annotated[int, Field(default=10)]

    basic_int_autoparam: Annotated[str | int, Field(default="auto")]

    basic_auto_with_valid_none: Annotated[
        int | Literal["auto"] | None,
        Field(default=None, description="valid none param"),
    ]

    basic_str_param: Annotated[
        str | None,
        Field(default=None, title="basic string"),
    ]

    basic_union_basic_input: Annotated[
        str | float | list[int] | None,
        Field(default=None, title="basic union input"),
    ]


class ParentGroup(Parameters):
    list_subgroup_param: Annotated[list[SubGroup], Field(title="list of subgroups")]

    autoparam_with_auto: Annotated[float | Literal["auto"], Field(default="auto", title="autoparam with auto")]


@pytest.fixture
def parent_fixture() -> ParentGroup:
    return ParentGroup(
        list_subgroup_param=[
            SubGroup(
                basic_int_param=10,
                basic_int_autoparam="auto",
                basic_auto_with_valid_none=None,
                basic_str_param=None,
                basic_union_basic_input=None,
            )
        ]
    )


@pytest.fixture
def subgroup_fixture() -> SubGroup:
    return SubGroup(
        basic_int_param=10,
        basic_int_autoparam="auto",
        basic_auto_with_valid_none=None,
        basic_str_param=None,
        basic_union_basic_input=None,
    )


class TestValueValidation:
    def test_value_validator_success(self):
        class TestParams(Parameters):
            validation_ratio: Annotated[
                str | float,
                ValueValidator(value_func=lambda v: 0 <= v <= 1),
                Field(default=0.0),
            ]

        # These should succeed
        assert TestParams().validation_ratio == 0.0
        assert TestParams(validation_ratio=0.5).validation_ratio == 0.5

    def test_value_validator_failure(self):
        class TestParams(Parameters):
            validation_ratio: Annotated[
                str | float,
                ValueValidator(value_func=lambda v: 0 <= v <= 1),
                Field(default=0.0),
            ]

        # This should fail validation
        with pytest.raises(ValidationError):
            TestParams(validation_ratio=1.2)


class TestParametersClass:
    def test_parameters_get_method(self, simple_safe_synthesizer_parameters):
        assert simple_safe_synthesizer_parameters.get("num_input_records_to_sample") == 100

    def test_parameters_nesting(self, simple_safe_synthesizer_parameters):
        assert simple_safe_synthesizer_parameters.get("num_input_records_to_sample") == 100

    def test_nested_auto_param_round_trip(self, subgroup_fixture, parent_fixture):
        subgroup_py = subgroup_fixture.model_dump()
        parent_py = parent_fixture.model_dump()
        subgroup_json = subgroup_fixture.model_dump_json()
        parent_json = parent_fixture.model_dump_json()
        assert ParentGroup.model_validate_json(parent_json) == parent_fixture
        assert SubGroup.model_validate_json(subgroup_json) == subgroup_fixture
        assert ParentGroup.model_validate(parent_py) == parent_fixture
        assert SubGroup.model_validate(subgroup_py) == subgroup_fixture


class TestPiiParameters:
    def test_pii_parameters_create_without_steps(self):
        with pytest.raises(ValidationError):
            _ = PiiReplacerConfig()

    def test_create_default(self):
        params = PiiReplacerConfig.get_default_config()
        assert params.globals.ner.ner_threshold == 0.3


class TestSafeSynthesizerParameters:
    @pytest.mark.parametrize(
        "value, expected",
        [(1, 1), (None, 1), ("auto", 1)],
        ids=["1", "None", "auto"],
    )
    def test_max_sequences_dp_setting(self, value, expected):
        # When DP is enabled, max_sequences_per_example must be set to 1 or aut
        print(f"value: {value}, expected: {expected}")
        if value is None:
            data = DataParameters()
        else:
            data = DataParameters(max_sequences_per_example=value)

        dp = DifferentialPrivacyHyperparams(dp_enabled=True)
        params = SafeSynthesizerParameters(
            data=data,
            privacy=dp,
        )
        assert params.get("max_sequences_per_example") == expected

    def test_parameter_values(self, simple_safe_synthesizer_parameters):
        params = simple_safe_synthesizer_parameters
        assert params.get("num_input_records_to_sample") == 100
        assert params.get("batch_size") == 10
        print(params.training)
        assert params.get("group_training_examples_by") == "my_col"

    @pytest.mark.parametrize(
        "replace_pii_kwarg, expected_pii_config",
        [({}, True), ({"replace_pii": None}, None)],
        ids=["enabled", "disabled"],
    )
    def test_enabled_pii(self, replace_pii_kwarg, expected_pii_config):
        params = SafeSynthesizerParameters.from_params(**replace_pii_kwarg)
        val = True if params.replace_pii is not None else None
        assert val == expected_pii_config

    def test_timestamp_required_for_time_series(self):
        """Test that is_timeseries=True requires timestamp_column or timestamp_interval_seconds."""
        with pytest.raises(ValidationError):
            TimeSeriesParameters(is_timeseries=True)

    def test_timestamp_only_allowed_when_time_series_enabled(self):
        """Test that timestamp_column can only be set when is_timeseries is True."""
        with pytest.raises(ValidationError):
            TimeSeriesParameters(timestamp_column="event_time")

    def test_time_series_configuration_passes_validation(self):
        params = TimeSeriesParameters(is_timeseries=True, timestamp_column="event_time")
        assert params.timestamp_column == "event_time"

    def test_read_from_yaml(self, yaml_config_str):
        p = SafeSynthesizerParameters.from_yaml_str(yaml_config_str)
        assert p.get("gradient_accumulation_steps") == 8


class TestGroupTrainingExamplesBy:
    def test_single_column_string_accepted(self):
        params = DataParameters(group_training_examples_by="patient_id")
        assert params.group_training_examples_by == "patient_id"

    def test_none_accepted(self):
        params = DataParameters(group_training_examples_by=None)
        assert params.group_training_examples_by is None

    def test_list_rejected_by_pydantic(self):
        with pytest.raises(ValidationError):
            DataParameters(group_training_examples_by=["patient_id", "event_id"])

    def test_comma_separated_string_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            params = DataParameters(group_training_examples_by="patient_id,event_id")
            assert params.group_training_examples_by == "patient_id,event_id"
        assert any("comma" in r.message.lower() for r in caplog.records)
        assert any("multi-column grouping is not currently supported" in r.message for r in caplog.records)
