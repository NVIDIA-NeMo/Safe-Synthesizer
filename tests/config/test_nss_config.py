# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Literal

import pytest
from pydantic import Field, ValidationError

from nemo_safe_synthesizer.config import (
    DataParameters,
    DifferentialPrivacyHyperparams,
    PiiReplacerConfig,
    SafeSynthesizerParameters,
    TimeSeriesColdStartTrainingParameters,
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
        ],
        autoparam_with_auto="auto",
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
        assert TestParams(validation_ratio=0.0).validation_ratio == 0.0
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
    def test_parameters_get_method(self, fixture_simple_safe_synthesizer_parameters):
        assert fixture_simple_safe_synthesizer_parameters.get("num_input_records_to_sample") == 100

    def test_parameters_nesting(self, fixture_simple_safe_synthesizer_parameters):
        assert fixture_simple_safe_synthesizer_parameters.get("num_input_records_to_sample") == 100

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
            _ = PiiReplacerConfig()  # ty: ignore[missing-argument] -- intentionally omits required field to test that ValidationError is raised

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

    def test_parameter_values(self, fixture_simple_safe_synthesizer_parameters):
        params = fixture_simple_safe_synthesizer_parameters
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

    def test_dp_and_time_series_are_mutually_exclusive(self):
        """DP + time-series raises because DP would force ``max_sequences_per_example=1``,
        collapsing the per-example temporal structure time-series mode is built around.
        """
        with pytest.raises(ValidationError, match="not supported in time-series mode"):
            SafeSynthesizerParameters(
                privacy=DifferentialPrivacyHyperparams(dp_enabled=True),
                time_series=TimeSeriesParameters(is_timeseries=True, timestamp_column="event_time"),
            )

    @pytest.mark.parametrize(
        "max_seq_input, expected",
        [
            pytest.param("auto", None, id="auto_resolves_to_none_in_timeseries"),
            pytest.param(None, None, id="explicit_none_preserved"),
            pytest.param(5, 5, id="explicit_value_preserved"),
        ],
    )
    def test_max_sequences_per_example_default_in_timeseries(self, max_seq_input, expected):
        """In time-series mode, ``max_sequences_per_example='auto'`` resolves to ``None``.

        The default of ``10`` would chop sequences into short fragments and
        lose the long-range temporal structure the model needs to learn,
        so we let each example fill the context window instead.
        """
        params = SafeSynthesizerParameters(
            data=DataParameters(max_sequences_per_example=max_seq_input),
            time_series=TimeSeriesParameters(is_timeseries=True, timestamp_column="event_time"),
        )
        assert params.data.max_sequences_per_example == expected

    @pytest.mark.parametrize(
        "max_seq_input, expected",
        [
            pytest.param("auto", 10, id="auto_resolves_to_10"),
            pytest.param(None, None, id="explicit_none_preserved"),
            pytest.param(5, 5, id="explicit_value_preserved"),
        ],
    )
    def test_max_sequences_per_example_default_non_timeseries(self, max_seq_input, expected):
        """Outside time-series mode, ``max_sequences_per_example='auto'`` resolves to ``10``."""
        params = SafeSynthesizerParameters(
            data=DataParameters(max_sequences_per_example=max_seq_input),
        )
        assert params.data.max_sequences_per_example == expected

    def test_timestamp_interval_must_be_positive(self):
        """Negative ``timestamp_interval_seconds`` is rejected at construction time."""
        with pytest.raises(ValueError, match="positive"):
            TimeSeriesParameters(is_timeseries=True, timestamp_interval_seconds=-1)

    def test_cold_start_training_defaults_to_noop_weight(self):
        """The cold-start training multiplier is no-op by default."""
        params = TimeSeriesColdStartTrainingParameters()
        assert params.enabled is False
        assert params.strategies == []
        assert params.start_example_weight == 1.0
        assert params.start_example_records is None

    def test_cold_start_training_weight_must_be_at_least_one(self):
        """``1.0`` is the no-op baseline, so lower weights are rejected."""
        with pytest.raises(ValidationError):
            TimeSeriesColdStartTrainingParameters(start_example_weight=0.5)

    def test_cold_start_training_rejects_unknown_strategy(self):
        """Only implemented cold-start training strategies are accepted."""
        with pytest.raises(ValidationError):
            TimeSeriesColdStartTrainingParameters(strategies=["empty"])  # ty: ignore[list-item]

    def test_timeseries_without_group_column_warns(self):
        """``is_timeseries=True`` without a group column warns about the auto-injected sequence id."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SafeSynthesizerParameters(
                time_series=TimeSeriesParameters(is_timeseries=True, timestamp_interval_seconds=5),
            )
        assert any("group_training_examples_by" in str(warning.message) for warning in w)

    def test_data_validation_error_without_phantom_dp_error(self):
        """A bad data section should not produce a phantom 'DP is enabled' error when DP is disabled."""
        with pytest.raises(ValidationError) as exc_info:
            SafeSynthesizerParameters.from_yaml_str(
                "data:\n  order_training_examples_by: event_id\n  group_training_examples_by: null\n"
            )
        error_messages = [e["msg"] for e in exc_info.value.errors()]
        assert any("order_training_examples_by" in msg for msg in error_messages)
        assert not any("DP is enabled" in msg for msg in error_messages)

    def test_read_from_yaml(self, fixture_yaml_config_str):
        p = SafeSynthesizerParameters.from_yaml_str(fixture_yaml_config_str)
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
            DataParameters(group_training_examples_by=["patient_id", "event_id"])  # ty: ignore[invalid-argument-type]

    def test_comma_separated_string_accepted_by_pydantic(self):
        params = DataParameters(group_training_examples_by="patient_id,event_id")
        assert params.group_training_examples_by == "patient_id,event_id"
