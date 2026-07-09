# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared time-series validation helpers."""

import pandas as pd
import pytest

from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.data_processing.timeseries_validation import (
    TimeSeriesDataValidationError,
    TimeSeriesGroupTimestampStats,
    TimeSeriesParameterValidationError,
    TimeSeriesValidationReason,
    validate_start_stop_consistency,
    validate_timeseries_data,
)
from nemo_safe_synthesizer.defaults import PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.errors import DataError


def test_validate_start_stop_consistency_valid():
    """Validation passes when all groups have the same start/stop."""
    stats = [
        TimeSeriesGroupTimestampStats("A", "2024-01-01", "2024-01-03", 3600),
        TimeSeriesGroupTimestampStats("B", "2024-01-01", "2024-01-03", 3600),
    ]

    start, stop = validate_start_stop_consistency(stats)

    assert start == "2024-01-01"
    assert stop == "2024-01-03"


def test_validate_start_stop_consistency_different_starts_raises():
    """A start timestamp mismatch raises a data error."""
    stats = [
        TimeSeriesGroupTimestampStats("A", "2024-01-01", "2024-01-03", 3600),
        TimeSeriesGroupTimestampStats("B", "2024-01-02", "2024-01-03", 3600),
    ]

    with pytest.raises(DataError, match="Start timestamps differ across groups"):
        validate_start_stop_consistency(stats)


def test_validate_start_stop_consistency_different_stops_raises():
    """A stop timestamp mismatch raises a data error."""
    stats = [
        TimeSeriesGroupTimestampStats("A", "2024-01-01", "2024-01-03", 3600),
        TimeSeriesGroupTimestampStats("B", "2024-01-01", "2024-01-04", 3600),
    ]

    with pytest.raises(DataError, match="Stop timestamps differ across groups"):
        validate_start_stop_consistency(stats)


@pytest.mark.parametrize(
    "values,config_overrides,expected_error,expected_reason",
    [
        pytest.param(
            ["not_a_date", "also_not"],
            {},
            TimeSeriesParameterValidationError,
            TimeSeriesValidationReason.TIMESTAMP_FORMAT_MISMATCH,
            id="non_datetime_strings",
        ),
        pytest.param(
            ["2024-01-01", "2024-01-02"],
            {"timestamp_format": "%m/%d/%Y"},
            TimeSeriesParameterValidationError,
            TimeSeriesValidationReason.TIMESTAMP_FORMAT_MISMATCH,
            id="explicit_format_mismatch",
        ),
        pytest.param(
            ["01/01/2024", "01/2024"],
            {},
            TimeSeriesDataValidationError,
            TimeSeriesValidationReason.TIMESTAMP_PARSE_FAILED,
            id="mixed_formats",
        ),
    ],
)
def test_validate_timeseries_data_reports_timestamp_format_errors(
    values, config_overrides, expected_error, expected_reason
):
    """Timestamp format failures are exposed through the public validator."""
    df = pd.DataFrame({"ts": values, "value": [1, 2]})
    config = SafeSynthesizerParameters.from_params(
        is_timeseries=True,
        timestamp_column="ts",
        rope_scaling_factor=1,
        **config_overrides,
    )

    with pytest.raises(expected_error) as exc_info:
        validate_timeseries_data(df, config)

    assert exc_info.value.reason is expected_reason


def test_validate_timeseries_data_rejects_empty_generated_timestamps():
    """Empty time-series data must raise a typed validation error, not crash."""
    df = pd.DataFrame({"value": []})
    config = SafeSynthesizerParameters.from_params(
        is_timeseries=True,
        timestamp_interval_seconds=60,
        rope_scaling_factor=1,
    )

    with pytest.raises(TimeSeriesDataValidationError) as exc_info:
        validate_timeseries_data(df, config)

    assert exc_info.value.reason is TimeSeriesValidationReason.TIMESERIES_EMPTY


def test_validate_timeseries_data_rejects_empty_explicit_timestamps():
    """Empty time-series data uses the same issue code with explicit timestamps."""
    df = pd.DataFrame({"ts": [], "value": []})
    config = SafeSynthesizerParameters.from_params(
        is_timeseries=True,
        timestamp_column="ts",
        rope_scaling_factor=1,
    )

    with pytest.raises(TimeSeriesDataValidationError) as exc_info:
        validate_timeseries_data(df, config)

    assert exc_info.value.reason is TimeSeriesValidationReason.TIMESERIES_EMPTY


def test_validate_timeseries_data_does_not_mutate_inputs():
    """Preflight calls the validator, so it must leave caller-owned objects unchanged."""
    df = pd.DataFrame(
        {
            "grp": ["A", "A", "B", "B"],
            "ts": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
            "value": [1, 2, 3, 4],
        }
    )
    original_df = df.copy(deep=True)
    config = SafeSynthesizerParameters.from_params(
        is_timeseries=True,
        timestamp_column="ts",
        group_training_examples_by="grp",
        rope_scaling_factor=1,
    )
    assert config.time_series.timestamp_format is None

    result = validate_timeseries_data(df, config)

    pd.testing.assert_frame_equal(df, original_df)
    assert config.time_series.timestamp_format is None
    assert result.timestamp_format == "%Y-%m-%d"


def test_validate_timeseries_data_generated_timestamp_uses_copy():
    """Generated timestamps and pseudo groups are returned without mutating inputs."""
    df = pd.DataFrame({"value": [1, 2, 3]})
    config = SafeSynthesizerParameters.from_params(
        is_timeseries=True,
        timestamp_interval_seconds=60,
        rope_scaling_factor=1,
    )

    result = validate_timeseries_data(df, config)

    assert PSEUDO_GROUP_COLUMN not in df.columns
    assert config.data.group_training_examples_by is None
    assert result.group_by_column == PSEUDO_GROUP_COLUMN
    assert result.timestamp_column == "elapsed_seconds"
    assert list(result.data["elapsed_seconds"]) == [0, 60, 120]


def test_validate_timeseries_data_generated_timestamp_uses_unique_column_name():
    """Generated timestamps do not overwrite existing elapsed-seconds columns."""
    df = pd.DataFrame(
        {
            "elapsed_seconds": [100, 100, 100],
            "_elapsed_seconds": [200, 200, 200],
            "value": [1, 2, 3],
        }
    )
    config = SafeSynthesizerParameters.from_params(
        is_timeseries=True,
        timestamp_interval_seconds=60,
        rope_scaling_factor=1,
    )

    result = validate_timeseries_data(df, config)

    assert result.timestamp_column not in {"elapsed_seconds", "_elapsed_seconds"}
    assert result.timestamp_column in result.data.columns
    assert list(result.data[result.timestamp_column]) == [0, 60, 120]
    assert list(result.data["elapsed_seconds"]) == [100, 100, 100]
    assert list(result.data["_elapsed_seconds"]) == [200, 200, 200]


@pytest.mark.parametrize(
    "values,expected_reason",
    [
        pytest.param([True, False, True], TimeSeriesValidationReason.TIMESTAMP_ELAPSED_INVALID, id="boolean"),
        pytest.param([0.0, float("nan"), 60.0], TimeSeriesValidationReason.TIMESTAMP_NULLS, id="nan"),
        pytest.param([0.0, float("inf"), 60.0], TimeSeriesValidationReason.TIMESTAMP_ELAPSED_INVALID, id="pos_inf"),
        pytest.param([0.0, float("-inf"), 60.0], TimeSeriesValidationReason.TIMESTAMP_ELAPSED_INVALID, id="neg_inf"),
    ],
)
def test_validate_timeseries_data_rejects_invalid_elapsed_second_values(values, expected_reason):
    """Elapsed-seconds timestamps reject boolean, null, and infinite values before interval work."""
    df = pd.DataFrame({"group": ["A", "A", "A"], "ts": values, "value": [1, 2, 3]})
    config = SafeSynthesizerParameters.from_params(
        is_timeseries=True,
        timestamp_column="ts",
        timestamp_format="elapsed_seconds",
        group_training_examples_by="group",
        rope_scaling_factor=1,
    )

    with pytest.raises(TimeSeriesDataValidationError) as exc_info:
        validate_timeseries_data(df, config)

    assert exc_info.value.reason is expected_reason


def test_validate_timeseries_data_rejects_interval_mismatch():
    """Interval mismatches are owned by the shared validator."""
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B"],
            "ts": [0, 60, 120, 0, 30, 120],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )
    config = SafeSynthesizerParameters.from_params(
        is_timeseries=True,
        timestamp_column="ts",
        timestamp_format="elapsed_seconds",
        group_training_examples_by="group",
        rope_scaling_factor=1,
    )

    with pytest.raises(TimeSeriesDataValidationError) as exc_info:
        validate_timeseries_data(df, config)

    assert exc_info.value.reason is TimeSeriesValidationReason.TIMESTAMP_INTERVAL_MISMATCH


def test_validate_timeseries_data_checks_every_delta_against_configured_interval():
    """Every group delta must be within tolerance of the configured interval."""
    df = pd.DataFrame({"group": ["A", "A", "A"], "ts": [0.0, 10.09, 20.27], "value": [1, 2, 3]})
    config = SafeSynthesizerParameters.from_params(
        is_timeseries=True,
        timestamp_column="ts",
        timestamp_format="elapsed_seconds",
        timestamp_interval_seconds=10,
        group_training_examples_by="group",
        rope_scaling_factor=1,
    )

    with pytest.raises(TimeSeriesDataValidationError) as exc_info:
        validate_timeseries_data(df, config)

    assert exc_info.value.reason is TimeSeriesValidationReason.TIMESTAMP_INTERVAL_MISMATCH


@pytest.mark.parametrize(
    "values,timestamp_format",
    [
        pytest.param([0.0, 0.5, 1.0], "elapsed_seconds", id="elapsed_seconds"),
        pytest.param(
            [
                "2024-01-01 00:00:00.000000",
                "2024-01-01 00:00:00.500000",
                "2024-01-01 00:00:01.000000",
            ],
            "%Y-%m-%d %H:%M:%S.%f",
            id="datetime",
        ),
    ],
)
def test_validate_timeseries_data_rejects_fractional_intervals(values, timestamp_format):
    """Inferred intervals must fit the integer-second generation contract."""
    df = pd.DataFrame({"group": ["A", "A", "A"], "ts": values, "value": [1, 2, 3]})
    config = SafeSynthesizerParameters.from_params(
        is_timeseries=True,
        timestamp_column="ts",
        timestamp_format=timestamp_format,
        group_training_examples_by="group",
        rope_scaling_factor=1,
    )

    with pytest.raises(TimeSeriesDataValidationError) as exc_info:
        validate_timeseries_data(df, config)

    assert exc_info.value.reason is TimeSeriesValidationReason.TIMESTAMP_INTERVAL_MISMATCH


def test_validate_timeseries_data_rejects_group_length_mismatch():
    """Unequal group lengths are owned by the shared validator."""
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B"],
            "ts": [0, 60, 120, 0, 60],
            "value": [1, 2, 3, 4, 5],
        }
    )
    config = SafeSynthesizerParameters.from_params(
        is_timeseries=True,
        timestamp_column="ts",
        timestamp_format="elapsed_seconds",
        group_training_examples_by="group",
        rope_scaling_factor=1,
    )

    with pytest.raises(TimeSeriesDataValidationError) as exc_info:
        validate_timeseries_data(df, config)

    assert exc_info.value.reason is TimeSeriesValidationReason.TIMESERIES_GROUP_LENGTH_MISMATCH
