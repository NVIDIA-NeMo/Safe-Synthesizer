# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared time-series validation helpers."""

import pandas as pd
import pytest

from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.data_processing.timeseries_validation import (
    TimeSeriesDataValidationError,
    TimeSeriesGroupTimestampStats,
    _infer_and_convert_timestamp_format,
    validate_start_stop_consistency,
    validate_timeseries_data,
)
from nemo_safe_synthesizer.defaults import PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.errors import DataError, ParameterError


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
    "column_name,values,expected_match",
    [
        pytest.param("ts", ["not_a_date", "also_not"], "Could not infer timestamp format", id="non_datetime_strings"),
        pytest.param("my_col", [42, 99], r"column 'my_col'.*first value: '42'", id="names_column_and_first_value"),
        pytest.param("ts", [100, 200], "elapsed_seconds", id="suggests_elapsed_seconds_for_numeric"),
    ],
)
def test_infer_and_convert_timestamp_format_raises_informative_error(column_name, values, expected_match):
    """Timestamp format inference failures include actionable context."""
    df = pd.DataFrame({column_name: values})
    config = SafeSynthesizerParameters.from_params(rope_scaling_factor=1)
    config.time_series.timestamp_column = column_name
    config.time_series.timestamp_format = None

    with pytest.raises(ParameterError, match=expected_match):
        _infer_and_convert_timestamp_format(df, config.time_series)


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

    assert exc_info.value.code == "timeseries_empty"


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

    assert exc_info.value.code == "timestamp_interval_mismatch"


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

    assert exc_info.value.code == "timeseries_group_length_mismatch"
