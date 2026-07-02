# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for timeseries_preprocessing module."""

import pandas as pd
import pytest

from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.defaults import PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.errors import DataError, ParameterError
from nemo_safe_synthesizer.training.timeseries_preprocessing import (
    _add_pseudo_group_if_needed,
    _create_elapsed_time_column,
    _GroupTimestampStats,
    _infer_and_convert_timestamp_format,
    _sort_by_group_and_timestamp,
    _validate_start_stop_consistency,
    process_timeseries_data,
)


def test_add_pseudo_group_when_no_group_column():
    """Test PSEUDO_GROUP_COLUMN is added when group_training_examples_by is None."""
    df = pd.DataFrame({"value": [1, 2, 3]})
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by=None,
        is_timeseries=True,
        timestamp_interval_seconds=60,
        rope_scaling_factor=1,
    )

    df_result, group_col = _add_pseudo_group_if_needed(df.copy(), config)

    assert PSEUDO_GROUP_COLUMN in df_result.columns
    assert group_col == PSEUDO_GROUP_COLUMN
    assert config.data.group_training_examples_by == PSEUDO_GROUP_COLUMN
    assert df_result[PSEUDO_GROUP_COLUMN].nunique() == 1


def test_add_pseudo_group_preserves_existing_group():
    """Test existing group column is preserved when specified."""
    df = pd.DataFrame({"group_id": ["A", "B"], "value": [1, 2]})
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by="group_id",
        is_timeseries=True,
        timestamp_interval_seconds=60,
        rope_scaling_factor=1,
    )

    df_result, group_col = _add_pseudo_group_if_needed(df.copy(), config)

    assert PSEUDO_GROUP_COLUMN not in df_result.columns
    assert group_col == "group_id"


def test_create_elapsed_time_column_with_groups():
    """Test elapsed time resets at start of each group."""
    df = pd.DataFrame({"group_id": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})
    config = SafeSynthesizerParameters.from_params(
        rope_scaling_factor=1,
    )
    config.time_series.is_timeseries = True
    config.time_series.timestamp_column = None
    config.time_series.timestamp_interval_seconds = 60

    df_result, is_elapsed = _create_elapsed_time_column(df.copy(), config.time_series, group_by_col="group_id")

    assert is_elapsed is True
    assert config.time_series.timestamp_column in df_result.columns
    # Group A: 0, 60; Group B: 0, 60
    assert list(df_result[config.time_series.timestamp_column]) == [0, 60, 0, 60]
    assert config.time_series.timestamp_format == "elapsed_seconds"


def test_create_elapsed_time_column_no_groups():
    """Test elapsed time uses global index when no groups."""
    df = pd.DataFrame({"value": [1, 2, 3]})
    config = SafeSynthesizerParameters.from_params(
        rope_scaling_factor=1,
    )
    config.time_series.is_timeseries = True
    config.time_series.timestamp_column = None
    config.time_series.timestamp_interval_seconds = 30

    df_result, is_elapsed = _create_elapsed_time_column(df.copy(), config.time_series, group_by_col=None)

    assert is_elapsed is True
    # Global index: 0, 30, 60
    assert list(df_result[config.time_series.timestamp_column]) == [0, 30, 60]


def test_create_elapsed_time_column_skips_when_timestamp_exists():
    """Test no elapsed time created when timestamp_column is already set."""
    df = pd.DataFrame({"timestamp": ["2024-01-01", "2024-01-02"], "value": [1, 2]})
    config = SafeSynthesizerParameters.from_params(
        rope_scaling_factor=1,
    )
    config.time_series.is_timeseries = True
    config.time_series.timestamp_column = "timestamp"

    df_result, is_elapsed = _create_elapsed_time_column(df.copy(), config.time_series, group_by_col=None)

    assert is_elapsed is False
    assert "elapsed_seconds" not in df_result.columns


def test_sort_by_group_and_timestamp():
    """Test sorting by group then timestamp."""
    df = pd.DataFrame(
        {
            "group_id": ["B", "A", "B", "A"],
            "timestamp": [2, 1, 1, 2],
            "value": [1, 2, 3, 4],
        }
    )

    df_result = _sort_by_group_and_timestamp(df, "group_id", "timestamp")

    # Should be sorted: A-1, A-2, B-1, B-2
    assert list(df_result["group_id"]) == ["A", "A", "B", "B"]
    assert list(df_result["timestamp"]) == [1, 2, 1, 2]


def test_sort_by_timestamp_only():
    """Test sorting by timestamp when no group column."""
    df = pd.DataFrame(
        {
            "timestamp": [3, 1, 2],
            "value": ["c", "a", "b"],
        }
    )

    df_result = _sort_by_group_and_timestamp(df, None, "timestamp")

    assert list(df_result["timestamp"]) == [1, 2, 3]
    assert list(df_result["value"]) == ["a", "b", "c"]


def test_validate_start_stop_consistency_valid():
    """Test validation passes when all groups have same start/stop."""
    stats = [
        _GroupTimestampStats("A", "2024-01-01", "2024-01-03", 3600),
        _GroupTimestampStats("B", "2024-01-01", "2024-01-03", 3600),
    ]

    start, stop = _validate_start_stop_consistency(stats)

    assert start == "2024-01-01"
    assert stop == "2024-01-03"


def test_validate_start_stop_consistency_different_starts_raises():
    """Test DataError when groups have different start timestamps."""
    stats = [
        _GroupTimestampStats("A", "2024-01-01", "2024-01-03", 3600),
        _GroupTimestampStats("B", "2024-01-02", "2024-01-03", 3600),  # Different start
    ]

    with pytest.raises(DataError, match="Start timestamps differ across groups"):
        _validate_start_stop_consistency(stats)


def test_validate_start_stop_consistency_different_stops_raises():
    """Test DataError when groups have different stop timestamps."""
    stats = [
        _GroupTimestampStats("A", "2024-01-01", "2024-01-03", 3600),
        _GroupTimestampStats("B", "2024-01-01", "2024-01-04", 3600),  # Different stop
    ]

    with pytest.raises(DataError, match="Stop timestamps differ across groups"):
        _validate_start_stop_consistency(stats)


class TestInferAndConvertTimestampFormat:
    """Tests for _infer_and_convert_timestamp_format."""

    @pytest.mark.parametrize(
        "column_name,values,expected_match",
        [
            pytest.param(
                "ts", ["not_a_date", "also_not"], "Could not infer timestamp format", id="non_datetime_strings"
            ),
            pytest.param("my_col", [42, 99], r"column 'my_col'.*first value: '42'", id="names_column_and_first_value"),
            pytest.param("ts", [100, 200], "elapsed_seconds", id="suggests_elapsed_seconds_for_numeric"),
        ],
    )
    def test_raises_parameter_error_with_informative_message(self, column_name, values, expected_match):
        """ParameterError is raised when format cannot be inferred, with an actionable message."""
        df = pd.DataFrame({column_name: values})
        config = SafeSynthesizerParameters.from_params(rope_scaling_factor=1)
        config.time_series.timestamp_column = column_name
        config.time_series.timestamp_format = None

        with pytest.raises(ParameterError, match=expected_match):
            _infer_and_convert_timestamp_format(df, config.time_series)


class TestProcessTimeseriesElapsedSecondsDetection:
    """Tests for numeric elapsed-seconds detection in process_timeseries_data."""

    @staticmethod
    def _make_config(**overrides):
        defaults = dict(
            is_timeseries=True,
            rope_scaling_factor=1,
        )
        defaults.update(overrides)
        return SafeSynthesizerParameters.from_params(**defaults)

    @pytest.mark.parametrize(
        "ts_values,expected_values",
        [
            pytest.param([0, 60, 120], [0, 60, 120], id="int"),
            pytest.param([0.0, 60.0, 120.0], [0.0, 60.0, 120.0], id="float"),
        ],
    )
    def test_explicit_elapsed_seconds_accepts_numeric_dtype(self, ts_values, expected_values):
        """timestamp_format='elapsed_seconds' is accepted for any numeric dtype.

        Float is allowed when the user opts in explicitly, even though downstream
        interval handling currently operates at integer-second resolution.
        """
        df = pd.DataFrame({"group": ["A", "A", "A"], "ts": ts_values, "value": [1, 2, 3]})
        config = self._make_config(
            timestamp_column="ts",
            timestamp_format="elapsed_seconds",
            group_training_examples_by="group",
        )

        result_df, result_config = process_timeseries_data(df.copy(), config)

        assert result_config.time_series.timestamp_format == "elapsed_seconds"
        assert list(result_df["ts"]) == expected_values

    @pytest.mark.parametrize(
        "ts_values,timestamp_format,expected_match",
        [
            pytest.param(
                ["0", "60", "120"],
                "elapsed_seconds",
                "requires timestamp column .* to be numeric",
                id="explicit_elapsed_seconds_rejects_string",
            ),
            pytest.param(
                [0.0, 0.5, 1.0],
                None,
                "Could not infer timestamp format",
                id="auto_detection_rejects_float",
            ),
        ],
    )
    def test_elapsed_seconds_detection_rejects_unsupported_dtypes(self, ts_values, timestamp_format, expected_match):
        """Non-supported dtypes raise ParameterError with an actionable message.

        Explicit ``elapsed_seconds`` on non-numeric data fails fast instead of surfacing
        a cryptic TypeError later. Auto-detection is intentionally restricted to integer
        dtypes (see ``_detect_elapsed_seconds_format``); floats must opt in explicitly.
        """
        df = pd.DataFrame({"group": ["A", "A", "A"], "ts": ts_values, "value": [1, 2, 3]})
        config = self._make_config(
            timestamp_column="ts",
            timestamp_format=timestamp_format,
            group_training_examples_by="group",
        )

        with pytest.raises(ParameterError, match=expected_match):
            process_timeseries_data(df.copy(), config)

    @pytest.mark.parametrize(
        "ts_values,expected_format_is_elapsed",
        [
            pytest.param([0, 30, 60], True, id="int_auto_detected"),
            pytest.param(
                ["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"],
                False,
                id="datetime_string_falls_through",
            ),
        ],
    )
    def test_auto_detection_when_format_not_provided(self, ts_values, expected_format_is_elapsed):
        """Auto-detection: integer columns are marked elapsed; datetime strings fall through."""
        df = pd.DataFrame({"group": ["A", "A", "A"], "ts": ts_values, "value": [1, 2, 3]})
        config = self._make_config(timestamp_column="ts", group_training_examples_by="group")
        assert config.time_series.timestamp_format is None

        _, result_config = process_timeseries_data(df.copy(), config)

        is_elapsed = result_config.time_series.timestamp_format == "elapsed_seconds"
        assert is_elapsed is expected_format_is_elapsed

    def test_numeric_column_auto_detected_without_group(self):
        """Auto-detection works when no group_by column is provided (pseudo-group path)."""
        df = pd.DataFrame({"ts": [0, 10, 20], "value": [1, 2, 3]})
        config = self._make_config(timestamp_column="ts")
        assert config.time_series.timestamp_format is None

        _, result_config = process_timeseries_data(df.copy(), config)

        assert result_config.time_series.timestamp_format == "elapsed_seconds"

    def test_missing_timestamp_column_raises_parameter_error(self):
        """A timestamp_column that doesn't exist in the DataFrame raises ParameterError.

        Previously, accessing the missing column during numeric dtype detection would
        surface a pandas KeyError before the friendlier validation error could run.
        """
        df = pd.DataFrame(
            {
                "group": ["A", "A", "A"],
                "value": [1, 2, 3],
            }
        )
        config = self._make_config(
            timestamp_column="not_present",
            group_training_examples_by="group",
        )

        with pytest.raises(ParameterError, match="Timestamp column 'not_present' not found"):
            process_timeseries_data(df.copy(), config)

    def test_explicit_elapsed_seconds_infers_interval(self):
        """Elapsed seconds with consistent intervals correctly infers timestamp_interval_seconds."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "A"],
                "ts": [0, 60, 120],
                "value": [1, 2, 3],
            }
        )
        config = self._make_config(
            timestamp_column="ts",
            timestamp_format="elapsed_seconds",
            group_training_examples_by="group",
        )

        _, result_config = process_timeseries_data(df.copy(), config)

        assert result_config.time_series.timestamp_interval_seconds == 60
