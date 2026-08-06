# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for timeseries_preprocessing module."""

import pandas as pd
import pytest

from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.defaults import PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.training.timeseries_preprocessing import process_timeseries_data


def test_process_timeseries_data_adds_pseudo_group_and_elapsed_timestamp():
    """Ungrouped time-series data is normalized through the shared validator."""
    df = pd.DataFrame({"value": [1, 2, 3]})
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by=None,
        is_timeseries=True,
        timestamp_interval_seconds=60,
        rope_scaling_factor=1,
    )

    df_result, result_config = process_timeseries_data(df.copy(), config)

    assert PSEUDO_GROUP_COLUMN in df_result.columns
    assert result_config.data.group_training_examples_by == PSEUDO_GROUP_COLUMN
    assert result_config.data.order_training_examples_by == "elapsed_seconds"
    assert result_config.time_series.timestamp_column == "elapsed_seconds"
    assert result_config.time_series.timestamp_format == "elapsed_seconds"
    assert result_config.time_series.timestamp_interval_seconds == 60
    assert result_config.time_series.start_timestamp == "0"
    assert result_config.time_series.stop_timestamp == "120"
    assert list(df_result["elapsed_seconds"]) == [0, 60, 120]
    assert df_result[PSEUDO_GROUP_COLUMN].nunique() == 1
    assert list(df_result.columns) == [PSEUDO_GROUP_COLUMN, "elapsed_seconds", "value"]


def test_process_timeseries_data_preserves_existing_group_column():
    """Grouped time-series data keeps its configured group column."""
    df = pd.DataFrame({"group_id": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by="group_id",
        is_timeseries=True,
        timestamp_interval_seconds=60,
        rope_scaling_factor=1,
    )

    df_result, result_config = process_timeseries_data(df.copy(), config)

    assert PSEUDO_GROUP_COLUMN not in df_result.columns
    assert result_config.data.group_training_examples_by == "group_id"
    assert list(df_result["elapsed_seconds"]) == [0, 60, 0, 60]
    assert list(df_result.columns) == ["group_id", "elapsed_seconds", "value"]


def test_process_timeseries_data_sorts_by_group_and_timestamp():
    """Processed data is sorted by group and timestamp."""
    df = pd.DataFrame(
        {
            "value": [1, 2, 3, 4],
            "timestamp": [2, 1, 1, 2],
            "group_id": ["B", "A", "B", "A"],
        }
    )
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by="group_id",
        is_timeseries=True,
        timestamp_column="timestamp",
        timestamp_format="elapsed_seconds",
        rope_scaling_factor=1,
    )

    df_result, _ = process_timeseries_data(df.copy(), config)

    # Should be sorted: A-1, A-2, B-1, B-2
    assert list(df_result.columns) == ["group_id", "timestamp", "value"]
    assert list(df_result["group_id"]) == ["A", "A", "B", "B"]
    assert list(df_result["timestamp"]) == [1, 2, 1, 2]


def test_process_timeseries_data_sorts_ungrouped_timestamp_data():
    """Ungrouped explicit timestamp data is sorted by timestamp."""
    df = pd.DataFrame(
        {
            "timestamp": [3, 1, 2],
            "value": ["c", "a", "b"],
        }
    )
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by=None,
        is_timeseries=True,
        timestamp_column="timestamp",
        timestamp_format="elapsed_seconds",
        rope_scaling_factor=1,
    )

    df_result, _ = process_timeseries_data(df.copy(), config)

    assert list(df_result["timestamp"]) == [1, 2, 3]
    assert list(df_result["value"]) == ["a", "b", "c"]


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
