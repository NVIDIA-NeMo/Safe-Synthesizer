# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for timeseries_preprocessing module."""

import pandas as pd
import pytest
from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.defaults import PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.errors import DataError
from nemo_safe_synthesizer.training.timeseries_preprocessing import (
    _add_pseudo_group_if_needed,
    _create_elapsed_time_column,
    _GroupTimestampStats,
    _sort_by_group_and_timestamp,
    _validate_start_stop_consistency,
)


def test_add_pseudo_group_when_no_group_column():
    """Test PSEUDO_GROUP_COLUMN is added when group_training_examples_by is None."""
    df = pd.DataFrame({"value": [1, 2, 3]})
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by=None,
        is_timeseries=True,
        timestamp_interval_seconds=60,
        use_unsloth=False,
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
        use_unsloth=False,
        rope_scaling_factor=1,
    )

    df_result, group_col = _add_pseudo_group_if_needed(df.copy(), config)

    assert PSEUDO_GROUP_COLUMN not in df_result.columns
    assert group_col == "group_id"


def test_create_elapsed_time_column_with_groups():
    """Test elapsed time resets at start of each group."""
    df = pd.DataFrame({"group_id": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})
    config = SafeSynthesizerParameters.from_params(
        use_unsloth=False,
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
        use_unsloth=False,
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
        use_unsloth=False,
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
