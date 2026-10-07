# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for flexible time-series control columns."""

import pandas as pd
import pytest

from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.data_processing.flexible_timeseries import finalize_flexible_timeseries_controls
from nemo_safe_synthesizer.training.timeseries_preprocessing import process_timeseries_data


def _flexible_training_data():
    data = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B"],
            "timestamp": [0, 1, 2, 0, 1],
            "value": [1, 2, 3, 4, 5],
        }
    )
    config = SafeSynthesizerParameters.from_params(
        is_timeseries=True,
        timestamp_column="timestamp",
        timestamp_format="elapsed_seconds",
        group_training_examples_by="group",
        rope_scaling_factor=1,
    )
    prepared, resolved, metadata = process_timeseries_data(data, config)
    assert metadata is not None
    assert metadata.flexible is not None
    return prepared, resolved, metadata.flexible


@pytest.mark.parametrize(
    ("dropped_value", "expected_a_indices", "expected_a_markers"),
    [
        pytest.param(2, [0, 1], [False, True], id="middle-row"),
        pytest.param(3, [0, 1], [False, True], id="final-row"),
    ],
)
def test_flexible_timeseries_finalize_controls_after_rows_are_removed(
    dropped_value, expected_a_indices, expected_a_markers
):
    prepared, resolved, metadata = _flexible_training_data()
    preprocessed = prepared[prepared["value"] != dropped_value]

    finalized, updated = finalize_flexible_timeseries_controls(preprocessed, resolved, metadata)

    group_a = finalized[finalized["group"] == "A"]
    assert group_a[metadata.index_column].tolist() == expected_a_indices
    assert group_a[metadata.marker_column].tolist() == expected_a_markers
    assert updated.max_records == 2
    assert resolved.time_series.stop_timestamp == 1


def test_flexible_timeseries_finalize_controls_keeps_unchanged_data():
    prepared, resolved, metadata = _flexible_training_data()

    finalized, updated = finalize_flexible_timeseries_controls(prepared, resolved, metadata)

    pd.testing.assert_frame_equal(finalized, prepared.reset_index(drop=True))
    assert updated == metadata
