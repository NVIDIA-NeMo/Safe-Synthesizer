# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.defaults import PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.evaluation.components.time_series_common import (
    group_frame,
    resolve_group_column,
    select_value_columns,
    shared_groups,
    summaries,
)
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets


def _datasets(training_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> EvaluationDatasets:
    return EvaluationDatasets.from_dataframes(training_df, synthetic_df, enable_sampling=False)


def _frame(points: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "time": range(points),
            "group": np.resize(["A", "B"], points),
            "zeta": rng.normal(size=points),
            "alpha": rng.normal(size=points),
            "count": np.round(10 + 80 * rng.random(points)).astype(int),
        }
    )


def test_select_value_columns_keeps_explicit_order_and_drops_duplicates():
    frame = _frame()

    columns, error, excluded = select_value_columns(
        _datasets(frame, frame.copy()), ["zeta", "alpha", "zeta"], "time", "group", "test"
    )

    assert columns == ["zeta", "alpha"]
    assert error is None
    assert excluded == []


def test_select_value_columns_reports_invalid_explicit_columns_with_label():
    frame = _frame()

    columns, error, _ = select_value_columns(
        _datasets(frame, frame.copy()), ["missing", "group"], "time", "group", "rolling stats"
    )

    assert columns == []
    assert error is not None
    assert error.startswith("Invalid rolling stats value columns: ")
    assert "'missing' is missing from training and synthetic data" in error
    assert "'group' is used for timestamp or sequence grouping" in error


def test_select_value_columns_excludes_timestamp_group_and_pseudo_group_columns():
    frame = _frame()
    frame[PSEUDO_GROUP_COLUMN] = frame["count"]

    columns, error, _ = select_value_columns(_datasets(frame, frame.copy()), None, "time", "group", "test")

    assert error is None
    assert columns == ["alpha", "count", "zeta"]


def test_select_value_columns_keeps_training_numeric_column_that_collapsed_in_synthetic():
    training_df = _frame()
    synthetic_df = training_df.copy()
    synthetic_df["count"] = 0

    columns, error, excluded = select_value_columns(_datasets(training_df, synthetic_df), None, "time", "group", "test")

    assert error is None
    assert "count" in columns
    assert excluded == []


def test_select_value_columns_excludes_columns_synthetic_data_stores_as_text():
    training_df = _frame()
    synthetic_df = training_df.copy()
    synthetic_df["count"] = synthetic_df["count"].map(lambda value: f"n{value}")

    columns, _, excluded = select_value_columns(_datasets(training_df, synthetic_df), None, "time", "group", "test")

    assert "count" not in columns
    assert excluded == ["count"]


def test_resolve_group_column_treats_missing_pseudo_group_as_global():
    frame = _frame()
    config = SafeSynthesizerParameters(data=DataParameters(group_training_examples_by=PSEUDO_GROUP_COLUMN))

    assert resolve_group_column(_datasets(frame, frame.copy()), config) == (None, None)


def test_resolve_group_column_reports_missing_configured_column():
    frame = _frame()
    config = SafeSynthesizerParameters(data=DataParameters(group_training_examples_by="sequence"))

    group_column, error = resolve_group_column(_datasets(frame, frame.copy()), config)

    assert group_column is None
    assert error == "Configured group column 'sequence' is missing from a dataset."


def test_resolve_group_column_without_config_is_global():
    frame = _frame()

    assert resolve_group_column(_datasets(frame, frame.copy()), None) == (None, None)


def test_shared_groups_sample_is_seeded_sorted_and_reports_one_sided_groups():
    training = pd.DataFrame({"group": list("ABCDEFX")})
    synthetic = pd.DataFrame({"group": list("ABCDEFY")})

    first = shared_groups(training, synthetic, "group", 3, seed=5)
    second = shared_groups(training, synthetic, "group", 3, seed=5)

    assert first == second
    groups, only_training, only_synthetic, shared_count = first
    assert groups == sorted(groups)
    assert len(groups) == 3
    assert only_training == ["X"]
    assert only_synthetic == ["Y"]
    assert shared_count == 6


def test_shared_groups_without_group_column_is_one_global_sequence():
    frame = pd.DataFrame({"value": [1, 2]})

    assert shared_groups(frame, frame, None, 3, seed=5) == ([None], [], [], 1)


def test_group_frame_sorts_stably_by_timestamp():
    frame = pd.DataFrame({"group": ["A", "A", "A", "B"], "time": [2, 1, 1, 0], "value": [0, 1, 2, 3]})

    ordered = group_frame(frame, "group", "A", "time")

    assert ordered["value"].tolist() == [1, 2, 0]


def test_summaries_average_by_key_in_stable_order():
    profiles = [
        {"group": "B", "similarity": 0.2},
        {"group": "A", "similarity": 1.0},
        {"group": "B", "similarity": 0.4},
    ]

    assert summaries(profiles, "group") == [
        {"group": "A", "similarity": 1.0, "count": 1},
        {"group": "B", "similarity": pytest.approx(0.3), "count": 2},
    ]
