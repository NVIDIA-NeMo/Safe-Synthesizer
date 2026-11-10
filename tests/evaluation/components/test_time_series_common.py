# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.evaluate import RollingStatsSimilarityParameters
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters
from nemo_safe_synthesizer.defaults import PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.evaluation.components.time_series_common import (
    PairResult,
    evaluate_pairs,
    group_frame,
    prepare_values,
    resolve_group_column,
    select_value_columns,
    shared_groups,
    summaries,
)
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets


def _datasets(training_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> EvaluationDatasets:
    return EvaluationDatasets.from_dataframes(training_df, synthetic_df, enable_sampling=False)


def _grouped_config() -> SafeSynthesizerParameters:
    return SafeSynthesizerParameters(
        data=DataParameters(group_training_examples_by="group"),
        time_series=TimeSeriesParameters(is_timeseries=True, timestamp_column="time"),
    )


def _score_by_column(similarities: dict[str, float | None]):
    """Return a pair scorer that gives each column a fixed similarity and skips columns mapped to ``None``."""

    def score_pair(training: pd.Series, synthetic: pd.Series) -> PairResult:
        similarity = similarities[str(training.name)]
        if similarity is None:
            return None, "unusable"
        return {"similarity": similarity}, None

    return score_pair


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


def test_select_value_columns_rejects_explicit_column_stored_as_text_in_synthetic_data():
    training_df = _frame()
    synthetic_df = training_df.copy()
    synthetic_df["count"] = synthetic_df["count"].map(lambda value: f"n{value}")

    _, error, _ = select_value_columns(_datasets(training_df, synthetic_df), ["count"], "time", "group", "test")

    assert error == "Invalid test value columns: 'count' is not numeric in synthetic data"


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


def test_shared_groups_is_empty_when_a_dataset_lacks_the_group_column():
    training = pd.DataFrame({"group": list("AB")})
    synthetic = pd.DataFrame({"other": list("AB")})

    assert shared_groups(training, synthetic, "group", 3, seed=5) == ([], [], [], 0)


def test_shared_groups_without_group_column_is_one_global_sequence():
    frame = pd.DataFrame({"value": [1, 2]})

    assert shared_groups(frame, frame, None, 3, seed=5) == ([None], [], [], 1)


def test_group_frame_sorts_stably_by_timestamp():
    frame = pd.DataFrame({"group": ["A", "A", "A", "B"], "time": [2, 1, 1, 0], "value": [0, 1, 2, 3]})

    ordered = group_frame(frame, "group", "A", "time")

    assert ordered["value"].tolist() == [1, 2, 0]


@pytest.mark.parametrize("dtype", [pd.Int64Dtype(), pd.Float64Dtype()], ids=["Int64", "Float64"])
def test_prepare_values_treats_nullable_missing_values_as_gaps(dtype: pd.Int64Dtype | pd.Float64Dtype):
    values = pd.Series([1, pd.NA, 3, pd.NA], dtype=dtype)

    np.testing.assert_array_equal(prepare_values(values), [1.0, np.nan, 3.0, np.nan])


def test_prepare_values_treats_non_finite_and_non_numeric_values_as_gaps():
    values = pd.Series([1.0, np.inf, -np.inf, "text", 2.0], dtype=object)

    np.testing.assert_array_equal(prepare_values(values), [1.0, np.nan, np.nan, np.nan, 2.0])


def test_evaluate_pairs_averages_scored_pairs_and_leaves_skipped_pairs_out():
    frame = _frame()

    score, details = evaluate_pairs(
        _datasets(frame, frame.copy()),
        _grouped_config(),
        RollingStatsSimilarityParameters(),
        _score_by_column({"alpha": 0.2, "count": None, "zeta": 0.6}),
    )

    assert score.score == 4.0
    assert score.notes == "Skipped 2 unusable group/column comparisons."
    assert {(item["group"], item["column"]) for item in details["skipped"]} == {("A", "count"), ("B", "count")}


def test_evaluate_pairs_only_scores_configured_value_columns():
    frame = _frame()

    score, details = evaluate_pairs(
        _datasets(frame, frame.copy()),
        _grouped_config(),
        RollingStatsSimilarityParameters(value_columns=["zeta"]),
        _score_by_column({"alpha": 0.0, "count": 0.0, "zeta": 0.9}),
    )

    assert score.score == 9.0
    assert {item["column"] for item in details["profiles"]} == {"zeta"}


def test_evaluate_pairs_samples_groups_reproducibly_above_the_cap():
    frame = pd.DataFrame({"group": np.repeat(list("ABCDEF"), 4), "value": np.arange(24.0)})
    cfg = RollingStatsSimilarityParameters(max_groups=2)
    score_pair = _score_by_column({"value": 0.5})

    first = evaluate_pairs(_datasets(frame, frame.copy()), _grouped_config(), cfg, score_pair)
    second = evaluate_pairs(_datasets(frame, frame.copy()), _grouped_config(), cfg, score_pair)

    assert first == second
    score, details = first
    assert len({item["group"] for item in details["profiles"]}) == 2
    assert score.notes == "Evaluated 2 of 6 shared groups using a reproducible sample."


@pytest.mark.parametrize(
    ("synthetic_value", "similarity"),
    [
        pytest.param([f"v{index}" for index in range(12)], 1.0, id="no_columns_remain"),
        pytest.param(list(range(12)), None, id="no_pair_is_usable"),
    ],
)
def test_evaluate_pairs_names_left_out_columns_when_unavailable(synthetic_value: list, similarity: float | None):
    training_df = pd.DataFrame({"value": range(12), "count": range(12)})
    synthetic_df = training_df.assign(value=synthetic_value, count=[f"n{index}" for index in range(12)])

    score, details = evaluate_pairs(
        _datasets(training_df, synthetic_df),
        None,
        RollingStatsSimilarityParameters(),
        _score_by_column({"value": similarity}),
    )

    assert score.score is None
    assert score.notes is not None
    assert "count" in score.notes
    assert "count" in details["excluded_columns"]


def test_evaluate_pairs_returns_unexpected_failures_as_notes():
    frame = _frame()

    def fail(training: pd.Series, synthetic: pd.Series) -> PairResult:
        raise RuntimeError("unexpected metric failure")

    score, details = evaluate_pairs(_datasets(frame, frame.copy()), None, RollingStatsSimilarityParameters(), fail)

    assert score.score is None
    assert score.notes == "unexpected metric failure"
    assert details == {}


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
