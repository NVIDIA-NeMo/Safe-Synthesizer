# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.evaluate import (
    EvaluationParameters,
    RollingStatsSimilarityParameters,
    TimeSeriesEvaluationParameters,
)
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters
from nemo_safe_synthesizer.defaults import PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.evaluation.components.rolling_stats_similarity import RollingStatsSimilarity
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets
from nemo_safe_synthesizer.training.timeseries_preprocessing import process_timeseries_data


def _config(
    metric: RollingStatsSimilarityParameters | None = None,
    *,
    group_column: str | None = None,
) -> SafeSynthesizerParameters:
    return SafeSynthesizerParameters(
        data=DataParameters(group_training_examples_by=group_column),
        time_series=TimeSeriesParameters(is_timeseries=True, timestamp_column="time"),
        evaluation=EvaluationParameters(
            time_series=TimeSeriesEvaluationParameters(
                enabled=True,
                rolling_stats=metric or RollingStatsSimilarityParameters(),
            )
        ),
    )


def _datasets(training_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> EvaluationDatasets:
    return EvaluationDatasets.from_dataframes(training_df, synthetic_df, enable_sampling=False)


def _grouped_rows(groups: list[tuple[str, int]], points: int = 30) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for group, offset in groups:
        for index in range(points):
            rows.append({"group": group, "time": index, "value": offset + index + 3 * np.sin(index)})
    return rows


def _noise(rng: np.random.Generator, points: int) -> np.ndarray:
    """Short-term noise: AR 0.5 scaled to unit variance."""
    values = np.zeros(points)
    for index in range(1, points):
        values[index] = 0.5 * values[index - 1] + rng.normal()
    return values * np.sqrt(0.75)


def _rising(rng: np.random.Generator, points: int) -> np.ndarray:
    """Level rises 3 noise standard deviations from start to end while the spread doubles."""
    position = np.linspace(0.0, 1.0, points)
    return 3 * position + (1 + position) * _noise(rng, points)


def _single_series_score(training: np.ndarray, synthetic: np.ndarray) -> float | None:
    datasets = _datasets(
        pd.DataFrame({"time": np.arange(len(training)), "value": training}),
        pd.DataFrame({"time": np.arange(len(synthetic)), "value": synthetic}),
    )
    return RollingStatsSimilarity.from_evaluation_datasets(datasets, _config()).score.score


def _mean_score(make_pair, points: int, trials: int = 20, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    scores = [_single_series_score(*make_pair(rng, points)) for _ in range(trials)]
    finite = [score for score in scores if score is not None]
    assert len(finite) == trials
    return float(np.mean(finite))


def _expected_error(profile: dict) -> float:
    excess = magnitude = 0.0
    for statistic, noise_key, reference in (
        ("level", "level_noise", 0.0),
        ("spread", "spread_noise", profile["spread_reference"]),
    ):
        training = np.array(profile[f"training_{statistic}"], dtype=float)
        synthetic = np.array(profile[f"synthetic_{statistic}"], dtype=float)
        noise = profile[noise_key]
        structure = np.maximum(np.abs(training - reference), np.abs(synthetic - reference))
        excess += np.sum((training - synthetic) ** 2 - noise) / noise
        magnitude += np.sum(np.maximum(structure**2, noise)) / noise
    return float(np.sqrt(max(excess, 0.0) / magnitude))


def test_rolling_stats_similarity_formula_is_noise_aware_and_pooled_over_segments():
    rng = np.random.default_rng(2)
    component = RollingStatsSimilarity.from_evaluation_datasets(
        _datasets(
            pd.DataFrame({"time": range(300), "value": _rising(rng, 300)}),
            pd.DataFrame({"time": range(300), "value": 0.6 * _rising(rng, 300)}),
        ),
        _config(),
    )

    profile = component.details["profiles"][0]
    expected = _expected_error(profile)
    assert profile["window"] == 20
    assert profile["segments"] == 10
    assert len(profile["training_level"]) == len(profile["synthetic_spread"]) == 10
    assert 0 < expected < 1
    assert profile["error"] == pytest.approx(expected, abs=1e-12)
    assert component.score.score == pytest.approx(10 * (1 - expected), abs=0.1)


@pytest.mark.parametrize("points", [200, 1000])
def test_rolling_stats_similarity_scores_same_process_high(points: int):
    assert _mean_score(lambda rng, n: (_rising(rng, n), _rising(rng, n)), points) >= 8.0


@pytest.mark.parametrize("points", [200, 1000])
def test_rolling_stats_similarity_scores_independent_noise_in_mild_tier_or_better(points: int):
    assert _mean_score(lambda rng, n: (_noise(rng, n), _noise(rng, n)), points, trials=40) >= 6.5


@pytest.mark.parametrize(
    "make_synthetic",
    [
        pytest.param(lambda rng, training: training[::-1], id="runs_backward"),
        pytest.param(
            lambda rng, training: (1 + np.linspace(0, 1, len(training))) * _noise(rng, len(training)),
            id="missing_trend",
        ),
        pytest.param(lambda rng, training: rng.permutation(training), id="shuffled"),
    ],
)
def test_rolling_stats_similarity_scores_lost_or_reversed_structure_in_lost_tier(make_synthetic):
    def make_pair(rng: np.random.Generator, points: int) -> tuple[np.ndarray, np.ndarray]:
        training = _rising(rng, points)
        return training, make_synthetic(rng, training)

    assert _mean_score(make_pair, 1000, trials=10) < 2.0


def test_rolling_stats_similarity_penalizes_a_smaller_spread():
    def make_pair(rng: np.random.Generator, points: int) -> tuple[np.ndarray, np.ndarray]:
        position = np.linspace(0.0, 1.0, points)
        return _rising(rng, points), 3 * position + 0.5 * (1 + position) * _noise(rng, points)

    assert _mean_score(make_pair, 1000, trials=10) < 5.0


def test_rolling_stats_similarity_identical_grouped_series_are_scored_per_pair():
    training_df = pd.DataFrame(_grouped_rows([("B", 100), ("A", 0)])).sample(frac=1.0, random_state=7)
    config = _config(group_column="group")
    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(training_df, training_df.copy()), config)

    assert component.score.score == 10
    assert component.details["counts"]["groups"] == 2
    assert component.details["counts"]["evaluated_profiles"] == 2
    assert [row["group"] for row in component.details["per_group"]] == ["A", "B"]


def test_rolling_stats_similarity_keeps_real_column_named_like_pseudo_group():
    training_df = pd.DataFrame(_grouped_rows([("B", 100), ("A", 0)])).rename(columns={"group": PSEUDO_GROUP_COLUMN})
    config = _config(group_column=PSEUDO_GROUP_COLUMN)
    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(training_df, training_df.copy()), config)

    assert component.details["counts"]["groups"] == 2


def test_rolling_stats_similarity_treats_inherited_pseudo_group_as_global_sequence():
    training_df = pd.DataFrame({"value": [3.0, 0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0]})
    config = SafeSynthesizerParameters.from_params(
        is_timeseries=True,
        timestamp_interval_seconds=1,
        rope_scaling_factor=1,
    )
    processed_df, config, _ = process_timeseries_data(training_df.copy(), config)
    synthetic_df = processed_df.drop(columns=PSEUDO_GROUP_COLUMN).sample(frac=1.0, random_state=7)

    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(training_df, synthetic_df), config)

    assert component.score.score == 10
    assert component.details["evaluation_mode"] == "global"
    assert component.details["group_column"] is None
    assert component.details["timestamp_column"] == "elapsed_seconds"


def test_rolling_stats_similarity_excludes_inherited_pseudo_group_from_value_columns():
    frame = pd.DataFrame(
        {
            PSEUDO_GROUP_COLUMN: [0] * 8,
            "time": range(8),
            "value": [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 4.0],
        }
    )
    config = _config()
    config.data.group_training_examples_by = PSEUDO_GROUP_COLUMN

    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), config)

    assert component.score.score == 10
    assert [row["column"] for row in component.details["per_column"]] == ["value"]


def test_rolling_stats_similarity_reports_missing_explicit_group_column():
    frame = pd.DataFrame({"time": range(6), "value": [0.0, 1.0, 2.0, 3.0, 2.0, 1.0]})
    config = _config(group_column="missing_group")

    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), config)

    assert component.score.score is None
    assert component.score.notes == "Configured group column 'missing_group' is missing from a dataset."


def test_rolling_stats_similarity_constant_training_series_is_unavailable_instead_of_perfect():
    training_df = pd.DataFrame({"time": range(8), "value": [1.0] * 8})
    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(training_df, training_df.copy()), _config())

    assert component.score.score is None
    assert component.score.notes == "No usable group/column rolling statistics profiles."
    assert "training sequence is constant" in component.details["skipped"][0]["reason"].lower()


def test_rolling_stats_similarity_scores_synthetic_constant_collapse_as_failure():
    training_df = pd.DataFrame(_grouped_rows([("A", 0), ("B", 100)]))
    synthetic_df = training_df.copy()
    synthetic_df.loc[synthetic_df["group"] == "B", "value"] = 100
    config = _config(group_column="group")

    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(training_df, synthetic_df), config)

    assert component.score.score == 5
    collapsed = next(item for item in component.details["profiles"] if item["group"] == "B")
    assert collapsed["similarity"] == 0
    assert collapsed["synthetic_level"] == [None] * collapsed["segments"]
    assert "synthetic sequence is constant" in collapsed["reason"].lower()
    assert component.details["counts"]["skipped"] == 0


def test_rolling_stats_similarity_handles_aligned_non_finite_gaps():
    values = [0.0, 1.0, np.inf, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0]
    training_df = pd.DataFrame({"time": range(10), "value": values})
    synthetic_df = pd.DataFrame({"time": range(10), "value": [-np.inf if np.isinf(v) else v for v in values]})

    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(training_df, synthetic_df), _config())

    assert component.score.score == 10
    profile = component.details["profiles"][0]
    assert all(value is not None for value in profile["training_level"])


def test_rolling_stats_similarity_preserves_non_finite_positions():
    rng = np.random.default_rng(4)
    training = _rising(rng, 200)
    synthetic = training.copy()
    synthetic[:40] = np.nan

    datasets = _datasets(
        pd.DataFrame({"time": range(200), "value": training}),
        pd.DataFrame({"time": range(200), "value": synthetic}),
    )
    profile = RollingStatsSimilarity.from_evaluation_datasets(datasets, _config()).details["profiles"][0]

    assert profile["synthetic_level"][0] is None
    assert profile["synthetic_level"][-1] == pytest.approx(profile["training_level"][-1])


def test_rolling_stats_similarity_revalidates_usable_length_with_non_finite_values():
    frame = pd.DataFrame({"time": range(5), "value": [0.0, np.inf, np.nan, 1.0, 2.0]})

    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), _config())

    assert component.score.score is None
    assert "at least 4 finite observations" in component.details["skipped"][0]["reason"]


def test_rolling_stats_similarity_caps_window_at_half_the_shorter_sequence():
    rng = np.random.default_rng(1)
    datasets = _datasets(
        pd.DataFrame({"time": range(30), "value": _rising(rng, 30)}),
        pd.DataFrame({"time": range(30), "value": _rising(rng, 30)}),
    )

    profile = RollingStatsSimilarity.from_evaluation_datasets(datasets, _config()).details["profiles"][0]

    assert profile["window"] == 15
    assert profile["segments"] == 10


@pytest.mark.parametrize(
    "values",
    [
        [1e-13, 2e-13, 4e-13, 3e-13, 6e-13, 5e-13, 7e-13, 4e-13],
        [1e12, 1e12 + 1, 1e12 + 4, 1e12 + 2, 1e12 + 7, 1e12 + 3, 1e12 + 5, 1e12 + 6],
    ],
)
def test_rolling_stats_similarity_constant_detection_is_scale_aware(values):
    frame = pd.DataFrame({"time": range(len(values)), "value": values})

    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), _config())

    assert component.score.score == 10


def test_rolling_stats_similarity_rejects_invalid_explicit_value_columns():
    training_df = pd.DataFrame({"time": range(6), "value": range(6), "label": list("abcdef")})
    synthetic_df = pd.DataFrame({"time": range(6), "value": range(6), "label": list("abcdef")})
    config = _config(RollingStatsSimilarityParameters(value_columns=["missing", "label"]))

    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(training_df, synthetic_df), config)

    assert component.score.score is None
    assert component.score.notes is not None
    assert "'missing' is missing from training and synthetic data" in component.score.notes
    assert "'label' is not numeric in training data" in component.score.notes


def test_rolling_stats_similarity_without_config_runs_with_component_defaults():
    frame = pd.DataFrame({"value": [0.0, 1.0, 3.0, 2.0, 4.0, 1.0, 5.0, 2.0]})

    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()))

    assert component.score.score == 10
    assert component.details["evaluation_mode"] == "global"


def test_rolling_stats_similarity_isolates_unexpected_metric_failures(monkeypatch):
    frame = pd.DataFrame({"time": range(8), "value": range(8)})
    datasets = _datasets(frame, frame.copy())

    def fail_sort(*_args, **_kwargs):
        raise RuntimeError("unexpected metric failure")

    monkeypatch.setattr(pd.DataFrame, "sort_values", fail_sort)

    config = _config(RollingStatsSimilarityParameters(value_columns=["value"]))
    component = RollingStatsSimilarity.from_evaluation_datasets(datasets, config)

    assert component.score.score is None
    assert component.score.notes == "unexpected metric failure"


def test_rolling_stats_similarity_group_cap_uses_seeded_selection():
    rows = []
    group_labels = ["A", "B", "C", "D", "E", "F"]
    for group_index, group in enumerate(group_labels):
        rows.extend(
            {"group": group, "time": index, "y": (group_index * 10 + index) ** 2 + np.sin(index)} for index in range(12)
        )
    frame = pd.DataFrame(rows)
    config = _config(RollingStatsSimilarityParameters(value_columns=["y"], max_groups=2), group_column="group")

    first = RollingStatsSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), config)
    second = RollingStatsSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), config)

    assert first.details == second.details
    selected_groups = [row["group"] for row in first.details["per_group"]]
    assert len(selected_groups) == 2
    assert selected_groups != group_labels[:2]
    assert first.details["group_selection"] == {
        "shared_groups": 6,
        "evaluated_groups": 2,
        "omitted_shared_groups": 4,
        "policy": "seeded_random_sample",
    }
    assert first.score.notes is not None
    assert "Evaluated 2 of 6 shared groups" in first.score.notes
