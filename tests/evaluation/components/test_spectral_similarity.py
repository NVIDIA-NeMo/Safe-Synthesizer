# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.evaluate import (
    EvaluationParameters,
    SpectralSimilarityParameters,
    TimeSeriesEvaluationParameters,
)
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters
from nemo_safe_synthesizer.evaluation.components.spectral_similarity import SpectralSimilarity
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets


def _config(
    metric: SpectralSimilarityParameters | None = None,
    *,
    group_column: str | None = None,
) -> SafeSynthesizerParameters:
    return SafeSynthesizerParameters(
        data=DataParameters(group_training_examples_by=group_column),
        time_series=TimeSeriesParameters(is_timeseries=True, timestamp_column="time"),
        evaluation=EvaluationParameters(
            time_series=TimeSeriesEvaluationParameters(
                enabled=True,
                spectral=metric or SpectralSimilarityParameters(),
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


def _seasonal(rng: np.random.Generator, points: int, period: float = 24, noise: float = 0.5) -> np.ndarray:
    """A cycle of unit amplitude at a random phase plus short-term noise."""
    phase = rng.uniform(0, 2 * np.pi)
    return np.sin(2 * np.pi * np.arange(points) / period + phase) + noise * _noise(rng, points)


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"time": np.arange(len(values)), "value": values})


def _component(training: np.ndarray, synthetic: np.ndarray, config: SafeSynthesizerParameters | None = None):
    return SpectralSimilarity.from_evaluation_datasets(
        _datasets(_frame(training), _frame(synthetic)), config or _config()
    )


def _mean_score(make_pair, points: int, trials: int = 20, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    scores = [_component(*make_pair(rng, points)).score.score for _ in range(trials)]
    finite = [score for score in scores if score is not None]
    assert len(finite) == trials
    return float(np.mean(finite))


def _seasonal_pair(make_synthetic):
    def make_pair(rng: np.random.Generator, points: int) -> tuple[np.ndarray, np.ndarray]:
        training = _seasonal(rng, points)
        return training, make_synthetic(rng, training)

    return make_pair


def _two_group_frames(
    rng: np.random.Generator, points: int, synthetic_b: np.ndarray | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = [
        pd.DataFrame({"group": group, "time": range(points), "value": _seasonal(rng, points)}) for group in "ab"
    ]
    synthetic_a = pd.DataFrame({"group": "a", "time": range(points), "value": _seasonal(rng, points)})
    values_b = _seasonal(rng, points) if synthetic_b is None else synthetic_b
    synthetic_b_frame = pd.DataFrame({"group": "b", "time": range(len(values_b)), "value": values_b})
    return pd.concat(training, ignore_index=True), pd.concat([synthetic_a, synthetic_b_frame], ignore_index=True)


def test_spectral_similarity_formula_is_noise_aware_and_pooled_over_bands():
    rng = np.random.default_rng(2)
    component = _component(_seasonal(rng, 300), _seasonal(rng, 300, noise=1.5))

    profile = component.details["profiles"][0]
    training = np.array(profile["training_bands"])
    synthetic = np.array(profile["synthetic_bands"])
    noise = np.array(profile["band_noise"])
    magnitude = np.maximum(np.maximum(training, synthetic), np.sqrt(noise))
    expected = np.sqrt(max(np.sum((training - synthetic) ** 2 - noise), 0.0) / np.sum(magnitude**2))

    assert len(training) == len(synthetic) == len(profile["band_cycle_lengths"])
    assert training.sum() == pytest.approx(1.0)
    assert 0 < expected < 1
    assert profile["error"] == pytest.approx(expected, abs=1e-12)
    assert component.score.score == pytest.approx(10 * (1 - expected), abs=0.1)


@pytest.mark.parametrize("points", [50, 200, 1000])
@pytest.mark.parametrize(
    ("make_pair", "trials"),
    [
        pytest.param(lambda rng, n: (_seasonal(rng, n), _seasonal(rng, n)), 20, id="same_process"),
        pytest.param(lambda rng, n: (rng.normal(size=n), rng.normal(size=n)), 40, id="independent_noise"),
    ],
)
def test_spectral_similarity_scores_matching_processes_high(make_pair, trials: int, points: int):
    assert _mean_score(make_pair, points, trials=trials) >= 8.0


def test_spectral_similarity_ignores_level_scale_phase_and_linear_trend():
    def make_pair(rng: np.random.Generator, points: int) -> tuple[np.ndarray, np.ndarray]:
        synthetic = 10 + 5 * _seasonal(rng, points) + 3 * np.linspace(0, 1, points)
        return _seasonal(rng, points), synthetic

    assert _mean_score(make_pair, 200) >= 9.0


@pytest.mark.parametrize(
    ("make_synthetic", "threshold"),
    [
        pytest.param(lambda rng, training: _seasonal(rng, len(training), period=12), 2.0, id="cycle_half_as_long"),
        pytest.param(lambda rng, training: rng.permutation(training), 2.0, id="shuffled"),
        pytest.param(lambda rng, training: 0.5 * _noise(rng, len(training)), 3.0, id="missing_cycle"),
    ],
)
def test_spectral_similarity_scores_wrong_or_lost_cycles_low(make_synthetic, threshold: float):
    assert _mean_score(_seasonal_pair(make_synthetic), 1000, trials=10) < threshold


@pytest.mark.parametrize("points", [1000, 1001])
def test_spectral_similarity_scores_a_lost_two_step_cycle_low_at_both_parities(points: int):
    def make_pair(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        alternating = np.where(np.arange(n) % 2 == 0, 1.0, -1.0) + 0.3 * rng.normal(size=n)
        return alternating, rng.normal(size=n)

    assert _mean_score(make_pair, points, trials=10) < 2.0


def test_spectral_similarity_penalizes_a_slightly_long_cycle_in_long_sequences():
    make_pair = _seasonal_pair(lambda rng, training: _seasonal(rng, len(training), period=26))
    assert _mean_score(make_pair, 1000, trials=10) < 8.0


@pytest.mark.parametrize(
    ("period", "points"),
    [pytest.param(24, 50, id="24_step_cycle_in_50_points"), pytest.param(8, 30, id="8_step_cycle_in_30_points")],
)
def test_spectral_similarity_scores_a_cycle_half_as_long_low_in_short_sequences(period: int, points: int):
    def make_pair(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        return _seasonal(rng, n, period=period), _seasonal(rng, n, period=period / 2)

    assert _mean_score(make_pair, points, trials=40) < 3.0


def test_spectral_similarity_compares_different_lengths_on_shared_bands():
    rng = np.random.default_rng(5)
    component = _component(_seasonal(rng, 400), _seasonal(rng, 100))

    profile = component.details["profiles"][0]
    assert profile["band_cycle_lengths"][-1] <= 100
    assert component.score.score >= 8


@pytest.mark.parametrize(
    ("points", "max_cycle_length", "longest"),
    [(1000, 128, 128), (1000, 64, 64), (60, 128, 60), (1000, 168, 168)],
)
def test_spectral_similarity_bands_are_half_an_octave_apart_up_to_the_shorter_of_length_and_max_cycle_length(
    points: int, max_cycle_length: int, longest: int
):
    rng = np.random.default_rng(6)
    config = _config(SpectralSimilarityParameters(max_cycle_length=max_cycle_length))
    profile = _component(_seasonal(rng, points), _seasonal(rng, points), config).details["profiles"][0]

    cycle_lengths = np.array(profile["band_cycle_lengths"])
    ratios = cycle_lengths[1:] / cycle_lengths[:-1]
    assert cycle_lengths[0] == pytest.approx(2.0)
    assert cycle_lengths[-1] == pytest.approx(longest)
    assert np.allclose(ratios[:-1], np.sqrt(2))
    if float(2 * np.log2(longest)).is_integer():
        assert ratios[-1] == pytest.approx(np.sqrt(2))


@pytest.mark.parametrize(
    ("replacement", "max_cycle_length", "separated"),
    [
        pytest.param(120, 128, False, id="shorter_cycle_default_limit"),
        pytest.param(120, 168, True, id="shorter_cycle_weekly_limit"),
        pytest.param(400, 168, False, id="slower_cycle_weekly_limit"),
        pytest.param(400, 336, True, id="slower_cycle_two_week_limit"),
    ],
)
def test_spectral_similarity_max_cycle_length_is_the_last_band_center(
    replacement: int, max_cycle_length: int, separated: bool
):
    """A weekly cycle in hourly data is told apart from cycles the bands below ``max_cycle_length`` resolve."""

    def hourly(rng: np.random.Generator, points: int, long_period: float) -> np.ndarray:
        return _seasonal(rng, points, period=24, noise=0.0) + _seasonal(rng, points, period=long_period)

    rng = np.random.default_rng(8)
    config = _config(SpectralSimilarityParameters(max_cycle_length=max_cycle_length))
    score = np.mean(
        [_component(hourly(rng, 2000, 168), hourly(rng, 2000, replacement), config).score.score for _ in range(5)]
    )

    assert (score < 5.0) if separated else (score >= 9.0)


def test_spectral_similarity_identical_grouped_series_are_scored_per_pair():
    training_df = pd.DataFrame(_grouped_rows([("B", 100), ("A", 0)])).sample(frac=1.0, random_state=7)
    config = _config(group_column="group")
    component = SpectralSimilarity.from_evaluation_datasets(_datasets(training_df, training_df.copy()), config)

    assert component.score.score == 10
    assert component.details["counts"]["groups"] == 2
    assert component.details["counts"]["evaluated_profiles"] == 2
    assert [row["group"] for row in component.details["per_group"]] == ["A", "B"]


def _with_non_finite_gaps(values: np.ndarray) -> np.ndarray:
    values = values.copy()
    values[[1, 4, 7, 10, 13]] = [np.inf, np.nan, np.inf, np.nan, -np.inf]
    return values


@pytest.mark.parametrize(
    ("training", "synthetic", "reason"),
    [
        pytest.param(
            np.ones(24),
            np.ones(24),
            "The training sequence is constant or indistinguishable from constant at its numeric scale.",
            id="constant",
        ),
        pytest.param(
            1e6 + 0.5 * np.arange(24),
            1e6 + 0.5 * np.arange(24),
            "The training sequence has no variation beyond a straight-line trend.",
            id="trend_only",
        ),
        pytest.param(
            np.sin(np.arange(12)),
            np.sin(np.arange(200)),
            "The training sequence needs at least 16 finite observations.",
            id="short",
        ),
        pytest.param(
            _with_non_finite_gaps(np.sin(np.arange(20))),
            _with_non_finite_gaps(np.sin(np.arange(20))),
            "The training sequence needs at least 16 finite observations.",
            id="short_after_non_finite_values",
        ),
    ],
)
def test_spectral_similarity_skips_unusable_training_sequences(
    training: np.ndarray, synthetic: np.ndarray, reason: str
):
    component = _component(training, synthetic)

    assert component.score.score is None
    assert component.score.notes == "No usable group/column spectral profiles."
    assert [item["reason"] for item in component.details["skipped"]] == [reason]


@pytest.mark.parametrize(
    ("synthetic_b", "reason"),
    [
        pytest.param(
            np.full(200, 100.0),
            "The synthetic sequence is constant or indistinguishable from constant at its numeric scale.",
            id="constant",
        ),
        pytest.param(
            100 + 0.5 * np.arange(200),
            "The synthetic sequence has no variation beyond a straight-line trend.",
            id="trend_only",
        ),
        pytest.param(
            np.where(np.arange(200) < 10, np.sin(np.arange(200)), np.nan),
            "The synthetic sequence has fewer than 16 finite observations.",
            id="mostly_missing",
        ),
        pytest.param(np.array([1.0]), "The synthetic sequence has fewer than 16 finite observations.", id="one_row"),
    ],
)
def test_spectral_similarity_scores_unusable_synthetic_sequences_0(synthetic_b: np.ndarray, reason: str):
    config = _config(group_column="group")
    complete = SpectralSimilarity.from_evaluation_datasets(
        _datasets(*_two_group_frames(np.random.default_rng(12), 200)), config
    )

    component = SpectralSimilarity.from_evaluation_datasets(
        _datasets(*_two_group_frames(np.random.default_rng(12), 200, synthetic_b)), config
    )

    pairs = {item["group"]: item for item in component.details["profiles"]}
    assert set(pairs) == {"a", "b"}
    assert component.details["skipped"] == []
    assert pairs["b"]["similarity"] == 0.0
    assert pairs["b"]["reason"] == reason
    assert pairs["b"]["synthetic_bands"] == [None] * len(pairs["b"]["band_cycle_lengths"])
    assert pairs["b"]["band_cycle_lengths"][0] == pytest.approx(2.0)
    assert component.score.score is not None
    assert component.score.score == pytest.approx(5 * pairs["a"]["similarity"], abs=0.06)
    assert complete.score.score is not None
    assert component.score.score < complete.score.score - 4


def test_spectral_similarity_handles_aligned_non_finite_gaps():
    values = 3 * np.sin(np.arange(40)) + np.arange(40) % 5
    training = values.copy()
    training[[3, 17]] = np.inf
    synthetic = values.copy()
    synthetic[[3, 17]] = -np.inf

    assert _component(training, synthetic).score.score == 10


def test_spectral_similarity_keeps_missing_values_in_place():
    rng = np.random.default_rng(8)
    training = _seasonal(rng, 1000, noise=0.2)
    synthetic = _seasonal(rng, 1000, noise=0.2)
    synthetic[rng.random(1000) < 0.2] = np.nan

    in_place = _component(training, synthetic).score.score
    spliced = _component(training, synthetic[np.isfinite(synthetic)]).score.score

    assert in_place is not None and spliced is not None
    assert in_place >= 8
    assert spliced < in_place - 2


@pytest.mark.parametrize("dtype", [pd.Int64Dtype(), pd.Float64Dtype()], ids=["Int64", "Float64"])
def test_spectral_similarity_treats_nullable_missing_values_as_gaps(dtype: pd.Int64Dtype | pd.Float64Dtype):
    values = np.round(10 * np.sin(2 * np.pi * np.arange(200) / 24))
    missing = [5, 60, 61, 150]
    float_values = values.copy()
    float_values[missing] = np.nan
    nullable_values = pd.Series(values).astype(dtype)
    nullable_values.iloc[missing] = pd.NA
    nullable_frame = _frame(values).assign(value=nullable_values)

    nullable = SpectralSimilarity.from_evaluation_datasets(_datasets(_frame(values), nullable_frame), _config())

    np.testing.assert_array_equal(SpectralSimilarity._prepare_values(nullable_values), float_values)
    assert nullable.score.score == _component(values, float_values).score.score


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(1e-13 * (3 * np.sin(np.arange(24)) + 5), id="tiny"),
        pytest.param(1e12 + np.round(3 * np.sin(np.arange(24))), id="large_offset"),
    ],
)
def test_spectral_similarity_constant_detection_is_scale_aware(values: np.ndarray):
    assert _component(values, values.copy()).score.score == 10


def test_spectral_similarity_without_config_runs_with_component_defaults():
    frame = pd.DataFrame({"value": 3 * np.sin(np.arange(24)) + np.arange(24) % 5})

    component = SpectralSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()))

    assert component.score.score == 10
    assert component.details["evaluation_mode"] == "global"


def test_spectral_similarity_isolates_unexpected_metric_failures(monkeypatch):
    frame = _frame(np.sin(np.arange(24)))
    datasets = _datasets(frame, frame.copy())

    def fail_sort(*_args, **_kwargs):
        raise RuntimeError("unexpected metric failure")

    monkeypatch.setattr(pd.DataFrame, "sort_values", fail_sort)

    config = _config(SpectralSimilarityParameters(value_columns=["value"]))
    component = SpectralSimilarity.from_evaluation_datasets(datasets, config)

    assert component.score.score is None
    assert component.score.notes == "unexpected metric failure"


def test_spectral_similarity_group_cap_uses_seeded_selection():
    rows = []
    group_labels = ["A", "B", "C", "D", "E", "F"]
    for group_index, group in enumerate(group_labels):
        rows.extend(
            {"group": group, "time": index, "y": (group_index * 10 + index) ** 2 + np.sin(index)} for index in range(20)
        )
    frame = pd.DataFrame(rows)
    config = _config(SpectralSimilarityParameters(value_columns=["y"], max_groups=2), group_column="group")

    first = SpectralSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), config)
    second = SpectralSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), config)

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


@pytest.mark.parametrize("collapsed_values", [[0], [0, 1, 2]])
def test_spectral_similarity_scores_collapsed_synthetic_integer_column(collapsed_values: list[int]):
    rng = np.random.default_rng(7)
    points = 200
    training_df = pd.DataFrame(
        {
            "time": range(points),
            "value": _seasonal(rng, points),
            "count": np.round(10 + 80 * rng.random(points)).astype(int),
        }
    )
    synthetic_df = training_df.copy()
    synthetic_df["count"] = np.resize(collapsed_values, points)

    component = SpectralSimilarity.from_evaluation_datasets(_datasets(training_df, synthetic_df), _config())

    profiled = {item["column"]: item for item in component.details["profiles"]}
    assert set(profiled) == {"count", "value"}
    assert component.details["excluded_columns"] == []
    assert component.score.score is not None
    assert component.score.score < 10
    if len(collapsed_values) == 1:
        assert profiled["count"]["similarity"] == 0


def test_spectral_similarity_names_left_out_columns_when_none_remain():
    training_df = pd.DataFrame({"time": range(64), "value": range(64), "count": range(64)})
    synthetic_df = training_df.copy()
    synthetic_df["value"] = [f"v{index}" for index in range(64)]
    synthetic_df["count"] = [f"n{index}" for index in range(64)]

    component = SpectralSimilarity.from_evaluation_datasets(_datasets(training_df, synthetic_df), _config())

    assert component.score.score is None
    assert component.details["excluded_columns"] == ["count", "value"]
    assert component.score.notes == (
        "No shared numeric value columns. Left out 2 numeric training columns that synthetic data lacks "
        "or stores as non-numeric values: count, value."
    )
