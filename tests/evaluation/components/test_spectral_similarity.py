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
from nemo_safe_synthesizer.defaults import PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.evaluation.components.spectral_similarity import SpectralSimilarity
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets
from nemo_safe_synthesizer.training.timeseries_preprocessing import process_timeseries_data


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


def test_spectral_similarity_bands_are_half_an_octave_apart_from_two_steps_to_the_longest_cycle():
    rng = np.random.default_rng(3)
    profile = _component(_seasonal(rng, 300), _seasonal(rng, 300)).details["profiles"][0]

    cycle_lengths = np.array(profile["band_cycle_lengths"])

    assert cycle_lengths[0] == pytest.approx(2.0)
    assert cycle_lengths[-1] == pytest.approx(128.0)
    assert np.allclose(cycle_lengths[1:] / cycle_lengths[:-1], np.sqrt(2))


@pytest.mark.parametrize("points", [50, 200, 1000])
def test_spectral_similarity_scores_same_process_high(points: int):
    assert _mean_score(lambda rng, n: (_seasonal(rng, n), _seasonal(rng, n)), points) >= 8.0


@pytest.mark.parametrize("points", [50, 200, 1000])
def test_spectral_similarity_scores_independent_noise_high(points: int):
    assert _mean_score(lambda rng, n: (rng.normal(size=n), rng.normal(size=n)), points, trials=40) >= 8.0


def test_spectral_similarity_ignores_level_scale_phase_and_linear_trend():
    def make_pair(rng: np.random.Generator, points: int) -> tuple[np.ndarray, np.ndarray]:
        synthetic = 10 + 5 * _seasonal(rng, points) + 3 * np.linspace(0, 1, points)
        return _seasonal(rng, points), synthetic

    assert _mean_score(make_pair, 200) >= 9.0


@pytest.mark.parametrize(
    "make_synthetic",
    [
        pytest.param(lambda rng, training: _seasonal(rng, len(training), period=12), id="cycle_half_as_long"),
        pytest.param(lambda rng, training: rng.permutation(training), id="shuffled"),
    ],
)
def test_spectral_similarity_scores_wrong_or_lost_cycles_in_lost_tier(make_synthetic):
    assert _mean_score(_seasonal_pair(make_synthetic), 1000, trials=10) < 2.0


def test_spectral_similarity_scores_a_missing_cycle_low():
    assert _mean_score(_seasonal_pair(lambda rng, training: 0.5 * _noise(rng, len(training))), 1000, trials=10) < 3.0


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


@pytest.mark.parametrize(("points", "max_cycle_length", "longest"), [(1000, 128, 128), (1000, 64, 64), (60, 128, 60)])
def test_spectral_similarity_distinguishes_cycles_up_to_the_shorter_of_length_and_max_cycle_length(
    points: int, max_cycle_length: int, longest: int
):
    rng = np.random.default_rng(6)
    config = _config(SpectralSimilarityParameters(max_cycle_length=max_cycle_length))
    profile = _component(_seasonal(rng, points), _seasonal(rng, points), config).details["profiles"][0]

    assert longest / np.sqrt(2) < profile["band_cycle_lengths"][-1] <= longest


def test_spectral_similarity_identical_grouped_series_are_scored_per_pair():
    training_df = pd.DataFrame(_grouped_rows([("B", 100), ("A", 0)])).sample(frac=1.0, random_state=7)
    config = _config(group_column="group")
    component = SpectralSimilarity.from_evaluation_datasets(_datasets(training_df, training_df.copy()), config)

    assert component.score.score == 10
    assert component.details["counts"]["groups"] == 2
    assert component.details["counts"]["evaluated_profiles"] == 2
    assert [row["group"] for row in component.details["per_group"]] == ["A", "B"]


def test_spectral_similarity_keeps_real_column_named_like_pseudo_group():
    training_df = pd.DataFrame(_grouped_rows([("B", 100), ("A", 0)])).rename(columns={"group": PSEUDO_GROUP_COLUMN})
    config = _config(group_column=PSEUDO_GROUP_COLUMN)
    component = SpectralSimilarity.from_evaluation_datasets(_datasets(training_df, training_df.copy()), config)

    assert component.details["counts"]["groups"] == 2


def test_spectral_similarity_treats_inherited_pseudo_group_as_global_sequence():
    training_df = pd.DataFrame({"value": 3 * np.sin(np.arange(24)) + np.arange(24) % 5})
    config = SafeSynthesizerParameters.from_params(
        is_timeseries=True,
        timestamp_interval_seconds=1,
        rope_scaling_factor=1,
    )
    processed_df, config, _ = process_timeseries_data(training_df.copy(), config)
    synthetic_df = processed_df.drop(columns=PSEUDO_GROUP_COLUMN).sample(frac=1.0, random_state=7)

    component = SpectralSimilarity.from_evaluation_datasets(_datasets(training_df, synthetic_df), config)

    assert component.score.score == 10
    assert component.details["evaluation_mode"] == "global"
    assert component.details["group_column"] is None
    assert component.details["timestamp_column"] == "elapsed_seconds"


def test_spectral_similarity_excludes_inherited_pseudo_group_from_value_columns():
    frame = pd.DataFrame(
        {
            PSEUDO_GROUP_COLUMN: [0] * 24,
            "time": range(24),
            "value": 3 * np.sin(np.arange(24)) + np.arange(24) % 5,
        }
    )
    config = _config()
    config.data.group_training_examples_by = PSEUDO_GROUP_COLUMN

    component = SpectralSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), config)

    assert component.score.score == 10
    assert [row["column"] for row in component.details["per_column"]] == ["value"]


def test_spectral_similarity_reports_missing_explicit_group_column():
    frame = _frame(np.sin(np.arange(24)))
    config = _config(group_column="missing_group")

    component = SpectralSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), config)

    assert component.score.score is None
    assert component.score.notes == "Configured group column 'missing_group' is missing from a dataset."


def test_spectral_similarity_constant_training_series_is_unavailable_instead_of_perfect():
    frame = _frame(np.ones(24))
    component = SpectralSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), _config())

    assert component.score.score is None
    assert component.score.notes == "No usable group/column spectral profiles."
    assert "training sequence is constant" in component.details["skipped"][0]["reason"].lower()


def test_spectral_similarity_trend_only_training_series_is_unavailable():
    frame = _frame(1e6 + 0.5 * np.arange(24))
    component = SpectralSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), _config())

    assert component.score.score is None
    assert "straight-line trend" in component.details["skipped"][0]["reason"]


@pytest.mark.parametrize(
    "synthetic_values",
    [pytest.param(np.full(30, 100.0), id="constant"), pytest.param(100 + 0.5 * np.arange(30), id="trend_only")],
)
def test_spectral_similarity_scores_synthetic_collapse_as_failure(synthetic_values: np.ndarray):
    training_df = pd.DataFrame(_grouped_rows([("A", 0), ("B", 100)]))
    synthetic_df = training_df.copy()
    synthetic_df.loc[synthetic_df["group"] == "B", "value"] = synthetic_values
    config = _config(group_column="group")

    component = SpectralSimilarity.from_evaluation_datasets(_datasets(training_df, synthetic_df), config)

    assert component.score.score == 5
    collapsed = next(item for item in component.details["profiles"] if item["group"] == "B")
    assert collapsed["similarity"] == 0
    assert collapsed["synthetic_bands"] == [None] * len(collapsed["band_cycle_lengths"])
    assert "synthetic sequence" in collapsed["reason"].lower()
    assert component.details["counts"]["skipped"] == 0


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


def test_spectral_similarity_revalidates_usable_length_with_non_finite_values():
    values = np.sin(np.arange(20))
    values[[1, 4, 7, 10, 13]] = [np.inf, np.nan, np.inf, np.nan, -np.inf]

    component = _component(values, values.copy())

    assert component.score.score is None
    assert "at least 16 finite observations" in component.details["skipped"][0]["reason"]


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(1e-13 * (3 * np.sin(np.arange(24)) + 5), id="tiny"),
        pytest.param(1e12 + np.round(3 * np.sin(np.arange(24))), id="large_offset"),
    ],
)
def test_spectral_similarity_constant_detection_is_scale_aware(values: np.ndarray):
    assert _component(values, values.copy()).score.score == 10


def test_spectral_similarity_rejects_invalid_explicit_value_columns():
    frame = pd.DataFrame({"time": range(24), "value": np.sin(np.arange(24)), "label": list("abcdefghijklmnopqrstuvwx")})
    config = _config(SpectralSimilarityParameters(value_columns=["missing", "label"]))

    component = SpectralSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), config)

    assert component.score.score is None
    assert component.score.notes is not None
    assert "'missing' is missing from training and synthetic data" in component.score.notes
    assert "'label' is not numeric in training data" in component.score.notes


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
