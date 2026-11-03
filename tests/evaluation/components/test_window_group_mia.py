# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.evaluate import (
    EvaluationParameters,
    TimeSeriesEvaluationParameters,
    WindowGroupMIAParameters,
)
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters
from nemo_safe_synthesizer.evaluation.components.window_group_mia import WindowGroupMIA
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets

GROUPS = 10


def _config(
    metric: WindowGroupMIAParameters | None = None,
    *,
    group_column: str | None = "entity",
) -> SafeSynthesizerParameters:
    return SafeSynthesizerParameters(
        data=DataParameters(group_training_examples_by=group_column),
        time_series=TimeSeriesParameters(is_timeseries=True, timestamp_column="time"),
        evaluation=EvaluationParameters(
            time_series=TimeSeriesEvaluationParameters(
                enabled=True,
                window_group_mia=metric or WindowGroupMIAParameters(),
            )
        ),
    )


def _ar(rng: np.random.Generator, phi: float, points: int) -> np.ndarray:
    """Stationary AR(1) with unit variance."""
    values = np.empty(points)
    values[0] = rng.normal()
    for index in range(1, points):
        values[index] = phi * values[index - 1] + np.sqrt(1 - phi**2) * rng.normal()
    return values


def _entity(rng: np.random.Generator, points: int) -> np.ndarray:
    """A 24-step cycle plus short-term noise, and a persistent series, each with its own level."""
    t = np.arange(points)
    cycle = np.sin(2 * np.pi * t / 24 + rng.uniform(0, 2 * np.pi)) + 0.5 * _ar(rng, 0.5, points)
    return np.column_stack([cycle, _ar(rng, 0.9, points)]) + 0.5 * rng.normal(size=2)


def _entities(rng: np.random.Generator, points: int, count: int = GROUPS) -> list[np.ndarray]:
    return [_entity(rng, points) for _ in range(count)]


def _frame(groups: list[np.ndarray], prefix: str) -> pd.DataFrame:
    columns = ["a", "b"] if groups[0].shape[1] == 2 else [f"c{index}" for index in range(groups[0].shape[1])]
    parts = [
        pd.DataFrame(
            {"entity": f"{prefix}{index}", "time": np.arange(len(values)), **dict(zip(columns, values.T, strict=True))}
        )
        for index, values in enumerate(groups)
    ]
    return pd.concat(parts, ignore_index=True)


def _datasets(training: pd.DataFrame, synthetic: pd.DataFrame, test: pd.DataFrame | None) -> EvaluationDatasets:
    return EvaluationDatasets.from_dataframes(training, synthetic, test=test, enable_sampling=False)


def _evaluate(
    training: list[np.ndarray],
    synthetic: list[np.ndarray],
    holdout: list[np.ndarray] | None,
    config: SafeSynthesizerParameters | None = None,
) -> WindowGroupMIA:
    datasets = _datasets(
        _frame(training, "train-"),
        _frame(synthetic, "train-"),
        None if holdout is None else _frame(holdout, "holdout-"),
    )
    return WindowGroupMIA.from_evaluation_datasets(datasets, config or _config())


def _mean_score(make_synthetic, points: int, trials: int, seed: int = 7, holdout_groups: int = GROUPS) -> float:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(trials):
        training, holdout = _entities(rng, points), _entities(rng, points, count=holdout_groups)
        score = _evaluate(training, make_synthetic(rng, points, training), holdout).score.score
        assert score is not None
        scores.append(score)
    return float(np.mean(scores))


def _noisy_copy(rng: np.random.Generator, training: list[np.ndarray], noise: float) -> list[np.ndarray]:
    """Training plus noise of ``noise`` times each column's standard deviation over all training groups."""
    scale = np.nanstd(np.vstack(training), axis=0)
    return [values + noise * scale * rng.normal(size=values.shape) for values in training]


def _one_copied(rng: np.random.Generator, points: int, training: list[np.ndarray]) -> list[np.ndarray]:
    """Fresh groups, except the first training group copied whole."""
    synthetic = _entities(rng, points)
    synthetic[0] = training[0].copy()
    return synthetic


def _with_missing(rng: np.random.Generator, groups: list[np.ndarray], share: float = 0.2) -> list[np.ndarray]:
    out = []
    for values in groups:
        values = values.copy()
        values[rng.random(values.shape) < share] = np.nan
        out.append(values)
    return out


def test_window_group_mia_score_removes_chance_from_the_privacy_loss():
    rng = np.random.default_rng(3)
    training, holdout = _entities(rng, 300), _entities(rng, 300)
    synthetic = _entities(rng, 300)
    synthetic[0] = training[0].copy()

    component = _evaluate(training, synthetic, holdout)

    attack = component.details["attack"]
    training_rate = (attack["flagged_training_windows"] + 0.5) / (component.details["counts"]["training_windows"] + 1)
    holdout_rate = (attack["flagged_holdout_windows"] + 0.5) / (component.details["counts"]["holdout_windows"] + 1)
    assert attack["privacy_loss"] == pytest.approx(np.log(training_rate / holdout_rate))
    noise = attack["binomial_noise"] + attack["holdout_offset_noise"]
    expected_loss = np.sqrt(max(attack["privacy_loss"] ** 2 - noise, 0.0))
    assert attack["privacy_loss_beyond_chance"] == pytest.approx(expected_loss)
    excess_exact = component.details["exact_copy_rate"] - attack["holdout_exact_match_rate"]
    expected_exact_loss = np.log1p(excess_exact * (1 - holdout_rate) / holdout_rate)
    assert attack["exact_copy_privacy_loss"] == pytest.approx(expected_exact_loss)
    assert attack["scored_privacy_loss"] == pytest.approx(max(expected_loss, expected_exact_loss))
    assert component.score.score == pytest.approx(10 * np.exp(-attack["scored_privacy_loss"]), abs=0.1)
    assert component.details["evaluation_mode"] == "full"
    assert component.details["window_size"] == 10
    assert component.details["stride"] == 5


@pytest.mark.parametrize(
    ("points", "holdout_groups", "trials", "seed"), [(200, GROUPS, 8, 7), (1000, GROUPS, 4, 7), (200, 2, 6, 24)]
)
def test_window_group_mia_scores_fresh_synthetic_groups_high(points: int, holdout_groups: int, trials: int, seed: int):
    fresh = _mean_score(lambda rng, n, training: _entities(rng, n), points, trials, seed, holdout_groups)

    assert fresh >= 8.0


@pytest.mark.parametrize(
    "make_synthetic",
    [
        pytest.param(lambda rng, training: [values.copy() for values in training], id="exact_copy"),
        pytest.param(lambda rng, training: [values[3:].copy() for values in training], id="copy_shifted_off_grid"),
        pytest.param(_with_missing, id="copy_with_missing_values"),
    ],
)
def test_window_group_mia_scores_copied_training_data_near_zero(make_synthetic):
    rng = np.random.default_rng(5)
    training, holdout = _entities(rng, 200), _entities(rng, 200)

    component = _evaluate(training, make_synthetic(rng, training), holdout)

    assert component.score.score is not None
    assert component.score.score <= 1.0


@pytest.mark.parametrize(("points", "trials"), [(200, 4), (1000, 1)])
def test_window_group_mia_scores_one_copied_group_of_ten_as_quoted_in_the_guide(points: int, trials: int):
    # The user guide's tiers quote about 3.5 for one whole group of 10 copied, and 3.4 at 1,000 points.
    assert 3.0 <= _mean_score(_one_copied, points, trials, seed=26) <= 4.0


@pytest.mark.parametrize(("holdout_groups", "trials"), [(GROUPS, 4), (2, 3)])
def test_window_group_mia_scores_a_noisy_copy_low_through_the_chance_correction(holdout_groups: int, trials: int):
    rng = np.random.default_rng(22)
    scores = []
    for _ in range(trials):
        training, holdout = _entities(rng, 200), _entities(rng, 200, count=holdout_groups)
        component = _evaluate(training, _noisy_copy(rng, training, 0.3), holdout)
        attack = component.details["attack"]
        assert component.details["exact_copy_rate"] == 0.0
        assert attack["privacy_loss_beyond_chance"] > attack["exact_copy_privacy_loss"]
        assert component.score.score is not None
        scores.append(component.score.score)

    assert np.mean(scores) <= 2.0


def test_window_group_mia_scores_synthetic_copies_of_the_holdout_groups_high():
    rng = np.random.default_rng(23)
    training, holdout = _entities(rng, 200), _entities(rng, 200)

    component = _evaluate(training, [values.copy() for values in holdout], holdout)

    assert component.score.score is not None
    assert component.score.score >= 9.5


def test_window_group_mia_mostly_misses_an_exact_copy_of_one_column_in_six():
    # Known blind spot, documented under Limitations in the user guide: the five fresh columns dominate
    # the window distance and no whole window matches, so the exact-copy floor never applies.
    rng = np.random.default_rng(25)

    def wide(count: int = GROUPS) -> list[np.ndarray]:
        return [np.hstack([_entity(rng, 200) for _ in range(3)]) for _ in range(count)]

    scores = []
    for _ in range(6):
        training, holdout, synthetic = wide(), wide(), wide()
        for copied, source in zip(synthetic, training, strict=True):
            copied[:, 0] = source[:, 0]
        component = _evaluate(training, synthetic, holdout)
        assert component.details["exact_copy_rate"] == 0.0
        assert component.score.score is not None
        scores.append(component.score.score)

    assert np.mean(scores) > 8.0


def test_window_group_mia_exact_copies_are_not_forgiven_by_a_small_holdout():
    rng = np.random.default_rng(19)
    training = _entities(rng, 200, count=16)
    synthetic = _entities(rng, 200, count=16)
    synthetic[3] = training[3].copy()

    component = _evaluate(training, synthetic, _entities(rng, 200, count=2))

    attack = component.details["attack"]
    assert component.details["exact_copy_rate"] == pytest.approx(1 / 16)
    assert attack["exact_copy_privacy_loss"] > attack["privacy_loss_beyond_chance"]
    assert component.score.score is not None
    assert component.score.score < 6.0


def test_window_group_mia_exact_matches_also_found_in_the_holdout_do_not_lower_the_score():
    rng = np.random.default_rng(20)

    def stuck(groups: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        for values in groups:
            values = values.copy()
            values[40:80] = 0.0
            out.append(values)
        return out

    training, holdout = stuck(_entities(rng, 200)), stuck(_entities(rng, 200))

    component = _evaluate(training, stuck(_entities(rng, 200)), holdout)

    attack = component.details["attack"]
    assert component.details["exact_copy_rate"] > 0
    assert attack["holdout_exact_match_rate"] > 0
    assert attack["exact_copy_privacy_loss"] == pytest.approx(0.0, abs=0.05)
    assert component.score.score is not None
    assert component.score.score >= 8.0


def test_window_group_mia_does_not_penalize_constant_synthetic_data():
    rng = np.random.default_rng(6)
    training, holdout = _entities(rng, 200), _entities(rng, 200)
    level = np.mean(np.vstack(training), axis=0)

    component = _evaluate(training, [np.tile(level, (200, 1)) for _ in training], holdout)

    assert component.score.score is not None
    assert component.score.score >= 8.0


def test_window_group_mia_without_group_column_reports_copy_diagnostics_only():
    rng = np.random.default_rng(8)
    training = _entities(rng, 300, count=1)

    component = _evaluate(training, [training[0].copy()], _entities(rng, 300, count=1), _config(group_column=None))

    assert component.score.score is None
    assert component.score.notes is not None
    assert "needs a group column" in component.score.notes
    assert component.details["evaluation_mode"] == "copy_diagnostics_only"
    assert component.details["exact_copy_rate"] == 1.0


def test_window_group_mia_without_holdout_reports_copy_diagnostics_only():
    rng = np.random.default_rng(10)
    training = _entities(rng, 100)

    component = _evaluate(training, [values.copy() for values in training], None)

    assert component.score.score is None
    assert component.score.notes == "No holdout data is available. Reporting copy diagnostics only."
    assert component.details["exact_copy_rate"] == 1.0
    assert [row["group"] for row in component.details["per_group"]][:2] == ["train-0", "train-1"]
    assert [figure["title"] for figure in component.jinja_context["figures"]] == [
        "Distance to synthetic data",
        "Closest match",
    ]


def test_window_group_mia_needs_twenty_holdout_windows_for_a_score():
    rng = np.random.default_rng(11)
    training = _entities(rng, 100)

    component = _evaluate(training, _entities(rng, 100), _entities(rng, 40, count=2))

    assert component.score.score is None
    assert component.score.notes is not None
    assert "at least 20 training and holdout windows; found 190 training and 14 holdout" in component.score.notes
    assert component.details["evaluation_mode"] == "copy_diagnostics_only"


def test_window_group_mia_needs_two_holdout_groups_for_a_score():
    rng = np.random.default_rng(12)
    training = _entities(rng, 200)
    synthetic = _entities(rng, 200)
    synthetic[0] = training[0].copy()

    component = _evaluate(training, synthetic, _entities(rng, 200, count=1))

    assert component.score.score is None
    assert component.details["evaluation_mode"] == "copy_diagnostics_only"
    assert component.details["counts"]["holdout_groups_with_windows"] == 1
    assert component.score.notes == (
        "The score needs at least 2 holdout groups; found 1. With 1 holdout group, chance can hide copying. "
        "Reporting copy diagnostics only."
    )
    assert component.details["exact_copy_rate"] == pytest.approx(1 / GROUPS)
    assert "attack" not in component.details


def test_window_group_mia_window_cap_is_balanced_and_reproducible():
    rng = np.random.default_rng(13)
    training, synthetic, holdout = _entities(rng, 300), _entities(rng, 300), _entities(rng, 300)
    config = _config(WindowGroupMIAParameters(max_windows=100))

    first = _evaluate(training, synthetic, holdout, config)
    second = _evaluate(training, synthetic, holdout, config)

    assert first.details == second.details
    counts = first.details["counts"]
    assert counts["training_windows"] == counts["holdout_windows"] == 100
    assert counts["training_windows_total"] == counts["holdout_windows_total"] == 590
    assert counts["synthetic_windows"] == counts["synthetic_windows_total"] == 2910
    assert {row["windows"] for row in first.details["per_group"]} == {10}


def test_window_group_mia_column_cap_uses_a_reproducible_sample():
    rng = np.random.default_rng(14)
    training, synthetic, holdout = (_frame(_entities(rng, 60), prefix) for prefix in ("t", "t", "h"))
    for frame in (training, synthetic, holdout):
        frame["c"] = frame["a"] * 2
    config = _config(WindowGroupMIAParameters(max_columns=2))

    first = WindowGroupMIA.from_evaluation_datasets(_datasets(training, synthetic, holdout), config)
    second = WindowGroupMIA.from_evaluation_datasets(_datasets(training, synthetic, holdout), config)

    assert len(first.details["columns"]) == 2
    assert first.details["columns"] == second.details["columns"]


@pytest.mark.parametrize(("with_holdout", "mode"), [(True, "full"), (False, "copy_diagnostics_only")])
def test_window_group_mia_names_numeric_columns_that_synthetic_data_stores_as_text(with_holdout: bool, mode: str):
    rng = np.random.default_rng(21)
    training, synthetic, holdout = (_frame(_entities(rng, 100), prefix) for prefix in ("t", "t", "h"))
    synthetic["b"] = synthetic["b"].map("{:.3f}".format)

    component = WindowGroupMIA.from_evaluation_datasets(
        _datasets(training, synthetic, holdout if with_holdout else None), _config()
    )

    assert component.details["evaluation_mode"] == mode
    assert component.details["columns"] == ["a"]
    assert component.details["excluded_columns"] == ["b"]
    assert component.score.notes is not None
    assert component.score.notes.endswith(
        "Left out 1 numeric training columns that synthetic data lacks or stores as non-numeric values: b."
    )


def test_window_group_mia_without_config_runs_with_component_defaults():
    rng = np.random.default_rng(17)
    frame = _frame(_entities(rng, 60, count=1), "t").drop(columns="entity")

    component = WindowGroupMIA.from_evaluation_datasets(_datasets(frame, frame.copy(), None))

    assert component.details["window_size"] == 10
    assert component.details["evaluation_mode"] == "copy_diagnostics_only"
    assert "jinja_context" not in component.details
    assert component.jinja_context["details"] is component.details


def test_window_group_mia_isolates_unexpected_metric_failures(monkeypatch):
    rng = np.random.default_rng(18)
    frame = _frame(_entities(rng, 30, count=2), "t")
    datasets = _datasets(frame, frame.copy(), None)

    def fail_sort(*_args, **_kwargs):
        raise RuntimeError("unexpected metric failure")

    monkeypatch.setattr(pd.DataFrame, "sort_values", fail_sort)

    component = WindowGroupMIA.from_evaluation_datasets(datasets, _config())

    assert component.score.score is None
    assert component.score.notes == "unexpected metric failure"
