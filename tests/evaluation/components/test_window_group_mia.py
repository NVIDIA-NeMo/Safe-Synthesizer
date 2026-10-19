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
from nemo_safe_synthesizer.defaults import PSEUDO_GROUP_COLUMN
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
    parts = [
        pd.DataFrame(
            {"entity": f"{prefix}{index}", "time": np.arange(len(values)), "a": values[:, 0], "b": values[:, 1]}
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


def _mean_score(make_synthetic, points: int, trials: int, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(trials):
        training, holdout = _entities(rng, points), _entities(rng, points)
        score = _evaluate(training, make_synthetic(rng, points, training), holdout).score.score
        assert score is not None
        scores.append(score)
    return float(np.mean(scores))


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
    assert component.score.score == pytest.approx(10 * np.exp(-expected_loss), abs=0.1)
    assert component.details["evaluation_mode"] == "full"
    assert component.details["window_size"] == 10
    assert component.details["stride"] == 5


@pytest.mark.parametrize("points", [200, 1000])
def test_window_group_mia_scores_fresh_synthetic_groups_high(points: int):
    assert _mean_score(lambda rng, n, training: _entities(rng, n), points, trials=8) >= 8.0


@pytest.mark.parametrize(
    "make_synthetic",
    [
        pytest.param(lambda rng, training: [values.copy() for values in training], id="exact_copy"),
        pytest.param(lambda rng, training: [values[3:].copy() for values in training], id="copy_shifted_off_grid"),
        pytest.param(lambda rng, training: _with_missing(rng, training), id="copy_with_missing_values"),
    ],
)
def test_window_group_mia_scores_copied_training_data_near_zero(make_synthetic):
    rng = np.random.default_rng(5)
    training, holdout = _entities(rng, 200), _entities(rng, 200)

    component = _evaluate(training, make_synthetic(rng, training), holdout)

    assert component.score.score is not None
    assert component.score.score <= 1.0


def test_window_group_mia_scores_one_copied_group_in_ten_below_five_with_long_sequences():
    def one_copied(rng: np.random.Generator, points: int, training: list[np.ndarray]) -> list[np.ndarray]:
        synthetic = _entities(rng, points)
        synthetic[0] = training[0].copy()
        return synthetic

    assert _mean_score(one_copied, 1000, trials=4) < 5.0


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


def test_window_group_mia_treats_inherited_pseudo_group_as_ungrouped():
    rng = np.random.default_rng(9)
    training = _entities(rng, 100, count=1)
    config = _config(group_column=None)
    config.data.group_training_examples_by = PSEUDO_GROUP_COLUMN

    component = _evaluate(training, _entities(rng, 100, count=1), None, config)

    assert component.details["group_column"] is None
    assert component.details["evaluation_mode"] == "copy_diagnostics_only"


def test_window_group_mia_without_holdout_reports_copy_diagnostics_only():
    rng = np.random.default_rng(10)
    training = _entities(rng, 100)

    component = _evaluate(training, [values.copy() for values in training], None)

    assert component.score.score is None
    assert component.score.notes == "No holdout data is available. Reporting copy diagnostics only."
    assert component.details["exact_copy_rate"] == 1.0
    assert [row["group"] for row in component.details["per_group"]][:2] == ["train-0", "train-1"]


def test_window_group_mia_needs_twenty_holdout_windows_for_a_score():
    rng = np.random.default_rng(11)
    training = _entities(rng, 100)

    component = _evaluate(training, _entities(rng, 100), _entities(rng, 40, count=2))

    assert component.score.score is None
    assert component.score.notes is not None
    assert "at least 20 training and holdout windows; found 190 training and 14 holdout" in component.score.notes
    assert component.details["evaluation_mode"] == "copy_diagnostics_only"


def test_window_group_mia_notes_a_single_holdout_group():
    rng = np.random.default_rng(12)
    training = _entities(rng, 200)

    component = _evaluate(training, _entities(rng, 200), _entities(rng, 200, count=1))

    assert component.score.score is not None
    assert component.details["counts"]["holdout_groups_with_windows"] == 1
    assert component.score.notes is not None
    assert "Only one holdout group" in component.score.notes


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


def test_window_group_mia_rejects_invalid_explicit_value_columns():
    rng = np.random.default_rng(15)
    training = _frame(_entities(rng, 30, count=2), "t").assign(label="x")
    config = _config(WindowGroupMIAParameters(value_columns=["missing", "label", "entity"]))

    component = WindowGroupMIA.from_evaluation_datasets(_datasets(training, training.copy(), None), config)

    assert component.score.score is None
    assert component.score.notes is not None
    assert "'missing' is missing from training and synthetic data" in component.score.notes
    assert "'label' is not numeric in training data" in component.score.notes
    assert "'entity' is used for timestamp or sequence grouping" in component.score.notes


def test_window_group_mia_reports_missing_explicit_group_column():
    rng = np.random.default_rng(16)
    frame = _frame(_entities(rng, 30, count=2), "t")

    component = WindowGroupMIA.from_evaluation_datasets(
        _datasets(frame, frame.copy(), None), _config(group_column="missing_group")
    )

    assert component.score.score is None
    assert component.score.notes == "Configured group column 'missing_group' is missing from a dataset."


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
