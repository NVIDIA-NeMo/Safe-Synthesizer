# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from jinja2 import Environment, FunctionLoader, select_autoescape

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.evaluate import (
    EvaluationParameters,
    RollingStatsSimilarityParameters,
    TimeSeriesEvaluationParameters,
)
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters
from nemo_safe_synthesizer.evaluation.components.rolling_stats_similarity import RollingStatsSimilarity
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets
from nemo_safe_synthesizer.evaluation.render import _get_template


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


def _rendered_card(component: RollingStatsSimilarity) -> str:
    env = Environment(loader=FunctionLoader(_get_template), autoescape=select_autoescape(["html", "xml"]))
    rendered = env.get_template(component.template).render(ctx={"rolling_stats_similarity": component.jinja_context})
    return " ".join(rendered.split())


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


def _smaller_spread(rng: np.random.Generator, points: int) -> np.ndarray:
    """The rising level of ``_rising`` with half its spread."""
    position = np.linspace(0.0, 1.0, points)
    return 3 * position + 0.5 * (1 + position) * _noise(rng, points)


def _with_gaps(rng: np.random.Generator, values: np.ndarray, gaps: float | str) -> np.ndarray:
    """Blank a random share of values, or every other value."""
    if gaps == "every_other":
        keep = np.arange(len(values)) % 2 == 0
    else:
        keep = rng.random(len(values)) >= gaps
    return np.where(keep, values, np.nan)


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


@pytest.mark.parametrize("points", [200, 1000])
def test_rolling_stats_similarity_scores_same_process_high(points: int):
    assert _mean_score(lambda rng, n: (_rising(rng, n), _rising(rng, n)), points) >= 8.0


@pytest.mark.parametrize("points", [200, 1000])
def test_rolling_stats_similarity_scores_independent_noise_in_mild_tier_or_better(points: int):
    assert _mean_score(lambda rng, n: (_noise(rng, n), _noise(rng, n)), points, trials=40) >= 6.5


def test_rolling_stats_similarity_scores_same_random_walk_low():
    """Pin the known blind spot: segments are compared by position, so changes at random times lose points.

    Two random walks from the same process put their changes in different segments, so even a perfect
    generator scores low. Raising this score is a deliberate change to the metric, not a regression fix.
    """

    def make_pair(rng: np.random.Generator, points: int) -> tuple[np.ndarray, np.ndarray]:
        return np.cumsum(rng.normal(size=points)), np.cumsum(rng.normal(size=points))

    assert _mean_score(make_pair, 200) < 6.0


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
    assert _mean_score(lambda rng, n: (_rising(rng, n), _smaller_spread(rng, n)), 1000, trials=10) < 5.0


@pytest.mark.parametrize(
    "gaps",
    [
        pytest.param(0.2, id="random_20_percent"),
        pytest.param(0.5, id="random_50_percent"),
        pytest.param("every_other", id="every_other_value"),
    ],
)
def test_rolling_stats_similarity_missing_values_do_not_lower_same_process_scores(gaps: float | str):
    """Gaps shrink the windows the segments average, so the chance estimate must grow with them."""

    def make_pair(rng: np.random.Generator, points: int) -> tuple[np.ndarray, np.ndarray]:
        return _with_gaps(rng, _rising(rng, points), gaps), _with_gaps(rng, _rising(rng, points), gaps)

    seeds = (7, 8, 9)
    complete = np.mean([_mean_score(lambda rng, n: (_rising(rng, n), _rising(rng, n)), 200, seed=s) for s in seeds])
    gapped = np.mean([_mean_score(make_pair, 200, seed=s) for s in seeds])

    assert gapped >= complete - 0.3


def test_rolling_stats_similarity_penalizes_a_smaller_spread_with_missing_values():
    def make_pair(rng: np.random.Generator, points: int) -> tuple[np.ndarray, np.ndarray]:
        return _with_gaps(rng, _rising(rng, points), 0.2), _with_gaps(rng, _smaller_spread(rng, points), 0.2)

    assert _mean_score(make_pair, 1000, trials=10) < 5.0


def test_rolling_stats_similarity_identical_grouped_series_score_perfectly_in_any_row_order():
    training_df = pd.DataFrame(_grouped_rows([("B", 100), ("A", 0)])).sample(frac=1.0, random_state=7)
    config = _config(group_column="group")
    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(training_df, training_df.copy()), config)

    assert component.score.score == 10


def test_rolling_stats_similarity_constant_training_series_is_unavailable_instead_of_perfect():
    training_df = pd.DataFrame({"time": range(8), "value": [1.0] * 8})
    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(training_df, training_df.copy()), _config())

    assert component.score.score is None
    assert [item["column"] for item in component.details["skipped"]] == ["value"]


@pytest.mark.parametrize(
    ("constant_groups", "score", "notice"),
    [
        pytest.param(["B"], 7.5, "1 pair has constant or mostly missing synthetic values.", id="one_pair"),
        pytest.param(["B", "C"], 5.0, "2 pairs have constant or mostly missing synthetic values.", id="two_pairs"),
    ],
)
def test_rolling_stats_similarity_scores_constant_synthetic_groups_zero_and_says_so(
    constant_groups: list[str], score: float, notice: str
):
    training_df = pd.DataFrame(_grouped_rows([("A", 0), ("B", 100), ("C", 200), ("D", 300)]))
    synthetic_df = training_df.copy()
    synthetic_df.loc[synthetic_df["group"].isin(constant_groups), "value"] = 100
    config = _config(group_column="group")

    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(training_df, synthetic_df), config)

    assert component.score.score == score
    assert notice in _rendered_card(component)


def test_rolling_stats_similarity_scores_only_the_configured_value_columns():
    rng = np.random.default_rng(5)
    training_df = pd.DataFrame({"time": range(200), "x": _rising(rng, 200), "y": _rising(rng, 200)})
    synthetic_variants = [
        training_df.copy(),
        training_df.assign(y=0.0),
        training_df.assign(y=rng.permutation(training_df["y"].to_numpy())),
    ]
    only_x = _config(RollingStatsSimilarityParameters(value_columns=["x"]))

    scores = [
        RollingStatsSimilarity.from_evaluation_datasets(_datasets(training_df, synthetic), only_x).score.score
        for synthetic in synthetic_variants
    ]
    all_columns = RollingStatsSimilarity.from_evaluation_datasets(
        _datasets(training_df, synthetic_variants[1]), _config()
    )

    assert scores == [10, 10, 10]
    assert all_columns.score.score is not None and all_columns.score.score < 10


@pytest.mark.parametrize(
    "min_points",
    [
        pytest.param(4, id="no_window_has_enough_values"),
        pytest.param(20, id="below_min_points"),
    ],
)
def test_rolling_stats_similarity_scores_mostly_missing_synthetic_sequence_as_failure(min_points: int):
    rng = np.random.default_rng(11)
    training_df = pd.concat(
        [pd.DataFrame({"group": group, "time": range(200), "value": _rising(rng, 200)}) for group in ("A", "B")],
        ignore_index=True,
    )
    synthetic_df = training_df.copy()
    in_b = synthetic_df["group"] == "B"
    kept = np.arange(200) % 20 == 0
    synthetic_df.loc[in_b, "value"] = synthetic_df.loc[in_b, "value"].where(kept)
    config = _config(RollingStatsSimilarityParameters(min_points=min_points), group_column="group")

    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(training_df, synthetic_df), config)

    assert component.score.score == 5
    assert "1 pair has constant or mostly missing synthetic values." in _rendered_card(component)


def test_rolling_stats_similarity_skips_pairs_with_too_few_finite_training_values():
    training_df = pd.DataFrame({"time": range(5), "value": [0.0, np.inf, np.nan, 1.0, 2.0]})
    synthetic_df = pd.DataFrame({"time": range(5), "value": [0.0, 3.0, 1.0, 1.0, 2.0]})

    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(training_df, synthetic_df), _config())

    assert component.score.score is None
    assert [item["column"] for item in component.details["skipped"]] == ["value"]


@pytest.mark.parametrize("collapsed_values", [[0], [0, 1, 2]])
def test_rolling_stats_similarity_scores_collapsed_synthetic_integer_column(collapsed_values: list[int]):
    rng = np.random.default_rng(7)
    points = 200
    training_df = pd.DataFrame(
        {
            "time": range(points),
            "value": _rising(rng, points),
            "count": np.round(10 + 80 * rng.random(points)).astype(int),
        }
    )
    synthetic_df = training_df.copy()
    synthetic_df["count"] = np.resize(collapsed_values, points)

    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(training_df, synthetic_df), _config())

    profiled = {item["column"]: item for item in component.details["profiles"]}
    assert set(profiled) == {"count", "value"}
    assert component.score.score is not None
    assert component.score.score < 10
    if len(collapsed_values) == 1:
        assert profiled["count"]["similarity"] == 0


def test_rolling_stats_similarity_handles_aligned_non_finite_gaps():
    values = [0.0, 1.0, np.inf, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0]
    training_df = pd.DataFrame({"time": range(10), "value": values})
    synthetic_df = pd.DataFrame({"time": range(10), "value": [-np.inf if np.isinf(v) else v for v in values]})

    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(training_df, synthetic_df), _config())

    assert component.score.score == 10


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


def test_rolling_stats_similarity_skips_pairs_with_no_segment_finite_in_both_sequences():
    rng = np.random.default_rng(3)
    training = _rising(rng, 200)
    synthetic = _rising(rng, 200)
    training[20:] = np.nan
    synthetic[:-20] = np.nan

    component = RollingStatsSimilarity.from_evaluation_datasets(
        _datasets(
            pd.DataFrame({"time": range(200), "value": training}),
            pd.DataFrame({"time": range(200), "value": synthetic}),
        ),
        _config(),
    )

    assert component.score.score is None
    assert [item["column"] for item in component.details["skipped"]] == ["value"]


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


def test_rolling_stats_similarity_without_config_runs_with_component_defaults():
    frame = pd.DataFrame({"value": [0.0, 1.0, 3.0, 2.0, 4.0, 1.0, 5.0, 2.0]})

    component = RollingStatsSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()))

    assert component.score.score == 10
