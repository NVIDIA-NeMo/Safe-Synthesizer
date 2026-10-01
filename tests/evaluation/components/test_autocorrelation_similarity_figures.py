# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from nemo_safe_synthesizer.errors import DataError, ParameterError
from nemo_safe_synthesizer.evaluation.components.autocorrelation_similarity_figures import (
    generate_autocorrelation_lag_error_figure,
    generate_autocorrelation_pair_score_figure,
    generate_autocorrelation_profile_figure,
    generate_autocorrelation_similarity_figure,
    generate_autocorrelation_summary_figure,
    order_autocorrelation_columns,
    shorten_column_label,
)


def _profile(group, column, training_acf, synthetic_acf, similarity):
    return {
        "group": group,
        "column": column,
        "lags": list(range(1, len(training_acf) + 1)),
        "training_acf": training_acf,
        "synthetic_acf": synthetic_acf,
        "similarity": similarity,
    }


_SUMMARY_PROFILES = [
    _profile("a", "good", [0.9, 0.8, 0.7], [0.9, 0.8, 0.7], 1.0),
    _profile("b", "good", [0.7, 0.6], [0.7, 0.6], 1.0),
    _profile("a", "bad", [0.9, 0.8, 0.7], [0.1, 0.0, None], 0.6),
    _profile("b", "bad", [0.5, 0.4, 0.3], [0.3, 0.2, 0.1], 0.9),
]


def test_summary_figure_groups_traces_by_column_with_lowest_scoring_column_visible():
    figure = generate_autocorrelation_summary_figure(_SUMMARY_PROFILES)

    assert order_autocorrelation_columns(_SUMMARY_PROFILES) == ["bad", "good"]
    assert len(figure.data) == 8
    assert [trace.visible for trace in figure.data] == [True] * 4 + [False] * 4
    medians = [trace for trace in figure.data if trace.name == "Training median ACF"]
    assert list(medians[0].x) == [1, 2, 3]
    assert list(medians[0].y) == pytest.approx([0.7, 0.6, 0.5])
    synthetic_median = next(trace for trace in figure.data if trace.name == "Synthetic median ACF")
    assert list(synthetic_median.y) == pytest.approx([0.2, 0.1, 0.1])
    assert not figure.layout.updatemenus


def test_lag_error_figure_averages_only_paired_finite_lags():
    figure = generate_autocorrelation_lag_error_figure(_SUMMARY_PROFILES)

    bar = figure.data[0]
    assert list(bar.x) == [1, 2, 3]
    assert list(bar.y) == pytest.approx([(0 + 0 + 0.8 + 0.2) / 4, (0 + 0 + 0.8 + 0.2) / 4, (0 + 0.2) / 2])
    assert list(bar.customdata) == [4, 4, 2]


def test_pair_score_figure_plots_every_pair_on_report_scale():
    figure = generate_autocorrelation_pair_score_figure(_SUMMARY_PROFILES, overall_score=8.8)

    assert [trace.name for trace in figure.data] == ["bad", "good"]
    assert list(figure.data[0].x) == pytest.approx([6.0, 9.0])
    assert list(figure.data[0].customdata) == ["a", "b"]
    assert figure.layout.shapes[0].x0 == 8.8


def test_pair_score_figure_limits_columns_and_shortens_labels_without_merging_rows():
    profiles = [_profile("a", f"channel_{index:02d}_descriptive_name", [0.5], [0.5], index / 10) for index in range(10)]

    figure = generate_autocorrelation_pair_score_figure(profiles, None, max_columns=3)

    assert [trace.name for trace in figure.data] == [
        "channel_00_descriptive_name",
        "channel_01_descriptive_name",
        "channel_02_descriptive_name",
    ]
    assert list(figure.layout.yaxis.tickvals) == list(reversed([trace.name for trace in figure.data]))
    assert list(figure.layout.yaxis.ticktext) == ["cha...ame"] * 3


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("pressure", "pressure"),
        ("exactly_10", "exactly_10"),
        ("temperature", "tem...ure"),
    ],
)
def test_shorten_column_label(name, label):
    assert shorten_column_label(name) == label


def test_pair_score_figure_labels_ungrouped_profiles():
    figure = generate_autocorrelation_pair_score_figure([_profile(None, "value", [0.5], [0.5], 1.0)], None)

    assert list(figure.data[0].customdata) == ["all rows"]
    assert not figure.layout.shapes


@pytest.mark.parametrize(
    "builder",
    [
        generate_autocorrelation_summary_figure,
        generate_autocorrelation_lag_error_figure,
        lambda profiles: generate_autocorrelation_pair_score_figure(profiles, None),
    ],
)
def test_summary_figures_require_profiles(builder):
    with pytest.raises(DataError, match="At least one autocorrelation profile"):
        builder([])


def test_autocorrelation_similarity_figure_uses_requested_lags_without_mutating_inputs():
    training_df = pd.Series(np.sin(np.arange(60) / 5))
    synthetic_df = training_df.shift(1).bfill()
    training_before = training_df.copy()
    synthetic_before = synthetic_df.copy()

    figure = generate_autocorrelation_similarity_figure(training_df, synthetic_df, max_lag=8)

    assert isinstance(figure, go.Figure)
    assert [trace.name for trace in figure.data] == ["Training ACF", "Synthetic ACF"]
    assert list(figure.data[0].x) == list(range(1, 9))
    assert figure.layout.yaxis.range == (-1.05, 1.05)
    pd.testing.assert_series_equal(training_df, training_before)
    pd.testing.assert_series_equal(synthetic_df, synthetic_before)


def test_autocorrelation_similarity_figure_preserves_non_finite_positions():
    training_df = pd.Series([0.0, 1.0, np.inf, 2.0, 3.0, 2.0, 1.0])
    synthetic_df = pd.Series([0.0, -np.inf, 1.0, 2.0, 3.0, 2.0, 1.0])

    figure = generate_autocorrelation_similarity_figure(training_df, synthetic_df)

    assert np.isfinite(figure.data[0].y).all()
    assert np.isfinite(figure.data[1].y).all()
    assert list(figure.data[0].y) != list(figure.data[1].y)


def test_autocorrelation_profile_figure_uses_precomputed_values():
    figure = generate_autocorrelation_profile_figure(
        [1, 2],
        [0.75, 0.25],
        [0.5, None],
    )

    assert list(figure.data[0].x) == [1, 2]
    assert list(figure.data[0].y) == [0.75, 0.25]
    assert list(figure.data[1].y) == [0.5, None]


@pytest.mark.parametrize(
    ("training_df", "synthetic_df", "message"),
    [
        (pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0, 3.0]), "At least 4 finite points"),
        (pd.Series([1.0, 2.0, np.inf, np.nan]), pd.Series(range(4)), "At least 4 finite points"),
        (pd.Series([1.0] * 8), pd.Series(range(8)), "constant or near-constant"),
    ],
)
def test_autocorrelation_similarity_figure_rejects_unusable_series(training_df, synthetic_df, message):
    with pytest.raises(DataError, match=message):
        generate_autocorrelation_similarity_figure(training_df, synthetic_df)


def test_autocorrelation_similarity_figure_rejects_invalid_max_lag():
    values = pd.Series(range(8))

    with pytest.raises(ParameterError, match="max_lag must be at least 1"):
        generate_autocorrelation_similarity_figure(values, values, max_lag=0)
