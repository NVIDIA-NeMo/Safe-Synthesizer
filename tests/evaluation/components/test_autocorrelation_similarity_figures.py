# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.errors import DataError
from nemo_safe_synthesizer.evaluation.components.autocorrelation_similarity_figures import (
    generate_autocorrelation_lag_error_figure,
    generate_autocorrelation_pair_score_figure,
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
    assert list(bar.marker.color) == pytest.approx(list(bar.y))
    assert bar.marker.colorscale[0][1] == "#ffffff"
    assert bar.marker.colorscale[-1][1] == "#dc2626"
    assert figure.layout.yaxis.range == (0.0, 1.0)


def test_lag_error_figure_axis_grows_past_one():
    profiles = [_profile("a", "flipped", [0.9, -0.9], [-0.9, 0.9], 0.0)]

    figure = generate_autocorrelation_lag_error_figure(profiles)

    assert figure.layout.yaxis.range[1] == pytest.approx(1.8 * 1.05)


def test_pair_score_figure_plots_every_pair_on_report_scale():
    figure = generate_autocorrelation_pair_score_figure(_SUMMARY_PROFILES, overall_score=8.8)

    assert [trace.name for trace in figure.data] == ["bad", "good"]
    assert list(figure.data[0].x) == pytest.approx([6.0, 9.0])
    assert list(figure.data[0].customdata) == ["a", "b"]
    assert "group %{customdata}" in figure.data[0].hovertemplate
    assert list(figure.data[0].marker.color) == pytest.approx([0.4, 0.1])
    assert list(figure.data[1].marker.color) == pytest.approx([0.0, 0.0])
    assert figure.layout.shapes[0].x0 == 8.8


def test_pair_score_figure_jitters_points_within_their_row():
    figure = generate_autocorrelation_pair_score_figure(_SUMMARY_PROFILES, None)

    bad, good = figure.data
    assert all(abs(y - 1) <= 0.25 for y in bad.y)
    assert all(abs(y) <= 0.25 for y in good.y)
    assert list(generate_autocorrelation_pair_score_figure(_SUMMARY_PROFILES, None).data[0].y) == list(bad.y)


def test_pair_score_figure_limits_columns_and_shortens_labels_without_merging_rows():
    profiles = [_profile("a", f"channel_{index:02d}_descriptive_name", [0.5], [0.5], index / 10) for index in range(10)]

    figure = generate_autocorrelation_pair_score_figure(profiles, None, max_columns=3)

    assert [trace.name for trace in figure.data] == [
        "channel_00_descriptive_name",
        "channel_01_descriptive_name",
        "channel_02_descriptive_name",
    ]
    assert list(figure.layout.yaxis.tickvals) == [0, 1, 2]
    assert list(figure.layout.yaxis.ticktext) == ["cha...ame"] * 3
    assert [round(sum(trace.y) / len(trace.y)) for trace in figure.data] == [2, 1, 0]


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
