# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.errors import DataError
from nemo_safe_synthesizer.evaluation.components.autocorrelation_similarity_figures import (
    generate_autocorrelation_lag_error_figure,
    generate_autocorrelation_summary_figure,
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


@pytest.mark.parametrize(
    "builder", [generate_autocorrelation_summary_figure, generate_autocorrelation_lag_error_figure]
)
def test_summary_figures_require_profiles(builder):
    with pytest.raises(DataError, match="At least one"):
        builder([])
