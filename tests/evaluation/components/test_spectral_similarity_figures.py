# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.errors import DataError
from nemo_safe_synthesizer.evaluation.components.spectral_similarity_figures import (
    SPECTRAL_SUMMARY_TRACES_PER_COLUMN,
    generate_spectral_band_difference_figure,
    generate_spectral_band_figure,
    generate_spectral_pair_score_figure,
    order_spectral_columns,
)

_CYCLES = [2.0, 2 * 2**0.5, 4.0, 4 * 2**0.5]


def _profile(group, column, training, synthetic, similarity, cycles=None):
    return {
        "group": group,
        "column": column,
        "band_cycle_lengths": cycles or _CYCLES[: len(training)],
        "training_bands": training,
        "synthetic_bands": synthetic,
        "similarity": similarity,
    }


_EVEN = [0.25, 0.25, 0.25, 0.25]
_PEAK = [0.1, 0.1, 0.7, 0.1]
_PROFILES = [
    _profile("a", "good", _EVEN, _EVEN, 1.0),
    _profile("b", "good", _PEAK, _PEAK, 1.0),
    _profile("a", "bad", _PEAK, _EVEN, 0.2),
    _profile("b", "bad", _PEAK, [None] * 4, 0.0),
]


def _trace(figure, name):
    return next(trace for trace in figure.data if trace.name == name)


def test_band_figure_groups_traces_by_column_with_lowest_scoring_column_visible():
    figure = generate_spectral_band_figure(_PROFILES)

    assert order_spectral_columns(_PROFILES) == ["bad", "good"]
    assert len(figure.data) == 2 * SPECTRAL_SUMMARY_TRACES_PER_COLUMN
    assert [trace.visible for trace in figure.data] == [True] * 4 + [False] * 4
    training = _trace(figure, "Training median share")
    synthetic = _trace(figure, "Synthetic median share")
    assert list(training.x) == ["2", "3", "4", "6"]
    assert list(training.y) == pytest.approx([10.0, 10.0, 70.0, 10.0])
    assert list(synthetic.y) == pytest.approx([25.0, 25.0, 25.0, 25.0])
    assert figure.layout.xaxis.type == "category"
    assert figure.layout.yaxis.rangemode == "tozero"


def test_band_figure_places_pairs_with_different_bands_by_cycle_length():
    short = _profile("a", "value", [0.5, 0.5], [0.5, 0.5], 1.0)
    long = _profile("b", "value", [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], 1.0)

    figure = generate_spectral_band_figure([short, long])

    training = _trace(figure, "Training median share")
    assert list(training.x) == ["2", "3", "4", "6"]
    assert list(training.y) == pytest.approx([30.0, 35.0, 30.0, 40.0])


def test_band_difference_figure_averages_paired_gaps_so_groups_cannot_cancel():
    first, second = [0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]
    profiles = [
        _profile("one", "value", first, second, 0.0),
        _profile("two", "value", second, first, 0.0),
    ]

    medians = generate_spectral_band_figure(profiles)
    difference = generate_spectral_band_difference_figure(profiles)

    assert list(_trace(medians, "Training median share").y) == pytest.approx(
        list(_trace(medians, "Synthetic median share").y)
    )
    bars = _trace(difference, "Mean difference")
    assert list(bars.x) == ["2", "3", "4", "6"]
    assert list(bars.y) == pytest.approx([60.0, 0.0, 0.0, 60.0])
    assert list(bars.customdata) == [2, 2, 2, 2]


def test_band_difference_figure_skips_constant_pairs_and_keeps_a_fixed_axis():
    figure = generate_spectral_band_difference_figure(_PROFILES)

    bars = _trace(figure, "Mean difference")
    assert list(bars.customdata) == [3, 3, 3, 3]
    assert bars.y[2] == pytest.approx(15.0)
    assert bars.marker.colorscale[0][1] == "#ffffff"
    assert figure.layout.yaxis.range[0] == 0.0
    assert figure.layout.yaxis.range[1] >= 25.0
    assert all("training" not in trace.name.lower() and "synthetic" not in trace.name.lower() for trace in figure.data)


def test_pair_score_figure_plots_every_pair_without_naming_traces_after_columns():
    figure = generate_spectral_pair_score_figure(_PROFILES, overall_score=5.5)

    assert [trace.name for trace in figure.data] == ["Pair scores", "Pair scores"]
    assert list(figure.data[0].x) == pytest.approx([2.0, 0.0])
    assert [list(item) for item in figure.data[0].customdata] == [["bad", "a"], ["bad", "b"]]
    assert list(figure.data[0].marker.color) == pytest.approx([0.8, 1.0])
    assert figure.layout.shapes[0].x0 == 5.5


def test_pair_score_figure_limits_columns_and_shortens_labels_without_merging_rows():
    profiles = [_profile("a", f"channel_{index:02d}_descriptive_name", [1.0], [1.0], index / 10) for index in range(10)]

    figure = generate_spectral_pair_score_figure(profiles, None, max_columns=3)

    assert [item[0][0] for item in (trace.customdata for trace in figure.data)] == [
        "channel_00_descriptive_name",
        "channel_01_descriptive_name",
        "channel_02_descriptive_name",
    ]
    assert list(figure.layout.yaxis.ticktext) == ["cha...ame"] * 3
    assert [round(sum(trace.y) / len(trace.y)) for trace in figure.data] == [2, 1, 0]


def test_pair_score_figure_labels_ungrouped_profiles():
    figure = generate_spectral_pair_score_figure([_profile(None, "value", [1.0], [1.0], 1.0)], None)

    assert list(figure.data[0].customdata[0]) == ["value", "all rows"]
    assert not figure.layout.shapes


@pytest.mark.parametrize(
    "builder",
    [
        generate_spectral_band_figure,
        generate_spectral_band_difference_figure,
        lambda profiles: generate_spectral_pair_score_figure(profiles, None),
    ],
)
def test_figures_require_profiles(builder):
    with pytest.raises(DataError, match="At least one spectral profile"):
        builder([])
