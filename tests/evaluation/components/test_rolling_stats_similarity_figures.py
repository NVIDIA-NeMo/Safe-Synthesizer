# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.errors import DataError
from nemo_safe_synthesizer.evaluation.components.rolling_stats_similarity_figures import (
    ROLLING_STATS_SUMMARY_TRACES_PER_COLUMN,
    generate_rolling_stats_pair_score_figure,
    generate_rolling_stats_profile_figure,
    generate_rolling_stats_segment_difference_figure,
    order_rolling_stats_columns,
)

_POSITIONS = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]


def _profile(group, column, training_level, synthetic_level, similarity, spread=None):
    spread = spread if spread is not None else [1.0] * len(training_level)
    return {
        "group": group,
        "column": column,
        "segments": len(training_level),
        "training_level": training_level,
        "synthetic_level": synthetic_level,
        "training_spread": spread,
        "synthetic_spread": [None] * len(spread) if all(v is None for v in synthetic_level) else spread,
        "similarity": similarity,
    }


_TEN = [float(index) for index in range(10)]
_PROFILES = [
    _profile("a", "good", _TEN, _TEN, 1.0),
    _profile("b", "good", [value + 2 for value in _TEN], [value + 2 for value in _TEN], 1.0),
    _profile("a", "bad", _TEN, [0.0] * 10, 0.2),
    _profile("b", "bad", _TEN, [None] * 10, 0.0),
]


def _trace(figure, name):
    return next(trace for trace in figure.data if trace.name == name)


def test_profile_figure_groups_traces_by_column_with_lowest_scoring_column_visible():
    figure = generate_rolling_stats_profile_figure(_PROFILES)

    assert order_rolling_stats_columns(_PROFILES) == ["bad", "good"]
    assert len(figure.data) == 2 * ROLLING_STATS_SUMMARY_TRACES_PER_COLUMN
    assert [trace.visible for trace in figure.data] == [True] * 8 + [False] * 8
    training = _trace(figure, "Training median local average")
    synthetic = _trace(figure, "Synthetic median local average")
    assert list(training.x) == pytest.approx(_POSITIONS)
    assert list(training.y) == pytest.approx(_TEN)
    assert list(synthetic.y) == pytest.approx([0.0] * 10)
    assert _trace(figure, "Training median local spread").yaxis == "y2"
    assert list(figure.layout.xaxis2.ticktext) == ["Start", "Middle", "End"]
    assert figure.layout.yaxis2.rangemode == "tozero"


def test_profile_figure_interpolates_pairs_with_fewer_segments_onto_ten_positions():
    figure = generate_rolling_stats_profile_figure([_profile(None, "short", [0.0, 1.0], [0.0, 1.0], 1.0)])

    training = _trace(figure, "Training median local average")
    assert len(training.x) == 10
    assert training.y[0] == pytest.approx(0.0)
    assert training.y[-1] == pytest.approx(1.0)


def test_segment_difference_figure_averages_paired_gaps_so_opposite_groups_cannot_cancel():
    rising, falling = _TEN, [-value for value in _TEN]
    profiles = [
        _profile("up", "value", rising, falling, 0.0),
        _profile("down", "value", falling, rising, 0.0),
    ]

    medians = generate_rolling_stats_profile_figure(profiles)
    difference = generate_rolling_stats_segment_difference_figure(profiles)

    assert list(_trace(medians, "Training median local average").y) == pytest.approx([0.0] * 10)
    assert list(_trace(medians, "Synthetic median local average").y) == pytest.approx([0.0] * 10)
    level = _trace(difference, "Average")
    assert list(level.x) == pytest.approx(_POSITIONS)
    assert list(level.y) == pytest.approx([2 * value for value in _TEN])
    assert list(level.customdata) == [2] * 10
    assert list(_trace(difference, "Spread").y) == pytest.approx([0.0] * 10)


def test_segment_difference_figure_skips_constant_pairs_and_keeps_a_fixed_axis():
    figure = generate_rolling_stats_segment_difference_figure(_PROFILES)

    level = _trace(figure, "Average")
    assert list(level.customdata) == [3] * 10
    assert level.y[1] == pytest.approx(1 / 3)
    assert level.marker.colorscale[0][1] == "#ffffff"
    assert _trace(figure, "Spread").marker.pattern.shape == "/"
    assert figure.layout.yaxis.range[0] == 0.0
    assert figure.layout.yaxis.range[1] >= 1.0
    assert all("training" not in trace.name.lower() and "synthetic" not in trace.name.lower() for trace in figure.data)


def test_pair_score_figure_plots_every_pair_without_naming_traces_after_columns():
    figure = generate_rolling_stats_pair_score_figure(_PROFILES, overall_score=5.5)

    assert [trace.name for trace in figure.data] == ["Pair scores", "Pair scores"]
    assert list(figure.data[0].x) == pytest.approx([2.0, 0.0])
    assert [list(item) for item in figure.data[0].customdata] == [["bad", "a"], ["bad", "b"]]
    assert list(figure.data[0].marker.color) == pytest.approx([0.8, 1.0])
    assert figure.layout.shapes[0].x0 == 5.5


def test_pair_score_figure_limits_columns_and_shortens_labels_without_merging_rows():
    profiles = [_profile("a", f"channel_{index:02d}_descriptive_name", [0.0], [0.0], index / 10) for index in range(10)]

    figure = generate_rolling_stats_pair_score_figure(profiles, None, max_columns=3)

    assert [item[0][0] for item in (trace.customdata for trace in figure.data)] == [
        "channel_00_descriptive_name",
        "channel_01_descriptive_name",
        "channel_02_descriptive_name",
    ]
    assert list(figure.layout.yaxis.ticktext) == ["cha...ame"] * 3
    assert [round(sum(trace.y) / len(trace.y)) for trace in figure.data] == [2, 1, 0]


def test_pair_score_figure_labels_ungrouped_profiles():
    figure = generate_rolling_stats_pair_score_figure([_profile(None, "value", [0.0], [0.0], 1.0)], None)

    assert list(figure.data[0].customdata[0]) == ["value", "all rows"]
    assert not figure.layout.shapes


@pytest.mark.parametrize(
    "builder",
    [
        generate_rolling_stats_profile_figure,
        generate_rolling_stats_segment_difference_figure,
        lambda profiles: generate_rolling_stats_pair_score_figure(profiles, None),
    ],
)
def test_figures_require_profiles(builder):
    with pytest.raises(DataError, match="At least one rolling statistics profile"):
        builder([])
