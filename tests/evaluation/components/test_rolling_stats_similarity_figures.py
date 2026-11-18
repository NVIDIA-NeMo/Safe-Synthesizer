# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.errors import DataError
from nemo_safe_synthesizer.evaluation.components.rolling_stats_similarity_figures import (
    ROLLING_STATS_SUMMARY_TRACES_PER_COLUMN,
    generate_rolling_stats_profile_figure,
    generate_rolling_stats_segment_difference_figure,
)

_POSITIONS = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]


def _profile(group, column, training_level, synthetic_level, similarity, spread=None):
    spread = spread if spread is not None else [1.0] * len(training_level)
    return {
        "group": group,
        "column": column,
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


def test_profile_figure_shows_one_column_at_a_time_lowest_scoring_first():
    figure = generate_rolling_stats_profile_figure(_PROFILES)

    assert len(figure.data) == 2 * ROLLING_STATS_SUMMARY_TRACES_PER_COLUMN
    assert [trace.visible for trace in figure.data] == [True] * 8 + [False] * 8
    training = _trace(figure, "Training median local average")
    assert list(training.x) == pytest.approx(_POSITIONS)
    assert list(training.y) == pytest.approx(_TEN)
    assert list(_trace(figure, "Synthetic median local average").y) == pytest.approx([0.0] * 10)


def test_profile_figure_medians_leave_out_pairs_without_synthetic_values():
    extreme = [100.0] * 10
    profiles = [
        _profile("a", "value", _TEN, _TEN, 1.0),
        _profile("b", "value", _TEN, _TEN, 1.0),
        _profile("c", "value", extreme, [None] * 10, 0.0, spread=extreme),
    ]

    figure = generate_rolling_stats_profile_figure(profiles)

    assert list(_trace(figure, "Training median local average").y) == pytest.approx(_TEN)
    assert list(_trace(figure, "Training median local spread").y) == pytest.approx([1.0] * 10)


def test_profile_figure_keeps_trace_layout_when_every_pair_lacks_synthetic_values():
    profiles = [_profile("a", "constant", _TEN, [None] * 10, 0.0), _profile("a", "value", _TEN, _TEN, 1.0)]

    figure = generate_rolling_stats_profile_figure(profiles)

    assert len(figure.data) == 2 * ROLLING_STATS_SUMMARY_TRACES_PER_COLUMN
    assert len(_trace(figure, "Training median local average").x) == 0


def test_profile_figure_interpolates_pairs_with_fewer_segments_onto_ten_positions():
    figure = generate_rolling_stats_profile_figure([_profile(None, "short", [0.0, 1.0], [0.0, 1.0], 1.0)])

    training = _trace(figure, "Training median local average")
    assert len(training.x) == 10
    assert training.y[0] == pytest.approx(0.0)
    assert training.y[-1] == pytest.approx(1.0)


def test_profile_figure_keeps_a_shorter_pair_with_one_usable_segment():
    level = [3.0] + [None] * 7
    profiles = [_profile(None, "sparse", level, level, 1.0, spread=[0.5] + [None] * 7)]

    profile_figure = generate_rolling_stats_profile_figure(profiles)
    difference_figure = generate_rolling_stats_segment_difference_figure(profiles)

    training = _trace(profile_figure, "Training median local average")
    assert list(training.x) == pytest.approx([_POSITIONS[0]])
    assert list(training.y) == pytest.approx([3.0])
    assert list(_trace(profile_figure, "Training median local spread").y) == pytest.approx([0.5])
    assert any(len(trace.x) for trace in difference_figure.data)


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


def test_segment_difference_figure_skips_pairs_without_synthetic_values_on_a_fixed_axis():
    figure = generate_rolling_stats_segment_difference_figure(_PROFILES)

    level = _trace(figure, "Average")
    assert list(level.customdata) == [3] * 10
    assert level.y[1] == pytest.approx(1 / 3)
    assert figure.layout.yaxis.range[0] == 0.0
    assert figure.layout.yaxis.range[1] >= 1.0
    assert all("training" not in trace.name.lower() and "synthetic" not in trace.name.lower() for trace in figure.data)


@pytest.mark.parametrize(
    "builder", [generate_rolling_stats_profile_figure, generate_rolling_stats_segment_difference_figure]
)
def test_figures_require_profiles(builder):
    with pytest.raises(DataError, match="At least one"):
        builder([])
