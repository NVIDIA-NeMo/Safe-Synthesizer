# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.errors import DataError
from nemo_safe_synthesizer.evaluation.components.time_series_common_figures import (
    PAIR_SCORE_MAX_COLUMNS,
    generate_pair_score_figure,
    order_columns_by_score,
    shorten_column_label,
)


def _profile(group, column, similarity):
    return {"group": group, "column": column, "similarity": similarity}


_PROFILES = [
    _profile("a", "good", 1.0),
    _profile("b", "good", 1.0),
    _profile("a", "bad", 0.2),
    _profile("b", "bad", 0.0),
]


def test_pair_score_figure_plots_every_pair_with_the_lowest_scoring_column_on_top():
    figure = generate_pair_score_figure(_PROFILES, 5.5, height=340)

    bad, good = figure.data
    assert list(bad.x) == pytest.approx([2.0, 0.0])
    assert [list(item) for item in bad.customdata] == [["bad", "a"], ["bad", "b"]]
    assert list(bad.marker.color) == pytest.approx([0.8, 1.0])
    assert all(abs(y - 1) <= 0.25 for y in bad.y)
    assert all(abs(y) <= 0.25 for y in good.y)
    assert list(figure.layout.yaxis.ticktext) == ["good", "bad"]
    assert figure.layout.shapes[0].x0 == 5.5
    assert figure.layout.height == 340


def test_pair_score_figure_trace_names_cannot_match_dataset_theming():
    figure = generate_pair_score_figure([_profile("a", "synthetic_total", 0.5)], None, height=300)

    assert "synthetic" not in figure.data[0].name.lower()
    assert list(figure.data[0].customdata[0]) == ["synthetic_total", "a"]


def test_pair_score_figure_limits_columns_and_shortens_labels_without_merging_rows():
    profiles = [_profile("a", f"channel_{index:02d}_descriptive_name", index / 10) for index in range(10)]

    figure = generate_pair_score_figure(profiles, None, height=300)

    plotted = [trace.customdata[0][0] for trace in figure.data]
    assert plotted == [f"channel_{index:02d}_descriptive_name" for index in range(PAIR_SCORE_MAX_COLUMNS)]
    assert list(figure.layout.yaxis.ticktext) == ["cha...ame"] * PAIR_SCORE_MAX_COLUMNS
    rows = [round(sum(trace.y) / len(trace.y)) for trace in figure.data]
    assert rows == list(reversed(range(PAIR_SCORE_MAX_COLUMNS)))


def test_pair_score_figure_renders_non_string_column_labels():
    profiles = [_profile("a", 0, 0.9), _profile("b", 0, 0.7), _profile("a", 1.5, 0.2)]

    figure = generate_pair_score_figure(profiles, None, height=300)

    assert order_columns_by_score(profiles) == ["1.5", "0"]
    assert [len(trace.x) for trace in figure.data] == [1, 2]
    assert list(figure.layout.yaxis.ticktext) == ["0", "1.5"]


def test_pair_score_figure_labels_ungrouped_profiles():
    figure = generate_pair_score_figure([_profile(None, "value", 1.0)], None, height=300)

    assert list(figure.data[0].customdata[0]) == ["value", "all rows"]
    assert not figure.layout.shapes


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


def test_pair_score_figure_requires_profiles():
    with pytest.raises(DataError, match="At least one pair profile"):
        generate_pair_score_figure([], None, height=300)
