# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from nemo_safe_synthesizer.errors import DataError
from nemo_safe_synthesizer.evaluation.components.window_group_mia_figures import (
    WINDOW_GROUP_MIA_CLOSEST_TRACES_PER_COLUMN,
    generate_window_group_mia_closest_match_figure,
    generate_window_group_mia_distance_figure,
    generate_window_group_mia_group_figure,
)

_DETAILS = {
    "window_size": 2,
    "columns": ["temperature", "pressure"],
    "distances": {"training": [0.0, 2.0, 4.0], "holdout": [2.0, 4.0, 6.0]},
    "attack": {"threshold": 2.0, "holdout_flag_rate": 0.05},
    "closest_match": {
        "training_values": [[1.0, 2.0], [10.0, 11.0]],
        "synthetic_values": [[1.0, 2.5], [10.0, 12.0]],
    },
}


def _trace(figure, name):
    return next(trace for trace in figure.data if trace.name == name)


def test_distance_figure_shows_difference_per_value_and_the_flag_threshold():
    figure = generate_window_group_mia_distance_figure(_DETAILS)

    assert list(_trace(figure, "Training windows").x) == pytest.approx([0.0, 1.0, 2.0])
    assert list(_trace(figure, "Holdout windows").x) == pytest.approx([1.0, 2.0, 3.0])
    assert figure.layout.barmode == "overlay"
    assert len(figure.layout.shapes) == 1
    assert figure.layout.shapes[0].x0 == pytest.approx(1.0)
    assert figure.layout.xaxis.range[0] == 0


def test_distance_figure_without_holdout_shows_training_windows_only():
    details = {**_DETAILS, "distances": {"training": [0.0, 2.0]}}
    details.pop("attack")

    figure = generate_window_group_mia_distance_figure(details)

    assert [trace.name for trace in figure.data] == ["Training windows"]
    assert not figure.layout.shapes


def test_distance_figure_requires_training_distances():
    with pytest.raises(DataError):
        generate_window_group_mia_distance_figure({**_DETAILS, "distances": {}})


def test_group_figure_puts_the_most_flagged_groups_on_top_and_shades_excess_over_holdout():
    per_group = [
        {"group": f"g{index}", "windows": 20, "exact_copy_rate": 0.0, "flag_rate": rate}
        for index, rate in enumerate([0.05, 0.5, 1.0, 0.2])
    ]

    figure = generate_window_group_mia_group_figure(per_group, 0.05, max_rows=3)

    bars = figure.data[0]
    assert list(bars.x) == pytest.approx([20.0, 50.0, 100.0])
    assert list(figure.layout.yaxis.ticktext) == ["g3", "g1", "g2"]
    assert list(bars.marker.color) == pytest.approx(np.clip((np.array([0.2, 0.5, 1.0]) - 0.05) / 0.95, 0, 1))
    assert tuple(figure.layout.xaxis.range) == (0, 100)
    assert figure.layout.shapes[0].x0 == pytest.approx(5.0)


def test_group_figure_requires_flag_rates():
    with pytest.raises(DataError):
        generate_window_group_mia_group_figure([{"group": "a", "windows": 3, "exact_copy_rate": 0.0}], 0.05)


def test_closest_match_figure_has_training_and_synthetic_traces_per_column_with_first_visible():
    figure = generate_window_group_mia_closest_match_figure(_DETAILS)

    assert len(figure.data) == 2 * WINDOW_GROUP_MIA_CLOSEST_TRACES_PER_COLUMN
    assert [trace.visible for trace in figure.data] == [True, True, False, False]
    assert [trace.name for trace in figure.data[:2]] == ["Training window", "Synthetic window"]
    assert list(figure.data[1].y) == pytest.approx([1.0, 2.5])
    assert list(figure.data[3].y) == pytest.approx([10.0, 12.0])


def test_closest_match_figure_requires_a_match():
    with pytest.raises(DataError):
        generate_window_group_mia_closest_match_figure({**_DETAILS, "closest_match": None})
