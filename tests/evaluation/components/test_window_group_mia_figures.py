# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from nemo_safe_synthesizer.errors import DataError
from nemo_safe_synthesizer.evaluation.components.window_group_mia_figures import (
    generate_window_group_mia_closest_match_figure,
    generate_window_group_mia_distance_figure,
    generate_window_group_mia_group_figure,
)

_EXACT_MATCH: dict[str, object] = {
    "exact_copy": True,
    "training_values": [[1.0, 2.0], [10.0, 11.0]],
    "synthetic_values": [[1.0, 2.0], [10.0, 11.0]],
}
_DETAILS = {
    "window_size": 2,
    "columns": ["temperature", "pressure"],
    "distances": {"training": [0.0, 2.0, 4.0], "holdout": [2.0, 4.0, 6.0]},
    "attack": {"threshold": 2.0, "holdout_flag_rate": 0.05},
    "closest_match": _EXACT_MATCH,
}


@pytest.mark.parametrize("with_holdout", [True, False])
def test_distance_figure_spans_the_distances_per_value_from_zero(with_holdout: bool):
    details = dict(_DETAILS)
    if not with_holdout:
        details["distances"] = {"training": _DETAILS["distances"]["training"]}
        details.pop("attack")

    figure = generate_window_group_mia_distance_figure(details)

    per_value = np.concatenate([trace.x for trace in figure.data])
    low, high = figure.layout.xaxis.range
    assert low == 0
    assert high >= np.quantile(per_value, 0.99)
    assert per_value.max() == pytest.approx(3.0 if with_holdout else 2.0)
    assert len(figure.data) == (2 if with_holdout else 1)
    if with_holdout:
        assert low < figure.layout.shapes[0].x0 < high
    else:
        assert not figure.layout.shapes


def test_group_figure_puts_the_most_flagged_groups_on_top_and_shades_excess_over_holdout():
    rates = [0.05, 0.5, 1.0, *([0.2] * 12)]
    per_group = [
        {"group": f"g{index:02d}", "windows": 20, "exact_copy_rate": 0.0, "flag_rate": rate}
        for index, rate in enumerate(rates)
    ]

    figure = generate_window_group_mia_group_figure(per_group, 0.05)

    bars = figure.data[0]
    assert len(bars.x) == 12
    assert list(figure.layout.yaxis.ticktext)[-2:] == ["g01", "g02"]
    assert list(bars.x)[-1] == pytest.approx(100.0)
    assert bars.marker.color[-1] == pytest.approx(1.0)
    assert list(bars.marker.color) == sorted(bars.marker.color)
    assert tuple(figure.layout.xaxis.range) == (0, 100)


@pytest.mark.parametrize("exact_copy", [True, False])
def test_closest_match_figure_draws_the_training_window_only_for_an_exact_copy(exact_copy: bool):
    match = _EXACT_MATCH if exact_copy else {"exact_copy": False, "synthetic_values": [[1.5, 2.5], [10.5, 12.5]]}

    figure = generate_window_group_mia_closest_match_figure({**_DETAILS, "closest_match": match})

    names = [trace.name for trace in figure.data]
    assert ("Training window" in names) is exact_copy
    assert names.count("Synthetic window") == 2
    assert [trace.visible for trace in figure.data].count(True) == len(figure.data) // 2
    assert list(figure.data[-1].y) == pytest.approx(match["synthetic_values"][1])


@pytest.mark.parametrize(
    ("generate", "details"),
    [
        (generate_window_group_mia_distance_figure, {**_DETAILS, "distances": {}}),
        (generate_window_group_mia_closest_match_figure, {**_DETAILS, "closest_match": None}),
        (
            lambda per_group: generate_window_group_mia_group_figure(per_group, 0.05),
            [{"group": "a", "windows": 3, "exact_copy_rate": 0.0}],
        ),
    ],
    ids=["distance", "closest_match", "group"],
)
def test_figures_require_their_inputs(generate, details):
    with pytest.raises(DataError):
        generate(details)
