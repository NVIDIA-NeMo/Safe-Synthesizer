# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plotly report figures shared by time-series pair metrics.

Pair metrics store one profile per group and value column, each with a
``similarity`` in ``[0, 1]``. The builders here only read ``group``,
``column`` and ``similarity``, so every pair metric can reuse them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from ...errors import DataError

# Matches the Correlation Stability difference heatmap in multi_modal_toggle.js.
DIFFERENCE_COLORSCALE = [[0, "#ffffff"], [0.25, "#fee2e2"], [0.5, "#fca5a5"], [0.75, "#f87171"], [1, "#dc2626"]]
PAIR_SCORE_MAX_COLUMNS = 8
_PAIR_SCORE_JITTER = 0.25
_LABEL_MAX_LENGTH = 10
_LABEL_EDGE_LENGTH = 3


def generate_pair_score_figure(
    profiles: Sequence[Mapping[str, Any]],
    overall_score: float | None,
    *,
    height: int,
) -> go.Figure:
    """Plot group and column pair scores on the report's 0-10 scale.

    Only the ``PAIR_SCORE_MAX_COLUMNS`` lowest-scoring columns are plotted so
    each row stays readable. Each row sits at an integer y position with
    seeded vertical jitter, so long names are shortened only in the axis
    labels and shortened names cannot merge rows. Dots shade from white at a
    score of 10 to red at 0. Traces are not named after columns, so the
    report's dataset theming cannot recolor a column named like a dataset.

    Args:
        profiles: Stored pair profiles from a metric's ``details``.
        overall_score: Aggregate metric score drawn as a reference line.
        height: Figure height in pixels, matching the metric's other charts.

    Returns:
        A Plotly strip figure with one marker trace per plotted value column.

    Raises:
        DataError: If no profiles are supplied.
    """
    columns = order_columns_by_score(profiles)[:PAIR_SCORE_MAX_COLUMNS]
    rows = list(reversed(columns))
    rng = np.random.default_rng(0)
    figure = go.Figure()
    for column in columns:
        column_profiles = [item for item in profiles if str(item["column"]) == column]
        labels = [[column, "all rows" if item["group"] is None else str(item["group"])] for item in column_profiles]
        similarity = np.array([float(item["similarity"]) for item in column_profiles])
        jitter = rng.uniform(-_PAIR_SCORE_JITTER, _PAIR_SCORE_JITTER, size=len(column_profiles))
        figure.add_trace(
            go.Scatter(
                x=10 * similarity,
                y=rows.index(column) + jitter,
                customdata=labels,
                mode="markers",
                marker={
                    "color": 1 - similarity,
                    "colorscale": DIFFERENCE_COLORSCALE,
                    "cmin": 0.0,
                    "cmax": 1.0,
                    "size": 7,
                    "opacity": 0.9,
                },
                name="Pair scores",
                hovertemplate="%{customdata[0]}<br>group %{customdata[1]}<br>score %{x:.1f}<extra></extra>",
            )
        )
    if overall_score is not None:
        figure.add_vline(
            x=overall_score,
            line={"color": "rgba(255,255,255,0.6)", "dash": "dash", "width": 1},
            annotation_text=f"score {overall_score:.1f}",
            annotation_position="top",
        )
    figure.update_layout(
        template="plotly_white",
        height=height,
        showlegend=False,
        xaxis_title="Pair score",
        margin={"l": 80, "r": 16, "t": 40, "b": 48},
    )
    figure.update_xaxes(range=[-0.25, 10.25])
    figure.update_yaxes(
        range=[-0.6, len(rows) - 0.4],
        tickmode="array",
        tickvals=list(range(len(rows))),
        ticktext=[shorten_column_label(row) for row in rows],
        zeroline=False,
    )
    return figure


def order_columns_by_score(profiles: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return value columns ordered from lowest to highest mean pair similarity.

    Column names are converted with ``str`` so non-string labels sort and
    render like any other name.

    Args:
        profiles: Stored pair profiles from a metric's ``details``.

    Returns:
        Column names, lowest mean pair similarity first, with names breaking ties.

    Raises:
        DataError: If no profiles are supplied.
    """
    if not profiles:
        raise DataError("At least one pair profile is required.")
    scores: dict[str, list[float]] = {}
    for item in profiles:
        scores.setdefault(str(item["column"]), []).append(float(item["similarity"]))
    return sorted(scores, key=lambda column: (float(np.mean(scores[column])), column))


def shorten_column_label(name: str) -> str:
    """Shorten a column name longer than 10 characters to its first and last 3.

    Args:
        name: Full column name.

    Returns:
        The name unchanged, or its first and last three characters joined by an ellipsis.
    """
    if len(name) <= _LABEL_MAX_LENGTH:
        return name
    return f"{name[:_LABEL_EDGE_LENGTH]}...{name[-_LABEL_EDGE_LENGTH:]}"
