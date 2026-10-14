# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plotly report figures for rolling statistics similarity.

Each builder summarizes the per-pair segment profiles stored by the metric in
``RollingStatsSimilarity.details``.

Functions:
    generate_rolling_stats_profile_figure: Plot median local averages and
        spreads with interquartile bands across groups, one column at a time.
    generate_rolling_stats_segment_difference_figure: Plot the mean paired
        difference in local average and spread in each segment.
    generate_rolling_stats_pair_score_figure: Plot group and column pair
        scores for the lowest-scoring columns against the overall score.
    order_rolling_stats_columns: Order value columns by mean pair score.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...errors import DataError
from .autocorrelation_similarity_figures import shorten_column_label
from .rolling_stats_similarity import ROLLING_STATS_SEGMENTS

_TRAINING_COLOR = "#3b82f6"
_SYNTHETIC_COLOR = "#f59e0b"
_TRAINING_BAND_COLOR = "rgba(59, 130, 246, 0.2)"
_SYNTHETIC_BAND_COLOR = "rgba(245, 158, 11, 0.2)"
# Matches the Correlation Stability difference heatmap in multi_modal_toggle.js.
_DIFFERENCE_COLORSCALE = [[0, "#ffffff"], [0.25, "#fee2e2"], [0.5, "#fca5a5"], [0.75, "#f87171"], [1, "#dc2626"]]
_SEGMENT_DIFFERENCE_COLOR_MAX = 1.0
_PAIR_SCORE_JITTER = 0.25
_FIGURE_HEIGHT = 340
_POSITIONS = 100 * (np.arange(ROLLING_STATS_SEGMENTS) + 0.5) / ROLLING_STATS_SEGMENTS
_STATISTICS = (("level", "Local average"), ("spread", "Local spread"))
_POSITION_TICKS = {"tickmode": "array", "tickvals": [0, 50, 100], "ticktext": ["Start", "Middle", "End"]}

ROLLING_STATS_SUMMARY_TRACES_PER_COLUMN = 8
ROLLING_STATS_PAIR_SCORE_MAX_COLUMNS = 8


def generate_rolling_stats_profile_figure(profiles: Sequence[Mapping[str, Any]]) -> go.Figure:
    """Plot typical training and synthetic local averages and spreads from start to end.

    The top panel shows the local average, in training standard deviations
    from the training mean, and the bottom panel the local spread, in training
    standard deviations. Each value column gets a median profile and an
    interquartile band for both datasets in both panels. Columns contribute
    ``ROLLING_STATS_SUMMARY_TRACES_PER_COLUMN`` consecutive traces in
    ``order_rolling_stats_columns`` order, and only the first column is visible
    so the report can switch columns by trace index. Pairs with fewer segments
    are interpolated onto the ten segment midpoints.

    Medians across groups can cancel when groups change in opposite directions,
    so ``generate_rolling_stats_segment_difference_figure`` shows paired gaps.

    Args:
        profiles: Stored profiles from ``RollingStatsSimilarity.details``.

    Returns:
        A two-panel Plotly figure with one visible column at a time.

    Raises:
        DataError: If no profiles are supplied.
    """
    columns = order_rolling_stats_columns(profiles)
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for column_index, column in enumerate(columns):
        column_profiles = [item for item in profiles if str(item["column"]) == column]
        for row, (statistic, label) in enumerate(_STATISTICS, start=1):
            _add_median_traces(figure, column_profiles, statistic, label, row, visible=column_index == 0)
    figure.update_layout(
        template="plotly_white",
        height=_FIGURE_HEIGHT,
        showlegend=False,
        margin={"l": 54, "r": 16, "t": 24, "b": 48},
    )
    figure.update_yaxes(title_text="Average", row=1, col=1)
    figure.update_yaxes(title_text="Spread", rangemode="tozero", row=2, col=1)
    figure.update_xaxes(range=[0, 100], **_POSITION_TICKS)
    figure.update_xaxes(title_text="Position in sequence", row=2, col=1)
    return figure


def generate_rolling_stats_segment_difference_figure(profiles: Sequence[Mapping[str, Any]]) -> go.Figure:
    """Plot the mean absolute paired difference in local average and local spread in each segment.

    Each group-column pair is compared with itself first, so groups that
    change in opposite directions cannot cancel. Only segments where both
    values of a pair are finite contribute. Bars shade from white to red as the
    difference grows, local spread bars are hatched, and the y-axis spans at
    least 0 to 1 training standard deviation so bar heights are comparable
    across reports. Trace names avoid dataset words so the report's dataset
    theming leaves their colors alone.

    Args:
        profiles: Stored profiles from ``RollingStatsSimilarity.details``.

    Returns:
        A Plotly grouped bar figure of mean absolute difference by segment.

    Raises:
        DataError: If no profiles are supplied.
    """
    if not profiles:
        raise DataError("At least one rolling statistics profile is required.")
    figure = go.Figure()
    top = 1.0
    for statistic, label in _STATISTICS:
        difference = np.abs(
            _segment_matrix(profiles, f"training_{statistic}") - _segment_matrix(profiles, f"synthetic_{statistic}")
        )
        support = np.sum(np.isfinite(difference), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_difference = np.nanmean(difference, axis=0)
        valid = support > 0
        top = max(top, float(np.max(mean_difference[valid], initial=0.0)) * 1.05)
        figure.add_trace(
            go.Bar(
                x=_POSITIONS[valid],
                y=mean_difference[valid],
                customdata=support[valid],
                marker={
                    "color": mean_difference[valid],
                    "colorscale": _DIFFERENCE_COLORSCALE,
                    "cmin": 0.0,
                    "cmax": _SEGMENT_DIFFERENCE_COLOR_MAX,
                    "line": {"color": "rgba(255,255,255,0.5)", "width": 1},
                    "pattern": {"shape": "/" if statistic == "spread" else ""},
                },
                name=label.removeprefix("Local ").capitalize(),
                hovertemplate=f"{label}<br>%{{x:.0f}}% through: %{{y:.2f}}<br>%{{customdata}} pairs<extra></extra>",
            )
        )
    figure.update_layout(
        template="plotly_white",
        height=_FIGURE_HEIGHT,
        barmode="group",
        showlegend=True,
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": 1.0, "yanchor": "bottom"},
        xaxis_title="Position in sequence",
        yaxis_title="Mean |difference| (training sd)",
        margin={"l": 54, "r": 16, "t": 40, "b": 48},
    )
    figure.update_xaxes(range=[0, 100], **_POSITION_TICKS)
    figure.update_yaxes(range=[0.0, top])
    return figure


def generate_rolling_stats_pair_score_figure(
    profiles: Sequence[Mapping[str, Any]],
    overall_score: float | None,
    *,
    max_columns: int = ROLLING_STATS_PAIR_SCORE_MAX_COLUMNS,
) -> go.Figure:
    """Plot group and column pair scores on the report's 0-10 scale.

    Only the ``max_columns`` lowest-scoring columns are plotted so each row
    stays readable. Each row sits at an integer y position with seeded
    vertical jitter, so long names are shortened only in the axis labels and
    shortened names cannot merge rows. Dots shade from white at a score of 10
    to red at 0. Traces are not named after columns, so the report's dataset
    theming cannot recolor a column named like a dataset.

    Args:
        profiles: Stored profiles from ``RollingStatsSimilarity.details``.
        overall_score: Aggregate metric score drawn as a reference line.
        max_columns: Largest number of value columns to plot.

    Returns:
        A Plotly strip figure with one marker trace per plotted value column.

    Raises:
        DataError: If no profiles are supplied.
    """
    columns = order_rolling_stats_columns(profiles)[:max_columns]
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
                    "colorscale": _DIFFERENCE_COLORSCALE,
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
        height=_FIGURE_HEIGHT,
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


def order_rolling_stats_columns(profiles: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return value columns ordered from lowest to highest mean pair similarity.

    Args:
        profiles: Stored profiles from ``RollingStatsSimilarity.details``.

    Returns:
        Column names, lowest mean pair similarity first, with names breaking ties.

    Raises:
        DataError: If no profiles are supplied.
    """
    if not profiles:
        raise DataError("At least one rolling statistics profile is required.")
    scores: dict[str, list[float]] = {}
    for item in profiles:
        scores.setdefault(str(item["column"]), []).append(float(item["similarity"]))
    return sorted(scores, key=lambda column: (float(np.mean(scores[column])), column))


def _add_median_traces(
    figure: go.Figure,
    profiles: Sequence[Mapping[str, Any]],
    statistic: str,
    label: str,
    row: int,
    *,
    visible: bool,
) -> None:
    """Add an interquartile band and a median line for training and synthetic data to one panel."""
    for dataset, line_color, band_color in (
        ("Training", _TRAINING_COLOR, _TRAINING_BAND_COLOR),
        ("Synthetic", _SYNTHETIC_COLOR, _SYNTHETIC_BAND_COLOR),
    ):
        positions, lower, median, upper = _segment_quantiles(profiles, f"{dataset.lower()}_{statistic}")
        figure.add_trace(
            go.Scatter(
                x=np.concatenate([positions, positions[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor=band_color,
                line={"width": 0},
                name=f"{dataset} interquartile range",
                hoverinfo="skip",
                showlegend=False,
                visible=visible,
            ),
            row=row,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=positions,
                y=median,
                mode="lines+markers",
                name=f"{dataset} median {label.lower()}",
                line={"color": line_color},
                hovertemplate=f"{dataset} median {label.lower()}<br>%{{x:.0f}}% through: %{{y:.2f}}<extra></extra>",
                visible=visible,
            ),
            row=row,
            col=1,
        )


def _segment_matrix(profiles: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    """Stack segment profiles onto the ten segment midpoints, NaN where a profile has no value."""
    matrix = np.full((len(profiles), ROLLING_STATS_SEGMENTS), np.nan)
    for row, item in enumerate(profiles):
        values = np.array([np.nan if value is None else value for value in item[key]], dtype=float)
        finite = np.isfinite(values)
        if len(values) == ROLLING_STATS_SEGMENTS:
            matrix[row] = values
        elif np.sum(finite) >= 2:
            positions = 100 * (np.arange(len(values)) + 0.5) / len(values)
            matrix[row] = np.interp(_POSITIONS, positions[finite], values[finite])
    return matrix


def _segment_quantiles(
    profiles: Sequence[Mapping[str, Any]],
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return segment midpoints with finite values and their 25th, 50th, and 75th percentiles."""
    matrix = _segment_matrix(profiles, key)
    valid = np.any(np.isfinite(matrix), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lower, median, upper = np.nanpercentile(matrix[:, valid], [25, 50, 75], axis=0)
    return _POSITIONS[valid], lower, median, upper
