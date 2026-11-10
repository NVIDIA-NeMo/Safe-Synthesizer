# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plotly report figures for rolling statistics similarity.

Each builder summarizes the per-pair segment profiles stored by the metric in
``RollingStatsSimilarity.details``.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...errors import DataError
from .time_series_common_figures import DIFFERENCE_COLORSCALE, order_columns_by_score

ROLLING_STATS_SEGMENTS = 10
"""Number of aligned segments each sequence's rolling profiles are averaged over."""

_TRAINING_COLOR = "#3b82f6"
_SYNTHETIC_COLOR = "#f59e0b"
_TRAINING_BAND_COLOR = "rgba(59, 130, 246, 0.2)"
_SYNTHETIC_BAND_COLOR = "rgba(245, 158, 11, 0.2)"
_SEGMENT_DIFFERENCE_COLOR_MAX = 1.0
_POSITIONS = 100 * (np.arange(ROLLING_STATS_SEGMENTS) + 0.5) / ROLLING_STATS_SEGMENTS
_STATISTICS = (("level", "Local average"), ("spread", "Local spread"))
_POSITION_TICKS = {"tickmode": "array", "tickvals": [0, 50, 100], "ticktext": ["Start", "Middle", "End"]}

ROLLING_STATS_FIGURE_HEIGHT = 340
ROLLING_STATS_SUMMARY_TRACES_PER_COLUMN = 8


def generate_rolling_stats_profile_figure(profiles: Sequence[Mapping[str, Any]]) -> go.Figure:
    """Plot typical training and synthetic local averages and spreads from start to end.

    The top panel shows the local average, in training standard deviations
    from the training mean, and the bottom panel the local spread, in training
    standard deviations. Each value column gets a median profile and an
    interquartile band for both datasets in both panels. Columns contribute
    ``ROLLING_STATS_SUMMARY_TRACES_PER_COLUMN`` consecutive traces in
    ``order_columns_by_score`` order, and only the first column is visible
    so the report can switch columns by trace index. Pairs with fewer segments
    are interpolated onto the ten segment midpoints. Pairs whose synthetic
    values are constant or mostly missing have no synthetic profile, so both
    medians leave them out and compare the same pairs.

    Medians across groups can cancel when groups change in opposite directions,
    so ``generate_rolling_stats_segment_difference_figure`` shows paired gaps.

    Args:
        profiles: Stored profiles from ``RollingStatsSimilarity.details``.

    Returns:
        A two-panel Plotly figure with one visible column at a time.

    Raises:
        DataError: If no profiles are supplied.
    """
    columns = order_columns_by_score(profiles)
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for column_index, column in enumerate(columns):
        column_profiles = [
            item
            for item in profiles
            if str(item["column"]) == column and any(value is not None for value in item["synthetic_level"])
        ]
        for row, (statistic, label) in enumerate(_STATISTICS, start=1):
            _add_median_traces(figure, column_profiles, statistic, label, row, visible=column_index == 0)
    figure.update_layout(
        template="plotly_white",
        height=ROLLING_STATS_FIGURE_HEIGHT,
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
                    "colorscale": DIFFERENCE_COLORSCALE,
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
        height=ROLLING_STATS_FIGURE_HEIGHT,
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
    if not profiles:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty
    matrix = _segment_matrix(profiles, key)
    valid = np.any(np.isfinite(matrix), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lower, median, upper = np.nanpercentile(matrix[:, valid], [25, 50, 75], axis=0)
    return _POSITIONS[valid], lower, median, upper
