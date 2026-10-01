# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plotly report figures for autocorrelation similarity.

Each builder summarizes the per-pair profiles stored by the metric in
``AutocorrelationSimilarity.details``.

Functions:
    generate_autocorrelation_summary_figure: Plot median profiles and
        interquartile bands across groups, one column at a time.
    generate_autocorrelation_lag_error_figure: Plot the mean paired profile
        difference at each lag.
    generate_autocorrelation_pair_score_figure: Plot group and column pair
        scores for the lowest-scoring columns against the overall score.
    order_autocorrelation_columns: Order value columns by mean pair score.
    shorten_column_label: Shorten long column names for axis labels.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from ...errors import DataError

_TRAINING_COLOR = "#3C2ED1"
_SYNTHETIC_COLOR = "#1AA2E6"
_TRAINING_BAND_COLOR = "rgba(59, 130, 246, 0.2)"
_SYNTHETIC_BAND_COLOR = "rgba(245, 158, 11, 0.2)"
_SUMMARY_COLOR = "#76B900"
_SUMMARY_FIGURE_HEIGHT = 300
_LABEL_MAX_LENGTH = 10
_LABEL_EDGE_LENGTH = 3

AUTOCORRELATION_SUMMARY_TRACES_PER_COLUMN = 4
AUTOCORRELATION_PAIR_SCORE_MAX_COLUMNS = 8


def generate_autocorrelation_summary_figure(profiles: Sequence[Mapping[str, Any]]) -> go.Figure:
    """Plot typical training and synthetic profiles across groups.

    Each value column gets a median profile and an interquartile band for both
    datasets. Columns contribute ``AUTOCORRELATION_SUMMARY_TRACES_PER_COLUMN``
    consecutive traces in ``order_autocorrelation_columns`` order, and only the
    first column is visible so the report can switch columns by trace index.

    Args:
        profiles: Stored profiles from ``AutocorrelationSimilarity.details``.

    Returns:
        A Plotly figure with one visible column at a time.

    Raises:
        DataError: If no profiles are supplied.
    """
    columns = order_autocorrelation_columns(profiles)
    figure = go.Figure()
    for column_index, column in enumerate(columns):
        column_profiles = [item for item in profiles if item["column"] == column]
        visible = column_index == 0
        for key, name, line_color, band_color in (
            ("training_acf", "Training", _TRAINING_COLOR, _TRAINING_BAND_COLOR),
            ("synthetic_acf", "Synthetic", _SYNTHETIC_COLOR, _SYNTHETIC_BAND_COLOR),
        ):
            lags, lower, median, upper = _profile_quantiles(column_profiles, key)
            figure.add_trace(
                go.Scatter(
                    x=np.concatenate([lags, lags[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor=band_color,
                    line={"width": 0},
                    name=f"{name} interquartile range",
                    hoverinfo="skip",
                    showlegend=False,
                    visible=visible,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=lags,
                    y=median,
                    mode="lines+markers",
                    name=f"{name} median ACF",
                    line={"color": line_color},
                    hovertemplate=f"{name} median<br>lag %{{x}}: %{{y:.3f}}<extra></extra>",
                    visible=visible,
                )
            )
    figure.update_layout(
        template="plotly_white",
        height=_SUMMARY_FIGURE_HEIGHT,
        showlegend=False,
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
        margin={"l": 54, "r": 16, "t": 40, "b": 48},
    )
    figure.update_yaxes(range=[-1.05, 1.05])
    return figure


def generate_autocorrelation_lag_error_figure(profiles: Sequence[Mapping[str, Any]]) -> go.Figure:
    """Plot the mean absolute paired profile difference at each lag.

    Only lags where both profiles of a pair are finite contribute, matching
    the lags used by the pair score.

    Args:
        profiles: Stored profiles from ``AutocorrelationSimilarity.details``.

    Returns:
        A Plotly bar figure of mean absolute difference by lag.

    Raises:
        DataError: If no profiles are supplied.
    """
    if not profiles:
        raise DataError("At least one autocorrelation profile is required.")
    training = _profile_matrix(profiles, "training_acf")
    synthetic = _profile_matrix(profiles, "synthetic_acf")
    difference = np.abs(training - synthetic)
    support = np.sum(np.isfinite(difference), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_difference = np.nanmean(difference, axis=0)
    lags = np.arange(1, difference.shape[1] + 1)
    valid = support > 0
    figure = go.Figure(
        go.Bar(
            x=lags[valid],
            y=mean_difference[valid],
            customdata=support[valid],
            marker={"color": _SUMMARY_COLOR},
            name="Mean absolute difference",
            hovertemplate="lag %{x}: %{y:.3f}<br>%{customdata} pairs<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=_SUMMARY_FIGURE_HEIGHT,
        showlegend=False,
        xaxis_title="Lag",
        yaxis_title="Mean |ACF difference|",
        margin={"l": 54, "r": 16, "t": 40, "b": 48},
    )
    figure.update_yaxes(rangemode="tozero")
    return figure


def generate_autocorrelation_pair_score_figure(
    profiles: Sequence[Mapping[str, Any]],
    overall_score: float | None,
    *,
    max_columns: int = AUTOCORRELATION_PAIR_SCORE_MAX_COLUMNS,
) -> go.Figure:
    """Plot group and column pair scores on the report's 0-10 scale.

    Only the ``max_columns`` lowest-scoring columns are plotted so each row
    stays readable. Rows are keyed by full column name, and long names are
    shortened only in the axis labels so shortened names cannot merge rows.

    Args:
        profiles: Stored profiles from ``AutocorrelationSimilarity.details``.
        overall_score: Aggregate metric score drawn as a reference line.
        max_columns: Largest number of value columns to plot.

    Returns:
        A Plotly strip figure with one row per plotted value column.

    Raises:
        DataError: If no profiles are supplied.
    """
    columns = order_autocorrelation_columns(profiles)[:max_columns]
    figure = go.Figure()
    for column in columns:
        column_profiles = [item for item in profiles if item["column"] == column]
        groups = ["all rows" if item["group"] is None else str(item["group"]) for item in column_profiles]
        figure.add_trace(
            go.Box(
                x=[10 * float(item["similarity"]) for item in column_profiles],
                y=[str(column)] * len(column_profiles),
                customdata=groups,
                orientation="h",
                boxpoints="all",
                jitter=0.5,
                pointpos=0,
                fillcolor="rgba(0,0,0,0)",
                line={"color": "rgba(0,0,0,0)"},
                marker={"color": _SUMMARY_COLOR, "size": 6, "opacity": 0.75},
                name=str(column),
                hoveron="points",
                hovertemplate="%{y}<br>group %{customdata}<br>score %{x:.1f}<extra></extra>",
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
        height=_SUMMARY_FIGURE_HEIGHT,
        showlegend=False,
        xaxis_title="Pair score",
        margin={"l": 80, "r": 16, "t": 40, "b": 48},
    )
    figure.update_xaxes(range=[-0.25, 10.25])
    rows = [str(column) for column in reversed(columns)]
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=rows,
        tickmode="array",
        tickvals=rows,
        ticktext=[shorten_column_label(row) for row in rows],
    )
    return figure


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


def order_autocorrelation_columns(profiles: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return value columns ordered from lowest to highest mean pair similarity.

    Args:
        profiles: Stored profiles from ``AutocorrelationSimilarity.details``.

    Returns:
        Column names, lowest mean pair similarity first, with names breaking ties.

    Raises:
        DataError: If no profiles are supplied.
    """
    if not profiles:
        raise DataError("At least one autocorrelation profile is required.")
    scores: dict[str, list[float]] = {}
    for item in profiles:
        scores.setdefault(str(item["column"]), []).append(float(item["similarity"]))
    return sorted(scores, key=lambda column: (float(np.mean(scores[column])), column))


def _profile_matrix(profiles: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    """Stack profiles of different lengths into a NaN-padded lag matrix."""
    width = max(len(item["lags"]) for item in profiles)
    matrix = np.full((len(profiles), width), np.nan)
    for row, item in enumerate(profiles):
        values = np.array([np.nan if value is None else value for value in item[key]], dtype=float)
        matrix[row, : len(values)] = values
    return matrix


def _profile_quantiles(
    profiles: Sequence[Mapping[str, Any]],
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return lags with finite values and their 25th, 50th, and 75th percentiles."""
    matrix = _profile_matrix(profiles, key)
    valid = np.any(np.isfinite(matrix), axis=0)
    lags = np.arange(1, matrix.shape[1] + 1)[valid]
    lower, median, upper = np.nanpercentile(matrix[:, valid], [25, 50, 75], axis=0)
    return lags, lower, median, upper
