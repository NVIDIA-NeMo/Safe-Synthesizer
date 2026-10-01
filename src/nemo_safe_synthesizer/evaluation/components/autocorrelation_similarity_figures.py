# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plotly diagnostics for autocorrelation similarity.

The public builder converts two numeric sequences into comparable positive-lag
autocorrelation profiles. It performs the same length, variance, and lag
validation as the metric-facing diagnostic path while leaving caller-owned
Series objects unchanged.

Functions:
    generate_autocorrelation_profile_figure: Plot already-computed profiles.
    generate_autocorrelation_similarity_figure: Build a training-versus-
        synthetic autocorrelation profile figure.
    generate_autocorrelation_summary_figure: Plot median profiles and
        interquartile bands across groups, one column at a time.
    generate_autocorrelation_lag_error_figure: Plot the mean paired profile
        difference at each lag.
    generate_autocorrelation_pair_score_figure: Plot every group and column
        pair score against the overall score.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ...config.evaluate import DEFAULT_AUTOCORRELATION_MAX_LAG, DEFAULT_AUTOCORRELATION_MIN_POINTS
from ...errors import DataError, ParameterError
from .autocorrelation_similarity import AutocorrelationSimilarity

_TRAINING_COLOR = "#3C2ED1"
_SYNTHETIC_COLOR = "#1AA2E6"
_TRAINING_BAND_COLOR = "rgba(59, 130, 246, 0.2)"
_SYNTHETIC_BAND_COLOR = "rgba(245, 158, 11, 0.2)"
_SUMMARY_COLOR = "#76B900"
_SUMMARY_FIGURE_HEIGHT = 300


def generate_autocorrelation_similarity_figure(
    training: pd.Series,
    synthetic: pd.Series,
    *,
    max_lag: int = DEFAULT_AUTOCORRELATION_MAX_LAG,
) -> go.Figure:
    """Build a figure comparing training and synthetic lag profiles.

    Values that cannot be converted to numbers and non-finite values remain as
    gaps in their original temporal positions. Each lag uses only pairs with
    two finite endpoints. The shorter finite sequence determines the largest
    stable lag. The function does not mutate either Series.

    Args:
        training: Training values in temporal order.
        synthetic: Synthetic values in temporal order.
        max_lag: Largest positive lag requested for the diagnostic.

    Returns:
        A Plotly figure containing the two autocorrelation profiles.

    Raises:
        ParameterError: If ``max_lag`` is below one.
        DataError: If either input has fewer than four finite values, is
            effectively constant, or has no stable positive lag.
    """
    if max_lag < 1:
        raise ParameterError("max_lag must be at least 1.")

    training_values = AutocorrelationSimilarity._prepare_values(training)
    synthetic_values = AutocorrelationSimilarity._prepare_values(synthetic)
    training_count = int(np.sum(np.isfinite(training_values)))
    synthetic_count = int(np.sum(np.isfinite(synthetic_values)))
    n = min(training_count, synthetic_count)
    if n < DEFAULT_AUTOCORRELATION_MIN_POINTS:
        raise DataError(f"At least {DEFAULT_AUTOCORRELATION_MIN_POINTS} finite points are required in each series.")
    if AutocorrelationSimilarity._is_effectively_constant(
        training_values
    ) or AutocorrelationSimilarity._is_effectively_constant(synthetic_values):
        raise DataError("Autocorrelation is unavailable for constant or near-constant series.")

    # Cap the lag so every plotted correlation retains at least half of the
    # shorter sequence as overlapping observations.
    effective_max_lag = min(max_lag, (n - 1) // 2)
    if effective_max_lag < 1:
        raise DataError("The series are too short to compute a stable lag profile.")
    lags = np.arange(1, effective_max_lag + 1)
    training_acf, _ = AutocorrelationSimilarity._acf_profile(training_values, effective_max_lag)
    synthetic_acf, _ = AutocorrelationSimilarity._acf_profile(synthetic_values, effective_max_lag)
    if not np.any(np.isfinite(training_acf) & np.isfinite(synthetic_acf)):
        raise DataError("The series have no lags with sufficient pair support.")

    return generate_autocorrelation_profile_figure(lags, training_acf, synthetic_acf)


def generate_autocorrelation_profile_figure(
    lags: Sequence[int] | np.ndarray,
    training_acf: Sequence[float | None] | np.ndarray,
    synthetic_acf: Sequence[float | None] | np.ndarray,
) -> go.Figure:
    """Build a figure from profiles already computed by the metric.

    Args:
        lags: Positive lag values shared by both profiles.
        training_acf: Training autocorrelation values.
        synthetic_acf: Synthetic autocorrelation values.

    Returns:
        A Plotly figure containing the supplied profiles.

    Raises:
        DataError: If the lag and profile lengths differ or are empty.
    """
    if len(lags) == 0 or len(lags) != len(training_acf) or len(lags) != len(synthetic_acf):
        raise DataError("Autocorrelation profile lags and values must have the same nonzero length.")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=lags,
            y=training_acf,
            mode="lines+markers",
            name="Training ACF",
            line={"color": _TRAINING_COLOR},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=lags,
            y=synthetic_acf,
            mode="lines+markers",
            name="Synthetic ACF",
            line={"color": _SYNTHETIC_COLOR},
        )
    )
    figure.update_layout(
        template="plotly_white",
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 60, "r": 20, "t": 45, "b": 55},
    )
    # A fixed theoretical range makes separate diagnostic figures visually
    # comparable instead of rescaling each result around its observed values.
    figure.update_yaxes(range=[-1.05, 1.05])
    return figure


def generate_autocorrelation_summary_figure(profiles: Sequence[Mapping[str, Any]]) -> go.Figure:
    """Plot typical training and synthetic profiles across groups.

    Each value column gets a median profile and an interquartile band for both
    datasets. A dropdown switches between columns, starting with the column
    whose mean pair similarity is lowest.

    Args:
        profiles: Stored profiles from ``AutocorrelationSimilarity.details``.

    Returns:
        A Plotly figure with one visible column at a time.

    Raises:
        DataError: If no profiles are supplied.
    """
    columns = _columns_by_similarity(profiles)
    figure = go.Figure()
    traces_per_column = 4
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
    if len(columns) > 1:
        buttons = []
        for column_index, column in enumerate(columns):
            visibility = [False] * (len(columns) * traces_per_column)
            start = column_index * traces_per_column
            visibility[start : start + traces_per_column] = [True] * traces_per_column
            buttons.append({"label": str(column), "method": "update", "args": [{"visible": visibility}]})
        figure.update_layout(
            updatemenus=[
                {
                    "buttons": buttons,
                    "direction": "down",
                    "x": 1.0,
                    "xanchor": "right",
                    "y": 1.18,
                    "yanchor": "top",
                    "bgcolor": "#292929",
                    "bordercolor": "#666666",
                    "font": {"color": "rgba(255,255,255,0.85)"},
                    "showactive": True,
                }
            ]
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
) -> go.Figure:
    """Plot every group and column pair score on the report's 0-10 scale.

    Args:
        profiles: Stored profiles from ``AutocorrelationSimilarity.details``.
        overall_score: Aggregate metric score drawn as a reference line.

    Returns:
        A Plotly strip figure with one row per value column.

    Raises:
        DataError: If no profiles are supplied.
    """
    columns = _columns_by_similarity(profiles)
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
                hovertemplate="group %{customdata}<br>score %{x:.1f}<extra></extra>",
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
        margin={"l": 90, "r": 16, "t": 40, "b": 48},
    )
    figure.update_xaxes(range=[-0.25, 10.25])
    figure.update_yaxes(categoryorder="array", categoryarray=list(reversed(columns)))
    return figure


def _columns_by_similarity(profiles: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return value columns ordered from lowest to highest mean pair similarity."""
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
