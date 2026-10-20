# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plotly report figures for spectral similarity.

Each builder summarizes the per-pair band profiles stored by the metric in
``SpectralSimilarity.details``. Pairs of different lengths can cover different
bands, so every figure places bands by their cycle length.

Functions:
    generate_spectral_band_figure: Plot median shares of variation by cycle
        length with interquartile bands across groups, one column at a time.
    generate_spectral_band_difference_figure: Plot the mean paired difference
        in share of variation for each cycle-length band.
    generate_spectral_pair_score_figure: Plot group and column pair scores
        for the lowest-scoring columns against the overall score.
    order_spectral_columns: Order value columns by mean pair score.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from ...errors import DataError
from .autocorrelation_similarity_figures import shorten_column_label

_TRAINING_COLOR = "#3b82f6"
_SYNTHETIC_COLOR = "#f59e0b"
_TRAINING_BAND_COLOR = "rgba(59, 130, 246, 0.2)"
_SYNTHETIC_BAND_COLOR = "rgba(245, 158, 11, 0.2)"
# Matches the Correlation Stability difference heatmap in multi_modal_toggle.js.
_DIFFERENCE_COLORSCALE = [[0, "#ffffff"], [0.25, "#fee2e2"], [0.5, "#fca5a5"], [0.75, "#f87171"], [1, "#dc2626"]]
# Differences are in percentage points of variation; bars reach full red at this gap.
_BAND_DIFFERENCE_COLOR_MAX = 25.0
_PAIR_SCORE_JITTER = 0.25
_FIGURE_HEIGHT = 340
_CYCLE_AXIS_TITLE = "Cycle length (observations)"

SPECTRAL_SUMMARY_TRACES_PER_COLUMN = 4
SPECTRAL_PAIR_SCORE_MAX_COLUMNS = 8


def generate_spectral_band_figure(profiles: Sequence[Mapping[str, Any]]) -> go.Figure:
    """Plot typical training and synthetic shares of variation in each cycle-length band.

    Each value column gets a median line and an interquartile band for both
    datasets, in percent of the sequence's variation. Columns contribute
    ``SPECTRAL_SUMMARY_TRACES_PER_COLUMN`` consecutive traces in
    ``order_spectral_columns`` order, and only the first column is visible so
    the report can switch columns by trace index. Pairs with a constant
    synthetic sequence have no synthetic shares and only add to the training
    traces.

    Medians across groups can cancel when groups put their variation in
    different bands, so ``generate_spectral_band_difference_figure`` shows
    paired gaps.

    Args:
        profiles: Stored profiles from ``SpectralSimilarity.details``.

    Returns:
        A Plotly figure with one visible column at a time.

    Raises:
        DataError: If no profiles are supplied.
    """
    columns = order_spectral_columns(profiles)
    cycle_lengths = _cycle_lengths(profiles)
    labels = _cycle_labels(cycle_lengths)
    figure = go.Figure()
    for column_index, column in enumerate(columns):
        column_profiles = [item for item in profiles if str(item["column"]) == column]
        _add_median_traces(figure, column_profiles, cycle_lengths, labels, visible=column_index == 0)
    figure.update_layout(
        template="plotly_white",
        height=_FIGURE_HEIGHT,
        showlegend=False,
        xaxis_title=_CYCLE_AXIS_TITLE,
        yaxis_title="Share of variation (%)",
        margin={"l": 54, "r": 16, "t": 24, "b": 48},
    )
    _update_cycle_axis(figure, cycle_lengths, labels)
    figure.update_yaxes(rangemode="tozero")
    return figure


def generate_spectral_band_difference_figure(profiles: Sequence[Mapping[str, Any]]) -> go.Figure:
    """Plot the mean absolute paired difference in share of variation for each cycle-length band.

    Each group-column pair is compared with itself first, so groups that put
    their variation in different bands cannot cancel. Only bands where both
    shares of a pair are known contribute. Bars shade from white to red as the
    difference grows, and the y-axis spans at least 0 to 25 percentage points
    so bar heights are comparable across reports. The trace name avoids
    dataset words so the report's dataset theming leaves its colors alone.

    Args:
        profiles: Stored profiles from ``SpectralSimilarity.details``.

    Returns:
        A Plotly bar figure of mean absolute difference by band.

    Raises:
        DataError: If no profiles are supplied.
    """
    if not profiles:
        raise DataError("At least one spectral profile is required.")
    cycle_lengths = _cycle_lengths(profiles)
    labels = np.array(_cycle_labels(cycle_lengths))
    difference = np.abs(
        _band_matrix(profiles, "training_bands", cycle_lengths)
        - _band_matrix(profiles, "synthetic_bands", cycle_lengths)
    )
    support = np.sum(np.isfinite(difference), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_difference = np.nanmean(difference, axis=0)
    valid = support > 0
    figure = go.Figure(
        go.Bar(
            x=labels[valid],
            y=mean_difference[valid],
            customdata=support[valid],
            marker={
                "color": mean_difference[valid],
                "colorscale": _DIFFERENCE_COLORSCALE,
                "cmin": 0.0,
                "cmax": _BAND_DIFFERENCE_COLOR_MAX,
                "line": {"color": "rgba(255,255,255,0.5)", "width": 1},
            },
            name="Mean difference",
            hovertemplate=("Cycles of about %{x} observations: %{y:.1f} points<br>%{customdata} pairs<extra></extra>"),
        )
    )
    top = max(_BAND_DIFFERENCE_COLOR_MAX, float(np.max(mean_difference[valid], initial=0.0)) * 1.05)
    figure.update_layout(
        template="plotly_white",
        height=_FIGURE_HEIGHT,
        showlegend=False,
        xaxis_title=_CYCLE_AXIS_TITLE,
        yaxis_title="Mean |difference| (points)",
        margin={"l": 54, "r": 16, "t": 24, "b": 48},
    )
    _update_cycle_axis(figure, cycle_lengths, list(labels))
    figure.update_yaxes(range=[0.0, top])
    return figure


def generate_spectral_pair_score_figure(
    profiles: Sequence[Mapping[str, Any]],
    overall_score: float | None,
    *,
    max_columns: int = SPECTRAL_PAIR_SCORE_MAX_COLUMNS,
) -> go.Figure:
    """Plot group and column pair scores on the report's 0-10 scale.

    Only the ``max_columns`` lowest-scoring columns are plotted so each row
    stays readable. Each row sits at an integer y position with seeded
    vertical jitter, so long names are shortened only in the axis labels and
    shortened names cannot merge rows. Dots shade from white at a score of 10
    to red at 0. Traces are not named after columns, so the report's dataset
    theming cannot recolor a column named like a dataset.

    Args:
        profiles: Stored profiles from ``SpectralSimilarity.details``.
        overall_score: Aggregate metric score drawn as a reference line.
        max_columns: Largest number of value columns to plot.

    Returns:
        A Plotly strip figure with one marker trace per plotted value column.

    Raises:
        DataError: If no profiles are supplied.
    """
    columns = order_spectral_columns(profiles)[:max_columns]
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


def order_spectral_columns(profiles: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return value columns ordered from lowest to highest mean pair similarity.

    Args:
        profiles: Stored profiles from ``SpectralSimilarity.details``.

    Returns:
        Column names, lowest mean pair similarity first, with names breaking ties.

    Raises:
        DataError: If no profiles are supplied.
    """
    if not profiles:
        raise DataError("At least one spectral profile is required.")
    scores: dict[str, list[float]] = {}
    for item in profiles:
        scores.setdefault(str(item["column"]), []).append(float(item["similarity"]))
    return sorted(scores, key=lambda column: (float(np.mean(scores[column])), column))


def _cycle_lengths(profiles: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Return every band's cycle length across profiles, shortest first."""
    lengths: set[float] = set()
    for item in profiles:
        lengths.update(round(float(value), 6) for value in item["band_cycle_lengths"])
    return np.array(sorted(lengths))


def _cycle_labels(cycle_lengths: np.ndarray) -> list[str]:
    """Label each band by its cycle length in whole observations."""
    return [f"{value:.0f}" for value in cycle_lengths]


def _update_cycle_axis(figure: go.Figure, cycle_lengths: np.ndarray, labels: list[str]) -> None:
    """Place bands in cycle-length order and label only whole octaves, so labels stay horizontal."""
    octaves = [label for label, value in zip(labels, cycle_lengths, strict=True) if float(np.log2(value)).is_integer()]
    figure.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=labels,
        tickmode="array",
        tickvals=octaves,
        tickangle=0,
    )


def _band_matrix(profiles: Sequence[Mapping[str, Any]], key: str, cycle_lengths: np.ndarray) -> np.ndarray:
    """Stack band shares, in percent, onto shared cycle lengths, NaN where a profile has no value."""
    matrix = np.full((len(profiles), len(cycle_lengths)), np.nan)
    positions = {value: index for index, value in enumerate(cycle_lengths.tolist())}
    for row, item in enumerate(profiles):
        for length, share in zip(item["band_cycle_lengths"], item[key], strict=True):
            if share is not None:
                matrix[row, positions[round(float(length), 6)]] = 100 * float(share)
    return matrix


def _add_median_traces(
    figure: go.Figure,
    profiles: Sequence[Mapping[str, Any]],
    cycle_lengths: np.ndarray,
    labels: list[str],
    *,
    visible: bool,
) -> None:
    """Add an interquartile band and a median line for training and synthetic data."""
    for dataset, line_color, band_color in (
        ("Training", _TRAINING_COLOR, _TRAINING_BAND_COLOR),
        ("Synthetic", _SYNTHETIC_COLOR, _SYNTHETIC_BAND_COLOR),
    ):
        matrix = _band_matrix(profiles, f"{dataset.lower()}_bands", cycle_lengths)
        valid = np.any(np.isfinite(matrix), axis=0)
        x = [label for label, keep in zip(labels, valid, strict=True) if keep]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lower, median, upper = np.nanpercentile(matrix[:, valid], [25, 50, 75], axis=0)
        figure.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor=band_color,
                line={"width": 0},
                name=f"{dataset} interquartile range",
                hoverinfo="skip",
                showlegend=False,
                visible=visible,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=median,
                mode="lines+markers",
                name=f"{dataset} median share",
                line={"color": line_color},
                hovertemplate=(
                    f"{dataset} median share<br>Cycles of about %{{x}} observations: %{{y:.1f}}%<extra></extra>"
                ),
                visible=visible,
            )
        )
