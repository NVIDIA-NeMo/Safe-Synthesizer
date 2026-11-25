# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plotly report figures for window membership inference protection.

Each builder summarizes values exported in ``WindowMembershipInferenceDetails``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from ...config.evaluation_report_data import WindowGroupSummary, WindowMembershipInferenceDetails
from ...errors import DataError
from .autocorrelation_similarity_figures import shorten_column_label

_TRAINING_COLOR = "#3b82f6"
_SYNTHETIC_COLOR = "#f59e0b"
_HOLDOUT_COLOR = "#9ca3af"
# Matches the Correlation Stability difference heatmap in multi_modal_toggle.js.
_DIFFERENCE_COLORSCALE = [[0, "#ffffff"], [0.25, "#fee2e2"], [0.5, "#fca5a5"], [0.75, "#f87171"], [1, "#dc2626"]]
_FIGURE_HEIGHT = 340
_DISTANCE_BINS = 30
_DISTANCE_RANGE_QUANTILE = 0.99


def _finite(values: Sequence[float | None]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def window_value_count(details: WindowMembershipInferenceDetails) -> int:
    """Return the number of values in a window, the window size times the number of columns.

    Raises:
        DataError: If no window size is exported.
    """
    if details.window_size is None:
        raise DataError("A window size is required.")
    return details.window_size * len(details.columns)


def generate_window_group_mia_distance_figure(details: WindowMembershipInferenceDetails) -> go.Figure:
    """Plot the distribution of each window's distance to its nearest synthetic window.

    Distances are shown as the typical difference per value, in training
    standard deviations: the Euclidean distance divided by the square root of
    the number of values in a window. Training windows are drawn in training
    blue, holdout windows in gray, and the flag threshold as a dashed line.
    When training windows pile up to the left of the holdout windows, the
    synthetic data sits closer to training data than to unseen data. The holdout
    trace is not named after a dataset, so the report's dataset theming leaves
    its color alone.

    Args:
        details: Exported details with ``distances`` and, for a full
            evaluation, ``attack``.

    Returns:
        A Plotly overlaid histogram figure.

    Raises:
        DataError: If no training distances are stored.
    """
    distances = details.distances
    if distances is None or not distances.training:
        raise DataError("At least one training window distance is required.")
    training = _finite(distances.training)
    holdout = _finite(distances.holdout or [])
    scale = float(np.sqrt(window_value_count(details)))
    threshold = details.attack.threshold if details.attack is not None else None
    # Clip each distribution's tail separately, so many exact training copies
    # can't push the holdout windows or the threshold off the chart.
    limits = [
        float(np.quantile(values / scale, _DISTANCE_RANGE_QUANTILE)) for values in (training, holdout) if values.size
    ]
    if threshold is not None:
        limits.append(threshold / scale)
    top = max([*limits, 1e-6]) * 1.05
    bins = {"start": 0.0, "end": top, "size": top / _DISTANCE_BINS}
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=training / scale,
            xbins=bins,
            histnorm="percent",
            name="Training windows",
            marker={"color": _TRAINING_COLOR},
            opacity=0.65,
            hovertemplate="Training windows<br>difference %{x}<br>%{y:.1f}% of windows<extra></extra>",
        )
    )
    if holdout.size:
        figure.add_trace(
            go.Histogram(
                x=holdout / scale,
                xbins=bins,
                histnorm="percent",
                name="Holdout windows",
                marker={"color": _HOLDOUT_COLOR},
                opacity=0.65,
                hovertemplate="Holdout windows<br>difference %{x}<br>%{y:.1f}% of windows<extra></extra>",
            )
        )
    if threshold is not None:
        figure.add_vline(
            x=threshold / scale,
            line={"color": "rgba(255,255,255,0.6)", "dash": "dash", "width": 1},
            annotation_text="flag threshold",
            annotation_position="top",
        )
    figure.update_layout(
        template="plotly_white",
        height=_FIGURE_HEIGHT,
        barmode="overlay",
        showlegend=False,
        xaxis_title="Difference per value (training sd)",
        yaxis_title="Share of windows (%)",
        margin={"l": 54, "r": 16, "t": 40, "b": 48},
    )
    figure.update_xaxes(range=[0.0, top])
    return figure


def generate_window_group_mia_group_figure(
    per_group: Sequence[WindowGroupSummary], holdout_flag_rate: float
) -> go.Figure:
    """Plot the share of each training group's windows that are flagged, most flagged first, as exported.

    Bars shade from white at the holdout flag rate, which is what chance
    alone produces, to red when every window of the group is flagged. The
    holdout flag rate is drawn as a dashed line, and the x-axis always spans 0
    to 100% so small excesses look small. Group names are shortened only in the
    axis labels; hover shows the full name, window count, and exact copies.

    Args:
        per_group: Exported group summaries from a full evaluation.
        holdout_flag_rate: Share of holdout windows flagged.

    Returns:
        A Plotly horizontal bar figure.

    Raises:
        DataError: If no group summaries with flag rates are supplied.
    """
    flagged = [(item, item.flag_rate) for item in per_group if item.flag_rate is not None]
    if not flagged:
        raise DataError("At least one group summary with a flag rate is required.")
    rows = flagged[::-1]
    rates = np.array([rate for _, rate in rows])
    excess = np.clip((rates - holdout_flag_rate) / max(1.0 - holdout_flag_rate, 1e-12), 0.0, 1.0)
    labels = ["all rows" if item.group is None else str(item.group) for item, _ in rows]
    figure = go.Figure(
        go.Bar(
            x=100 * rates,
            y=list(range(len(rows))),
            orientation="h",
            customdata=[
                [label, item.windows, 100 * item.exact_copy_rate] for label, (item, _) in zip(labels, rows, strict=True)
            ],
            marker={
                "color": excess,
                "colorscale": _DIFFERENCE_COLORSCALE,
                "cmin": 0.0,
                "cmax": 1.0,
                "line": {"color": "rgba(255,255,255,0.5)", "width": 1},
            },
            name="Flagged share",
            hovertemplate=(
                "group %{customdata[0]}<br>%{x:.1f}% of %{customdata[1]} windows flagged"
                "<br>%{customdata[2]:.1f}% exact copies<extra></extra>"
            ),
        )
    )
    figure.add_vline(
        x=100 * holdout_flag_rate,
        line={"color": "rgba(255,255,255,0.6)", "dash": "dash", "width": 1},
        annotation_text="holdout",
        annotation_position="top",
    )
    figure.update_layout(
        template="plotly_white",
        height=_FIGURE_HEIGHT,
        showlegend=False,
        xaxis_title="Windows flagged (%)",
        margin={"l": 80, "r": 16, "t": 40, "b": 48},
    )
    figure.update_xaxes(range=[0, 100])
    figure.update_yaxes(
        range=[-0.6, len(rows) - 0.4],
        tickmode="array",
        tickvals=list(range(len(rows))),
        ticktext=[shorten_column_label(label) for label in labels],
        zeroline=False,
    )
    return figure


def generate_window_group_mia_closest_match_figure(details: WindowMembershipInferenceDetails) -> go.Figure:
    """Plot the synthetic window closest to the training data, with its training window for exact copies.

    The training window is plotted only when the match has its values,
    which the export has only for exact copies, so the report never
    shows training values that the synthetic data doesn't already contain.
    Each value column contributes two consecutive traces, training then
    synthetic, or only the synthetic trace when the training window is
    hidden. Columns follow ``details.columns`` order, and only the first
    column is visible so the report can switch columns by trace index.

    Args:
        details: Exported details with ``closest_match`` and ``columns``.

    Returns:
        A Plotly line figure with one visible column at a time.

    Raises:
        DataError: If no closest match is stored.
    """
    match = details.closest_match
    if match is None or details.window_size is None:
        raise DataError("A closest training and synthetic window match is required.")
    steps = np.arange(1, details.window_size + 1)
    figure = go.Figure()
    windows = [(match.synthetic_values, "Synthetic window", _SYNTHETIC_COLOR, "dash")]
    if match.training_values is not None:
        windows.insert(0, (match.training_values, "Training window", _TRAINING_COLOR, "solid"))
    for column_index, column in enumerate(details.columns):
        visible = column_index == 0
        for values, name, color, dash in windows:
            figure.add_trace(
                go.Scatter(
                    x=steps,
                    y=values[column_index],
                    mode="lines+markers",
                    name=name,
                    line={"color": color, "dash": dash},
                    hovertemplate=f"{name}<br>{column}<br>step %{{x}}: %{{y:.3g}}<extra></extra>",
                    visible=visible,
                )
            )
    figure.update_layout(
        template="plotly_white",
        height=_FIGURE_HEIGHT,
        showlegend=False,
        xaxis_title="Step in window",
        yaxis_title="Value",
        margin={"l": 54, "r": 16, "t": 40, "b": 48},
    )
    figure.update_xaxes(range=[0.5, details.window_size + 0.5])
    return figure
