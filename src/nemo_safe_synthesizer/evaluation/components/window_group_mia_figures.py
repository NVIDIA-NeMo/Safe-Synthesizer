# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plotly report figures for window membership inference protection.

Each builder summarizes values stored by the metric in ``WindowGroupMIA.details``.

Functions:
    generate_window_group_mia_distance_figure: Plot how close training and
        holdout windows come to the synthetic data, with the flag threshold.
    generate_window_group_mia_group_figure: Plot the share of each training
        group's windows that are flagged, for the most-flagged groups.
    generate_window_group_mia_closest_match_figure: Plot the training window
        closest to the synthetic data against its synthetic match, one column
        at a time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

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

WINDOW_GROUP_MIA_GROUP_MAX_ROWS = 12
WINDOW_GROUP_MIA_CLOSEST_TRACES_PER_COLUMN = 2


def generate_window_group_mia_distance_figure(details: Mapping[str, Any]) -> go.Figure:
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
        details: ``WindowGroupMIA.details`` with ``distances`` and, for a full
            evaluation, ``attack``.

    Returns:
        A Plotly overlaid histogram figure.

    Raises:
        DataError: If no training distances are stored.
    """
    distances = details.get("distances", {})
    training = np.asarray(distances.get("training", []), dtype=float)
    if training.size == 0:
        raise DataError("At least one training window distance is required.")
    holdout = np.asarray(distances.get("holdout", []), dtype=float)
    scale = float(np.sqrt(details["window_size"] * len(details["columns"])))
    combined = np.concatenate([training, holdout]) / scale
    top = max(float(np.quantile(combined, _DISTANCE_RANGE_QUANTILE)), 1e-6) * 1.05
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
    attack = details.get("attack")
    if attack is not None:
        figure.add_vline(
            x=attack["threshold"] / scale,
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
    per_group: Sequence[Mapping[str, Any]],
    holdout_flag_rate: float,
    *,
    max_rows: int = WINDOW_GROUP_MIA_GROUP_MAX_ROWS,
) -> go.Figure:
    """Plot the share of each training group's windows that are flagged, most-flagged groups first.

    Bars shade from white at the holdout flag rate, which is what chance
    alone produces, to red when every window of the group is flagged. The
    holdout flag rate is drawn as a dashed line, and the x-axis always spans 0
    to 100% so small excesses look small. Group names are shortened only in the
    axis labels; hover shows the full name, window count, and exact copies.

    Args:
        per_group: ``WindowGroupMIA.details["per_group"]`` from a full evaluation.
        holdout_flag_rate: Share of holdout windows flagged.
        max_rows: Largest number of groups to plot.

    Returns:
        A Plotly horizontal bar figure.

    Raises:
        DataError: If no group summaries with flag rates are supplied.
    """
    flagged = [item for item in per_group if "flag_rate" in item]
    if not flagged:
        raise DataError("At least one group summary with a flag rate is required.")
    rows = sorted(flagged, key=lambda item: (-float(item["flag_rate"]), str(item["group"])))[:max_rows]
    rows.reverse()
    rates = np.array([float(item["flag_rate"]) for item in rows])
    excess = np.clip((rates - holdout_flag_rate) / max(1.0 - holdout_flag_rate, 1e-12), 0.0, 1.0)
    labels = ["all rows" if item["group"] is None else str(item["group"]) for item in rows]
    figure = go.Figure(
        go.Bar(
            x=100 * rates,
            y=list(range(len(rows))),
            orientation="h",
            customdata=[
                [label, item["windows"], 100 * float(item["exact_copy_rate"])]
                for label, item in zip(labels, rows, strict=True)
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


def generate_window_group_mia_closest_match_figure(details: Mapping[str, Any]) -> go.Figure:
    """Plot the training window closest to the synthetic data against its synthetic match.

    Each value column contributes ``WINDOW_GROUP_MIA_CLOSEST_TRACES_PER_COLUMN``
    consecutive traces, training then synthetic, in ``details["columns"]``
    order, and only the first column is visible so the report can switch
    columns by trace index. Lines that lie on top of each other mean the
    synthetic data contains a copy of this stretch of training data.

    Args:
        details: ``WindowGroupMIA.details`` with ``closest_match`` and ``columns``.

    Returns:
        A Plotly line figure with one visible column at a time.

    Raises:
        DataError: If no closest match is stored.
    """
    match = details.get("closest_match")
    if not match:
        raise DataError("A closest training and synthetic window match is required.")
    steps = np.arange(1, details["window_size"] + 1)
    figure = go.Figure()
    for column_index, column in enumerate(details["columns"]):
        visible = column_index == 0
        for key, name, color, dash in (
            ("training_values", "Training window", _TRAINING_COLOR, "solid"),
            ("synthetic_values", "Synthetic window", _SYNTHETIC_COLOR, "dash"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=steps,
                    y=match[key][column_index],
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
    figure.update_xaxes(range=[0.5, details["window_size"] + 0.5])
    return figure
