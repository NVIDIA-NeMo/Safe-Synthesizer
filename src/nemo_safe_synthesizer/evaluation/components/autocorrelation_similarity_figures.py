# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plotly diagnostics for autocorrelation similarity.

The public builder converts two numeric sequences into comparable positive-lag
autocorrelation profiles. It performs the same length, variance, and lag
validation as the metric-facing diagnostic path while leaving caller-owned
Series objects unchanged.

Functions:
    generate_autocorrelation_similarity_figure: Build a training-versus-
        synthetic autocorrelation profile figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ...errors import DataError, ParameterError
from .autocorrelation_similarity import AutocorrelationSimilarity

_TRAINING_COLOR = "#3C2ED1"
_SYNTHETIC_COLOR = "#1AA2E6"


def generate_autocorrelation_similarity_figure(
    training: pd.Series,
    synthetic: pd.Series,
    *,
    max_lag: int = 20,
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
    training_count = int(np.count_nonzero(np.isfinite(training_values)))
    synthetic_count = int(np.count_nonzero(np.isfinite(synthetic_values)))
    n = min(training_count, synthetic_count)
    if n < 4:
        raise DataError("At least 4 finite points are required in each series.")
    if np.nanstd(training_values) <= 1e-12 or np.nanstd(synthetic_values) <= 1e-12:
        raise DataError("Autocorrelation is unavailable for constant or near-constant series.")

    # Cap the lag so every plotted correlation retains at least half of the
    # shorter sequence as overlapping observations.
    effective_max_lag = min(max_lag, (n - 1) // 2)
    if effective_max_lag < 1:
        raise DataError("The series are too short to compute a stable lag profile.")
    lags = np.arange(1, effective_max_lag + 1)
    training_acf = AutocorrelationSimilarity._acf_vector(training_values, effective_max_lag)
    synthetic_acf = AutocorrelationSimilarity._acf_vector(synthetic_values, effective_max_lag)
    if not np.any(np.isfinite(training_acf) & np.isfinite(synthetic_acf)):
        raise DataError("The series have no lags with sufficient pair support.")

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
