# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from nemo_safe_synthesizer.errors import DataError, ParameterError
from nemo_safe_synthesizer.evaluation.components.autocorrelation_similarity_figures import (
    generate_autocorrelation_similarity_figure,
)


def test_autocorrelation_similarity_figure_uses_requested_lags_without_mutating_inputs():
    training_df = pd.Series(np.sin(np.arange(60) / 5))
    synthetic_df = training_df.shift(1).bfill()
    training_before = training_df.copy()
    synthetic_before = synthetic_df.copy()

    figure = generate_autocorrelation_similarity_figure(training_df, synthetic_df, max_lag=8)

    assert isinstance(figure, go.Figure)
    assert [trace.name for trace in figure.data] == ["Training ACF", "Synthetic ACF"]
    assert list(figure.data[0].x) == list(range(1, 9))
    assert figure.layout.yaxis.range == (-1.05, 1.05)
    pd.testing.assert_series_equal(training_df, training_before)
    pd.testing.assert_series_equal(synthetic_df, synthetic_before)


def test_autocorrelation_similarity_figure_preserves_non_finite_positions():
    training_df = pd.Series([0.0, 1.0, np.inf, 2.0, 3.0, 2.0, 1.0])
    synthetic_df = pd.Series([0.0, -np.inf, 1.0, 2.0, 3.0, 2.0, 1.0])

    figure = generate_autocorrelation_similarity_figure(training_df, synthetic_df)

    assert np.isfinite(figure.data[0].y).all()
    assert np.isfinite(figure.data[1].y).all()
    assert list(figure.data[0].y) != list(figure.data[1].y)


@pytest.mark.parametrize(
    ("training_df", "synthetic_df", "message"),
    [
        (pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0, 3.0]), "At least 4 finite points"),
        (pd.Series([1.0, 2.0, np.inf, np.nan]), pd.Series(range(4)), "At least 4 finite points"),
        (pd.Series([1.0] * 8), pd.Series(range(8)), "constant or near-constant"),
    ],
)
def test_autocorrelation_similarity_figure_rejects_unusable_series(training_df, synthetic_df, message):
    with pytest.raises(DataError, match=message):
        generate_autocorrelation_similarity_figure(training_df, synthetic_df)


def test_autocorrelation_similarity_figure_rejects_invalid_max_lag():
    values = pd.Series(range(8))

    with pytest.raises(ParameterError, match="max_lag must be at least 1"):
        generate_autocorrelation_similarity_figure(values, values, max_lag=0)
