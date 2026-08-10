# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

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
    assert [trace.name for trace in figure.data] == ["Real ACF", "Synthetic ACF"]
    assert list(figure.data[0].x) == list(range(1, 9))
    assert figure.layout.yaxis.range == (-1.05, 1.05)
    pd.testing.assert_series_equal(training_df, training_before)
    pd.testing.assert_series_equal(synthetic_df, synthetic_before)


@pytest.mark.parametrize(
    ("training_df", "synthetic_df", "message"),
    [
        (pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0, 3.0]), "At least 4 finite points"),
        (pd.Series([1.0] * 8), pd.Series(range(8)), "constant or near-constant"),
    ],
)
def test_autocorrelation_similarity_figure_rejects_unusable_series(training_df, synthetic_df, message):
    with pytest.raises(ValueError, match=message):
        generate_autocorrelation_similarity_figure(training_df, synthetic_df)
