# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from nemo_safe_synthesizer.evaluation.components.autocorrelation_similarity_figures import (
    generate_autocorrelation_similarity_figure,
)


def test_autocorrelation_figure_uses_requested_lags_without_mutating_inputs():
    reference = pd.Series(np.sin(np.arange(60) / 5))
    synthetic = reference.shift(1).bfill()
    reference_before = reference.copy()
    synthetic_before = synthetic.copy()

    figure = generate_autocorrelation_similarity_figure(reference, synthetic, max_lag=8)

    assert isinstance(figure, go.Figure)
    assert [trace.name for trace in figure.data] == ["Real ACF", "Synthetic ACF"]
    assert list(figure.data[0].x) == list(range(1, 9))
    assert figure.layout.yaxis.range == (-1.05, 1.05)
    pd.testing.assert_series_equal(reference, reference_before)
    pd.testing.assert_series_equal(synthetic, synthetic_before)


@pytest.mark.parametrize(
    ("reference", "synthetic", "message"),
    [
        (pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0, 3.0]), "At least 4 finite points"),
        (pd.Series([1.0] * 8), pd.Series(range(8)), "constant or near-constant"),
    ],
)
def test_autocorrelation_figure_rejects_unusable_series(reference, synthetic, message):
    with pytest.raises(ValueError, match=message):
        generate_autocorrelation_similarity_figure(reference, synthetic)
