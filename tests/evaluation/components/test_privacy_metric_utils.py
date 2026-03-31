# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip(
    "sentence_transformers",
    reason="sentence_transformers is required for these tests (install with: uv sync --extra cpu)",
)

from nemo_safe_synthesizer.evaluation.components.privacy_metric_utils import (
    divide_tabular_text,
    embed_text,
)


@pytest.fixture
def mock_embedder():
    """A mock SentenceTransformer whose .encode() returns deterministic arrays."""
    embedder = MagicMock()

    def _encode(data, **kwargs):
        # Return a distinct but deterministic embedding per string.
        # Use the length of each string as a simple seed for reproducibility.
        return np.array([[float(len(s)), float(len(s)) * 2, float(len(s)) * 3] for s in data])

    embedder.encode = MagicMock(side_effect=_encode)
    return embedder


def test_divide_tabular_text(train_df):
    text_fields = ["text", "other"]
    tabular, text = divide_tabular_text(train_df, text_fields)

    assert "text" not in tabular.columns
    assert "other" not in tabular.columns
    assert set(text.columns) == {"other", "text"}
    assert len(tabular) == len(train_df)
    assert len(text) == len(train_df)


def test_embed_text(mock_embedder):
    """Regression test: with 3+ columns the old pairwise-averaging code
    over-weighted later columns.  The corrected np.stack/np.mean must give
    each column equal weight.
    """
    df = pd.DataFrame(
        {
            "a": ["x"],  # len 1 → embedding [1, 2, 3]
            "b": ["xx"],  # len 2 → embedding [2, 4, 6]
            "c": ["xxxx"],  # len 4 → embedding [4, 8, 12]
        }
    )
    result = embed_text(df, mock_embedder)

    # True mean of [1,2,3], [2,4,6], [4,8,12] across columns:
    # = [(1+2+4)/3, (2+4+8)/3, (3+6+12)/3] = [7/3, 14/3, 21/3]
    expected = np.array([7 / 3, 14 / 3, 7.0])
    np.testing.assert_array_almost_equal(result["embedding"].iloc[0], expected)
