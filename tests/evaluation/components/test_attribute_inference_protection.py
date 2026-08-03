# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402
import logging
import warnings

import numpy as np
import pandas as pd
import pytest

# Skip all tests in this module if sentence_transformers is not available
pytest.importorskip(
    "sentence_transformers",
    reason="sentence_transformers is required for these tests (install with: uv sync --extra cpu)",
)

from nemo_safe_synthesizer.evaluation.components.attribute_inference_protection import AttributeInferenceProtection
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets

logger = logging.getLogger(__name__)


def test_aia_tabular_unit_exercises_entropy_weighting(monkeypatch: pytest.MonkeyPatch):
    """Cover the fast tabular AIA path without loading text embedding models."""
    training_df = pd.DataFrame(
        {
            "stable": [1, 1, 1, 1],
            "binary": [1, 1, 2, 2],
            "varied": [1, 2, 3, 4],
        }
    )
    synthetic_df = training_df.copy()

    def fake_get_synth_nn(*_args, **_kwargs) -> pd.DataFrame:
        return synthetic_df.head(2)

    monkeypatch.setattr(AttributeInferenceProtection, "_get_synth_nn", staticmethod(fake_get_synth_nn))

    score, col_accuracy_df = AttributeInferenceProtection._aia(
        training_df=training_df,
        synthetic_df=synthetic_df,
        quasi_identifier_count=1,
    )

    assert score.score is not None
    assert col_accuracy_df is not None
    assert list(col_accuracy_df["Column"]) == ["stable", "binary", "varied"]


def test_aia_wide_table_uses_windowed_quasi_identifier_combinations(monkeypatch: pytest.MonkeyPatch):
    """Cover the wide-table fallback that avoids materializing all combinations."""
    columns = [f"col_{index}" for index in range(501)]
    values = np.vstack([np.arange(501), np.arange(501) + 1])
    training_df = pd.DataFrame(values, columns=columns)
    synthetic_df = training_df.copy()
    synth_calls: list[tuple[str, ...]] = []

    def fake_get_synth_nn(train_row, *_args, **_kwargs) -> pd.DataFrame:
        synth_calls.append(tuple(train_row.columns))
        return synthetic_df.head(2)

    monkeypatch.setattr(AttributeInferenceProtection, "_get_synth_nn", staticmethod(fake_get_synth_nn))

    score, col_accuracy_df = AttributeInferenceProtection._aia(
        training_df=training_df,
        synthetic_df=synthetic_df,
        quasi_identifier_count=500,
    )

    assert score.score is not None
    assert col_accuracy_df is not None
    assert len(col_accuracy_df) == len(columns)
    assert len(synth_calls) == 2
    assert set(synth_calls) == {tuple(columns[:500]), tuple(columns[1:])}


def test_aia_preserves_non_string_column_labels(monkeypatch: pytest.MonkeyPatch):
    """Cover DataFrames whose column labels are not strings."""
    training_df = pd.DataFrame(
        {
            0: [1, 2, 3, 4],
            1: [10, 20, 30, 40],
            2: [100, 200, 300, 400],
        }
    )
    synthetic_df = training_df.copy()
    synth_calls: list[tuple[int, ...]] = []

    def fake_get_synth_nn(train_row, *_args, **_kwargs) -> pd.DataFrame:
        synth_calls.append(tuple(train_row.columns))
        return synthetic_df.head(2)

    monkeypatch.setattr(AttributeInferenceProtection, "_get_synth_nn", staticmethod(fake_get_synth_nn))

    score, col_accuracy_df = AttributeInferenceProtection._aia(
        training_df=training_df,
        synthetic_df=synthetic_df,
        quasi_identifier_count=1,
    )

    assert score.score is not None
    assert col_accuracy_df is not None
    assert set(col_accuracy_df["Column"]) == {0, 1, 2}
    assert synth_calls
    assert all(isinstance(column, int) for call in synth_calls for column in call)


@pytest.mark.slow
def test_attribute_inference_protection(fixture_training_df_5k, fixture_synthetic_df_5k, fixture_test_df):
    """Test AIA with tabular-only data (sklearn NearestNeighbors path)."""
    evaluation_datasets = EvaluationDatasets.from_dataframes(
        fixture_training_df_5k, fixture_synthetic_df_5k, fixture_test_df
    )
    attribute_inference_protection = AttributeInferenceProtection.from_evaluation_datasets(evaluation_datasets)
    logger.info(attribute_inference_protection.col_accuracy_df)
    assert (
        attribute_inference_protection.col_accuracy_df is not None
        and not attribute_inference_protection.col_accuracy_df.empty
    )


@pytest.mark.slow
@pytest.mark.requires_gpu
def test_attribute_inference_protection_mixed_text_tabular(
    fixture_training_df_mixed_5k, fixture_synthetic_df_mixed_5k, fixture_test_df_mixed
):
    """Test AIA with mixed text+tabular data (hybrid sklearn + sentence-transformers path).

    This test exercises the hybrid nearest neighbor path that combines:
    - sentence-transformers for text column similarity
    - sklearn NearestNeighbors for tabular column similarity
    - weighted hybrid distance calculation
    """
    evaluation_datasets = EvaluationDatasets.from_dataframes(
        fixture_training_df_mixed_5k, fixture_synthetic_df_mixed_5k, fixture_test_df_mixed
    )
    attribute_inference_protection = AttributeInferenceProtection.from_evaluation_datasets(evaluation_datasets)

    logger.info(f"AIA columns evaluated: {attribute_inference_protection.col_accuracy_df}")
    assert attribute_inference_protection.col_accuracy_df is not None
    assert not attribute_inference_protection.col_accuracy_df.empty

    # Verify the protection score was computed
    assert attribute_inference_protection.score is not None


@pytest.mark.requires_gpu
def test_attribute_inference_protection_text_only(
    fixture_training_df_text_only, fixture_synthetic_df_text_only, fixture_test_df_text_only
):
    """Test AIA with text-only data (sentence-transformers only, no sklearn).

    This test exercises the text-only nearest neighbor path that uses
    only sentence-transformers for semantic similarity search.
    """
    evaluation_datasets = EvaluationDatasets.from_dataframes(
        fixture_training_df_text_only, fixture_synthetic_df_text_only, fixture_test_df_text_only
    )
    attribute_inference_protection = AttributeInferenceProtection.from_evaluation_datasets(evaluation_datasets)

    logger.info(f"AIA text-only columns evaluated: {attribute_inference_protection.col_accuracy_df}")
    assert attribute_inference_protection.col_accuracy_df is not None
    assert not attribute_inference_protection.col_accuracy_df.empty

    # Verify the protection score was computed
    assert attribute_inference_protection.score is not None


def test_aia_text_column_similarity_returns_scalar(monkeypatch: pytest.MonkeyPatch):
    """Text-column similarity must not rely on array-to-scalar conversion.

    ``SentenceTransformer.encode`` returns a batch, so encoding one string yields a
    ``(1, dim)`` array. Taking ``np.dot`` of that against the ``(dim,)`` synthetic
    vector produces a one-element array, and ``float()`` on it is deprecated in
    NumPy 1.25 and an error from 2.4 on. Warnings are escalated so this fails on
    the pinned NumPy in CI as well as on newer releases.
    """
    training_df = pd.DataFrame(
        {
            "quasi": [1, 2, 3, 4],
            "text": [
                "reserve a table for two at eight",
                "cancel my reservation for friday",
                "what is the weather in denver",
                "add milk to my shopping list",
            ],
        }
    )
    synthetic_df = training_df.copy()

    class StubEmbedder:
        """Minimal stand-in that mirrors the real batch-shaped return value."""

        def encode(self, sentences, **_kwargs):
            return np.ones((len(list(sentences)), 8), dtype=np.float32)

    monkeypatch.setattr(
        "nemo_safe_synthesizer.evaluation.components.attribute_inference_protection.SentenceTransformer",
        lambda *_args, **_kwargs: StubEmbedder(),
    )
    monkeypatch.setattr(
        "nemo_safe_synthesizer.evaluation.components.attribute_inference_protection.find_text_fields",
        lambda df: [column for column in df.columns if column == "text"],
    )
    monkeypatch.setattr(
        AttributeInferenceProtection,
        "_get_synth_nn",
        staticmethod(lambda *_args, **_kwargs: synthetic_df.head(2)),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        score, col_accuracy_df = AttributeInferenceProtection._aia(
            training_df=training_df,
            synthetic_df=synthetic_df,
            quasi_identifier_count=1,
        )

    assert score.score is not None
    assert col_accuracy_df is not None
    assert "text" in list(col_accuracy_df["Column"])
