# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402
import logging

import pytest

# Skip all tests in this module if sentence_transformers is not available
pytest.importorskip(
    "sentence_transformers",
    reason="sentence_transformers is required for these tests (install with: uv sync --extra cpu)",
)

from nemo_safe_synthesizer.evaluation.components.attribute_inference_protection import AttributeInferenceProtection
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets

logger = logging.getLogger(__name__)


@pytest.mark.slow
def test_attribute_inference_protection(training_df_5k, synthetic_df_5k, test_df):
    """Test AIA with tabular-only data (sklearn NearestNeighbors path)."""
    evaluation_datasets = EvaluationDatasets.from_dataframes(training_df_5k, synthetic_df_5k, test_df)
    attribute_inference_protection = AttributeInferenceProtection.from_evaluation_datasets(evaluation_datasets)
    logger.info(attribute_inference_protection.col_accuracy_df)
    assert (
        attribute_inference_protection.col_accuracy_df is not None
        and not attribute_inference_protection.col_accuracy_df.empty
    )


@pytest.mark.slow
@pytest.mark.requires_gpu
def test_attribute_inference_protection_mixed_text_tabular(training_df_mixed_5k, synthetic_df_mixed_5k, test_df_mixed):
    """Test AIA with mixed text+tabular data (hybrid sklearn + sentence-transformers path).

    This test exercises the hybrid nearest neighbor path that combines:
    - sentence-transformers for text column similarity
    - sklearn NearestNeighbors for tabular column similarity
    - weighted hybrid distance calculation
    """
    evaluation_datasets = EvaluationDatasets.from_dataframes(training_df_mixed_5k, synthetic_df_mixed_5k, test_df_mixed)
    attribute_inference_protection = AttributeInferenceProtection.from_evaluation_datasets(evaluation_datasets)

    logger.info(f"AIA columns evaluated: {attribute_inference_protection.col_accuracy_df}")
    assert attribute_inference_protection.col_accuracy_df is not None
    assert not attribute_inference_protection.col_accuracy_df.empty

    # Verify the protection score was computed
    assert attribute_inference_protection.score is not None


@pytest.mark.requires_gpu
def test_attribute_inference_protection_text_only(training_df_text_only, synthetic_df_text_only, test_df_text_only):
    """Test AIA with text-only data (sentence-transformers only, no sklearn).

    This test exercises the text-only nearest neighbor path that uses
    only sentence-transformers for semantic similarity search.
    """
    evaluation_datasets = EvaluationDatasets.from_dataframes(
        training_df_text_only, synthetic_df_text_only, test_df_text_only
    )
    attribute_inference_protection = AttributeInferenceProtection.from_evaluation_datasets(evaluation_datasets)

    logger.info(f"AIA text-only columns evaluated: {attribute_inference_protection.col_accuracy_df}")
    assert attribute_inference_protection.col_accuracy_df is not None
    assert not attribute_inference_protection.col_accuracy_df.empty

    # Verify the protection score was computed
    assert attribute_inference_protection.score is not None
