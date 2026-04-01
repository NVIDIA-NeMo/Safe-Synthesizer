# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

# Skip all tests in this module if sentence_transformers is not available
pytest.importorskip(
    "sentence_transformers",
    reason="sentence_transformers is required for these tests (install with: uv sync --extra cpu)",
)

import logging

from nemo_safe_synthesizer.evaluation.components.membership_inference_protection import MembershipInferenceProtection
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets

logger = logging.getLogger(__name__)


@pytest.mark.requires_gpu
def test_membership_inference_protection(training_df_5k, synthetic_df_5k, test_df):
    """Test MIA with tabular-only data (sklearn NearestNeighbors path)."""
    evaluation_datasets = EvaluationDatasets.from_dataframes(training_df_5k, synthetic_df_5k, test_df)
    membership_inference_protection = MembershipInferenceProtection.from_evaluation_datasets(evaluation_datasets)

    logger.info(membership_inference_protection.attack_sum_df)
    assert (
        membership_inference_protection.attack_sum_df is not None
        and not membership_inference_protection.attack_sum_df.empty
    )

    logger.info(membership_inference_protection.tps_values)
    assert len(membership_inference_protection.tps_values) > 0

    logger.info(membership_inference_protection.fps_values)
    assert len(membership_inference_protection.fps_values) > 0


@pytest.mark.requires_gpu
def test_membership_inference_protection_mixed_text_tabular(training_df_mixed_5k, synthetic_df_mixed_5k, test_df_mixed):
    """Test MIA with mixed text+tabular data (hybrid sklearn + sentence-transformers path).

    This test exercises the hybrid nearest neighbor path that combines:
    - sentence-transformers for text column similarity
    - sklearn NearestNeighbors for tabular column similarity
    - weighted hybrid distance calculation
    """
    evaluation_datasets = EvaluationDatasets.from_dataframes(training_df_mixed_5k, synthetic_df_mixed_5k, test_df_mixed)
    membership_inference_protection = MembershipInferenceProtection.from_evaluation_datasets(evaluation_datasets)

    logger.info(f"MIA attack summary: {membership_inference_protection.attack_sum_df}")
    assert membership_inference_protection.attack_sum_df is not None
    assert not membership_inference_protection.attack_sum_df.empty

    # Verify TPR/FPR values were computed
    assert len(membership_inference_protection.tps_values) > 0
    assert len(membership_inference_protection.fps_values) > 0

    # Verify the protection score was computed
    assert membership_inference_protection.score is not None


@pytest.mark.requires_gpu
def test_membership_inference_protection_text_only(training_df_text_only, synthetic_df_text_only, test_df_text_only):
    """Test MIA with text-only data (sentence-transformers only, no sklearn).

    This test exercises the text-only nearest neighbor path that uses
    only sentence-transformers for semantic similarity search.
    """
    evaluation_datasets = EvaluationDatasets.from_dataframes(
        training_df_text_only, synthetic_df_text_only, test_df_text_only
    )
    membership_inference_protection = MembershipInferenceProtection.from_evaluation_datasets(evaluation_datasets)

    logger.info(f"MIA text-only attack summary: {membership_inference_protection.attack_sum_df}")
    assert membership_inference_protection.attack_sum_df is not None
    assert not membership_inference_protection.attack_sum_df.empty

    # Verify TPR/FPR values were computed
    assert len(membership_inference_protection.tps_values) > 0
    assert len(membership_inference_protection.fps_values) > 0

    # Verify the protection score was computed
    assert membership_inference_protection.score is not None
