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
from nemo_safe_synthesizer.evaluation.data_model.evaluation_dataset import EvaluationDataset

logger = logging.getLogger(__name__)


@pytest.mark.skip(reason="Times out")
def test_attribute_inference_protection(train_df_5k, synth_df_5k, test_df):
    evaluation_dataset = EvaluationDataset.from_dataframes(train_df_5k, synth_df_5k, test_df)
    membership_inference_protection = MembershipInferenceProtection.from_evaluation_dataset(evaluation_dataset)

    logger.info(membership_inference_protection.attack_sum_df)
    assert (
        membership_inference_protection.attack_sum_df is not None
        and not membership_inference_protection.attack_sum_df.empty
    )

    logger.info(membership_inference_protection.tps_values)
    assert len(membership_inference_protection.tps_values) > 0

    logger.info(membership_inference_protection.fps_values)
    assert len(membership_inference_protection.fps_values) > 0
