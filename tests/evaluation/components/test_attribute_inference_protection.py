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

# Skip all tests in this module if faiss is not properly available
# Note: faiss module may exist but lack IndexFlatL2 if improperly installed
faiss = pytest.importorskip(
    "faiss",
    reason="faiss is required for these tests",
)
if not hasattr(faiss, "IndexFlatL2"):
    pytest.skip(
        "faiss is installed but lacks IndexFlatL2 - likely a stub package (install faiss-cpu or faiss-gpu)",
        allow_module_level=True,
    )

from nemo_safe_synthesizer.evaluation.components.attribute_inference_protection import AttributeInferenceProtection
from nemo_safe_synthesizer.evaluation.data_model.evaluation_dataset import EvaluationDataset

logger = logging.getLogger(__name__)


@pytest.mark.slow
def test_attribute_inference_protection(train_df_5k, synth_df_5k, test_df):
    evaluation_dataset = EvaluationDataset.from_dataframes(train_df_5k, synth_df_5k, test_df)
    attribute_inference_protection = AttributeInferenceProtection.from_evaluation_dataset(evaluation_dataset)
    logger.info(attribute_inference_protection.col_accuracy_df)
    assert (
        attribute_inference_protection.col_accuracy_df is not None
        and not attribute_inference_protection.col_accuracy_df.empty
    )
