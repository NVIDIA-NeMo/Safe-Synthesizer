# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_safe_synthesizer.evaluation.components.deep_structure import DeepStructure
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import Grade


@pytest.mark.slow
def test_deep_structure(evaluation_datasets_5k):
    """PCA (Principal Component Analysis) on 5k rows - very computationally expensive."""
    deep_structure = DeepStructure.from_evaluation_datasets(evaluation_datasets_5k)
    assert deep_structure.name == "Deep Structure Stability"
    assert deep_structure.score.score > 0
    assert deep_structure.training_pca.shape == (5000, 2)
    assert deep_structure.synthetic_pca.shape == (5000, 2)


def test_too_few_cols(training_df, synthetic_df, test_df):
    """Edge case test - small dataset, should be fast."""
    training_df = training_df[["num"]]
    synthetic_df = synthetic_df[["num"]]
    test_df = test_df[["num"]]
    evaluation_datasets = EvaluationDatasets.from_dataframes(training_df, synthetic_df, test_df)
    deep_structure = DeepStructure.from_evaluation_datasets(evaluation_datasets)
    assert deep_structure.name == "Deep Structure Stability"
    assert deep_structure.training_pca is None
    assert deep_structure.synthetic_pca is None
    assert deep_structure.score.grade == Grade.UNAVAILABLE
    assert "Missing input Dataframe." in deep_structure.score.notes
