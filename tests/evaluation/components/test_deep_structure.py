# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_safe_synthesizer.evaluation.components.deep_structure import DeepStructure
from nemo_safe_synthesizer.evaluation.data_model.evaluation_dataset import EvaluationDataset
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import Grade


@pytest.mark.slow
def test_deep_structure(evaluation_dataset_5k):
    """PCA (Principal Component Analysis) on 5k rows - very computationally expensive."""
    deep_structure = DeepStructure.from_evaluation_dataset(evaluation_dataset_5k)
    assert deep_structure.name == "Deep Structure Stability"
    assert deep_structure.score.score > 0
    assert deep_structure.reference_pca.shape == (5000, 2)
    assert deep_structure.output_pca.shape == (5000, 2)


def test_too_few_cols(train_df, synth_df, test_df):
    """Edge case test - small dataset, should be fast."""
    train_df = train_df[["num"]]
    synth_df = synth_df[["num"]]
    test_df = test_df[["num"]]
    evaluation_dataset = EvaluationDataset.from_dataframes(train_df, synth_df, test_df)
    deep_structure = DeepStructure.from_evaluation_dataset(evaluation_dataset)
    assert deep_structure.name == "Deep Structure Stability"
    assert deep_structure.reference_pca is None
    assert deep_structure.output_pca is None
    assert deep_structure.score.grade == Grade.UNAVAILABLE
    assert "Missing input Dataframe." in deep_structure.score.notes
