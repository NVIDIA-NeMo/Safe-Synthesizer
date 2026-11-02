# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from nemo_safe_synthesizer.evaluation.components.deep_structure import DeepStructure
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import Grade


@pytest.mark.slow
def test_deep_structure(fixture_evaluation_datasets_5k):
    """PCA (Principal Component Analysis) on 5k rows - very computationally expensive."""
    deep_structure = DeepStructure.from_evaluation_datasets(fixture_evaluation_datasets_5k)
    assert deep_structure.name == "Deep Structure Stability"
    assert deep_structure.score.score is not None and deep_structure.score.score > 0
    assert deep_structure.training_pca is not None
    assert deep_structure.training_pca.shape == (5000, 2)
    assert deep_structure.synthetic_pca is not None
    assert deep_structure.synthetic_pca.shape == (5000, 2)


def test_too_few_cols(fixture_training_df, fixture_synthetic_df, fixture_test_df):
    """Edge case test - small dataset, should be fast."""
    fixture_training_df = fixture_training_df[["num"]]
    fixture_synthetic_df = fixture_synthetic_df[["num"]]
    fixture_test_df = fixture_test_df[["num"]]
    evaluation_datasets = EvaluationDatasets.from_dataframes(fixture_training_df, fixture_synthetic_df, fixture_test_df)
    deep_structure = DeepStructure.from_evaluation_datasets(evaluation_datasets)
    assert deep_structure.name == "Deep Structure Stability"
    assert deep_structure.training_pca is None
    assert deep_structure.synthetic_pca is None
    assert deep_structure.score.grade == Grade.UNAVAILABLE
    assert deep_structure.score.notes is not None and "Missing input Dataframe." in deep_structure.score.notes


def test_pii_columns_are_encoded_by_their_own_value_counts() -> None:
    training = pd.DataFrame({"cardholder": ["Ada", "Ada", "Ada", "Bo"], "state": ["CA", "CA", "TX", "TX"]})
    synthetic = pd.DataFrame({"cardholder": ["Cy", "Cy", "Cy", "Di"], "state": ["CA", "NV", "TX", "TX"]})

    training_labels, synthetic_labels = DeepStructure._encode_categorical_columns(
        training, synthetic, ["cardholder", "state"], frozenset({"cardholder"})
    )

    assert training_labels is not None and synthetic_labels is not None
    # A replaced name with three transactions lands where the training name with three does.
    assert synthetic_labels["cardholder"].tolist() == training_labels["cardholder"].tolist() == [3, 3, 3, 1]
    # A regular category keeps the training counts, so a value the training data never had is 0.
    assert synthetic_labels["state"].tolist() == [2, 0, 2, 2]
