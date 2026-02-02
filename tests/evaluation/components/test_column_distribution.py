# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_safe_synthesizer.evaluation.components.column_distribution import ColumnDistribution
from nemo_safe_synthesizer.evaluation.data_model.evaluation_dataset import (
    EvaluationDataset,
)


@pytest.mark.slow
def test_column_distribution(evaluation_dataset_5k):
    """Statistical analysis on 5k rows - computationally expensive."""
    column_distribution = ColumnDistribution.from_evaluation_dataset(evaluation_dataset_5k)
    assert column_distribution.name == "Column Distribution Stability"
    assert len(column_distribution.evaluation_fields) == 8


@pytest.mark.slow
def test_column_distribution_text_other(train_df_5k):
    """Statistical analysis on 5k rows - computationally expensive."""
    evaluation_dataset = EvaluationDataset.from_dataframes(
        train_df_5k[["text", "other"]], train_df_5k[["text", "other"]], None
    )
    column_distribution = ColumnDistribution.from_evaluation_dataset(evaluation_dataset)
    assert column_distribution.score.score is None
    assert column_distribution.score.grade.value == "Unavailable"
    assert column_distribution.score.notes == "No tabular columns detected."
