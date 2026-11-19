# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from nemo_safe_synthesizer.evaluation.components.column_distribution import ColumnDistribution
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import (
    EvaluationDatasets,
)
from nemo_safe_synthesizer.pii_replacer.transform_result import ColumnStatistics


def test_jinja_context_leaves_out_training_values():
    df = pd.DataFrame({"amount": [float(i) for i in range(50)], "color": ["red", "blue"] * 25})
    column_statistics = {
        "color": ColumnStatistics(
            assigned_type="text",
            assigned_entity="color",
            detected_entity_counts={"color": 2},
            detected_entity_values={"color": {"red", "blue"}},
        )
    }
    evaluation_datasets = EvaluationDatasets.from_dataframes(df, df, None, column_statistics=column_statistics)

    context = ColumnDistribution.from_evaluation_datasets(evaluation_datasets).jinja_context

    for field in context["evaluation_fields"]:
        assert "training_distribution" not in field
        assert "synthetic_distribution" not in field
        assert "unique_values_list" not in field["training_field_features"]
        assert "unique_values_list" not in field["synthetic_field_features"]
    assert "detected_entity_values" not in context["column_statistics"]["color"]
    assert context["column_statistics"]["color"]["detected_entity_counts"] == {"color": 2}


@pytest.mark.slow
def test_column_distribution(fixture_evaluation_datasets_5k):
    """Statistical analysis on 5k rows - computationally expensive."""
    column_distribution = ColumnDistribution.from_evaluation_datasets(fixture_evaluation_datasets_5k)
    assert column_distribution.name == "Column Distribution Stability"
    assert len(column_distribution.evaluation_fields) == 8


@pytest.mark.slow
def test_column_distribution_text_other(fixture_training_df_5k):
    """Statistical analysis on 5k rows - computationally expensive."""
    evaluation_datasets = EvaluationDatasets.from_dataframes(
        fixture_training_df_5k[["text", "other"]], fixture_training_df_5k[["text", "other"]], None
    )
    column_distribution = ColumnDistribution.from_evaluation_datasets(evaluation_datasets)
    assert column_distribution.score.score is None
    assert column_distribution.score.grade.value == "Unavailable"
    assert column_distribution.score.notes == "No tabular columns detected."
