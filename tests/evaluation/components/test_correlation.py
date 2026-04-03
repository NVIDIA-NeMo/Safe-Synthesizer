# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from nemo_safe_synthesizer.evaluation.components.correlation import (
    Correlation,
)
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import (
    EvaluationDatasets,
)
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import Grade


@pytest.mark.slow
def test_correlation(training_df_5k, synthetic_df_5k, test_df):
    """Correlation matrix computation on 5k rows - computationally expensive."""
    evaluation_datasets = EvaluationDatasets.from_dataframes(training_df_5k, synthetic_df_5k, test_df)
    correlation = Correlation.from_evaluation_datasets(evaluation_datasets)

    assert correlation.name == "Column Correlation Stability"
    assert correlation.score.raw_score > 0
    assert correlation.score.score == 10
    assert correlation.score.grade == Grade.EXCELLENT
    assert correlation.score.notes is None

    # Text and Other columns excluded
    assert correlation.training_correlation.shape == (6, 6)
    assert ((correlation.training_correlation >= -1) & (correlation.training_correlation <= 1)).all().all()
    assert correlation.synthetic_correlation.shape == (6, 6)
    assert ((correlation.synthetic_correlation >= -1) & (correlation.synthetic_correlation <= 1)).all().all()
    assert correlation.correlation_difference.shape == (6, 6)
    assert ((correlation.correlation_difference >= 0) & (correlation.correlation_difference <= 1)).all().all()


@pytest.mark.slow
def test_correlation_all_numeric(training_df_5k, synthetic_df_5k, test_df):
    """Correlation matrix computation on 5k rows - computationally expensive."""
    # We should still do correlation if no nominal columns
    nominal_columns = ["num_cat", "num_cat_Int64", "small_cat", "boolean"]
    training_df_5k = training_df_5k.drop(nominal_columns, axis=1)
    synthetic_df_5k = synthetic_df_5k.drop(nominal_columns, axis=1)

    evaluation_datasets = EvaluationDatasets.from_dataframes(training_df_5k, synthetic_df_5k, test_df)
    correlation = Correlation.from_evaluation_datasets(evaluation_datasets)

    assert correlation.name == "Column Correlation Stability"
    assert correlation.score.raw_score > 0
    assert correlation.score.score == 10
    assert correlation.score.grade == Grade.EXCELLENT
    assert correlation.score.notes is None

    assert correlation.training_correlation.shape == (2, 2)


@pytest.mark.slow
def test_correlation_not_enough_columns(training_df_5k, synthetic_df_5k, test_df, caplog):
    """Correlation matrix computation on 5k rows - computationally expensive."""
    caplog.set_level(logging.INFO)

    # Two columns, one is text. We will not have enough to do correlations.
    training_df_5k = training_df_5k[["num_cat", "text"]]
    synthetic_df_5k = synthetic_df_5k[["num_cat", "text"]]

    evaluation_datasets = EvaluationDatasets.from_dataframes(training_df_5k, synthetic_df_5k, test_df)
    correlation = Correlation.from_evaluation_datasets(evaluation_datasets)

    assert correlation.name == "Column Correlation Stability"
    assert correlation.score.raw_score is None
    assert correlation.score.score is None
    assert correlation.score.grade == Grade.UNAVAILABLE
    assert "Less than two correlatable columns found. Skipping correlation calculations." in caplog.text

    assert correlation.training_correlation is None
    assert correlation.synthetic_correlation is None
    assert correlation.correlation_difference is None
