# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from nemo_safe_synthesizer.artifacts.analyzers.field_features import (
    FieldType,
)
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import (
    EvaluationDatasets,
)


def test_from_dataframes_happy_path(training_df, synthetic_df, test_df):
    evaluation_datasets = EvaluationDatasets.from_dataframes(training_df, synthetic_df, test_df)

    assert evaluation_datasets is not None
    assert len(evaluation_datasets.training) == 100
    assert len(evaluation_datasets.synthetic) == 100
    assert len(evaluation_datasets.test) == 100

    assert len(evaluation_datasets.evaluation_fields) == 8
    for f in evaluation_datasets.evaluation_fields:
        if f.name in ["num", "num_Int64"]:
            assert f.training_field_features.type == FieldType.NUMERIC
            assert f.synthetic_field_features.type == FieldType.NUMERIC
            assert len(f.training_distribution) > 0
            assert f.distribution_distance > 0.01
        elif f.name in ["num_cat", "num_cat_Int64", "small_cat"]:
            assert f.training_field_features.type == FieldType.CATEGORICAL
            assert f.synthetic_field_features.type == FieldType.CATEGORICAL
            assert f.training_field_features.unique_count < 10
            assert len(f.training_distribution) > 0
            assert f.distribution_distance > 0.01
        elif f.name in ["other"]:
            assert f.training_field_features.type == FieldType.OTHER
            assert f.synthetic_field_features.type == FieldType.OTHER
            assert f.training_field_features.unique_count > 90
            assert f.training_distribution is None
            assert f.distribution_distance is None
        elif f.name in ["boolean"]:
            assert f.training_field_features.type == FieldType.BINARY
            assert f.synthetic_field_features.type == FieldType.BINARY
            assert f.training_field_features.unique_count == 2
            assert len(f.training_distribution) > 0
            assert f.distribution_distance > 0.01
        elif f.name in ["text"]:
            assert f.training_field_features.type == FieldType.TEXT
            assert f.synthetic_field_features.type == FieldType.TEXT
            assert f.training_field_features.unique_count > 90
            assert f.training_distribution is None
            assert f.distribution_distance is None


def test_from_dataframes_with_sampling(training_df_5k, synthetic_df_5k, test_df):
    evaluation_datasets = EvaluationDatasets.from_dataframes(training_df_5k, synthetic_df_5k, test_df, rows=1000)

    assert evaluation_datasets is not None
    assert len(evaluation_datasets.training) == 1000
    assert len(evaluation_datasets.synthetic) == 1000
    assert len(evaluation_datasets.test) == 100

    assert len(evaluation_datasets.evaluation_fields) == 8


def test_degenerate_input(synthetic_df_5k, test_df):
    with pytest.raises(ValueError):
        EvaluationDatasets.from_dataframes(None, synthetic_df_5k, test_df)
    with pytest.raises(ValueError):
        EvaluationDatasets.from_dataframes(pd.DataFrame(), synthetic_df_5k, test_df)


def test_column_intersection(training_df, synthetic_df, test_df):
    training_df = training_df[["num", "num_cat"]]
    synthetic_df = synthetic_df[["num", "other"]]
    test_df = test_df[["num", "text"]]
    evaluation_datasets = EvaluationDatasets.from_dataframes(training_df, synthetic_df, test_df)
    assert len(evaluation_datasets.evaluation_fields) == 1
    assert evaluation_datasets.evaluation_fields[0].name == "num"


def test_empty_column_intersection(training_df, synthetic_df):
    training_df = training_df[["num", "num_cat"]]
    synthetic_df = synthetic_df[["other", "text"]]
    with pytest.raises(ValueError):
        EvaluationDatasets.from_dataframes(training_df, synthetic_df)


def test_empty_testdf_intersection(training_df, synthetic_df, test_df):
    training_df = training_df[["num", "num_cat"]]
    synthetic_df = synthetic_df[["num", "num_cat"]]
    test_df = test_df[["other", "text"]]
    with pytest.raises(ValueError):
        EvaluationDatasets.from_dataframes(training_df, synthetic_df, test_df)


def test_get_columns_of_type(training_df):
    dataset = EvaluationDatasets.from_dataframes(training_df, training_df, training_df)
    assert set(dataset.get_tabular_columns()) == set(
        ["num", "num_Int64", "num_cat", "num_cat_Int64", "small_cat", "boolean"]
    )
    assert set(dataset.get_nominal_columns()) == set(["num_cat", "num_cat_Int64", "small_cat", "boolean"])
    assert dataset.get_text_columns() == ["text"]
