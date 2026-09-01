# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

from nemo_safe_synthesizer.artifacts.analyzers.field_features import (
    FieldType,
)
from nemo_safe_synthesizer.data_processing.dataset_profile import discover_dataset_profile
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import (
    EvaluationDatasets,
)


def test_from_dataframes_happy_path(fixture_training_df, fixture_synthetic_df, fixture_test_df):
    evaluation_datasets = EvaluationDatasets.from_dataframes(fixture_training_df, fixture_synthetic_df, fixture_test_df)

    assert evaluation_datasets is not None
    assert len(evaluation_datasets.training) == 100
    assert len(evaluation_datasets.synthetic) == 100
    assert evaluation_datasets.test is not None
    assert len(evaluation_datasets.test) == 100

    assert len(evaluation_datasets.evaluation_fields) == 8
    for f in evaluation_datasets.evaluation_fields:
        if f.name == "num":
            assert f.training_field_features.type == FieldType.FLOAT
            assert f.synthetic_field_features.type == FieldType.FLOAT
            assert f.training_distribution is not None
            assert len(f.training_distribution) > 0
            assert f.distribution_distance is not None
            assert f.distribution_distance > 0.01
        elif f.name == "num_Int64":
            assert f.training_field_features.type == FieldType.INTEGER
            assert f.synthetic_field_features.type == FieldType.INTEGER
            assert f.training_distribution is not None
            assert len(f.training_distribution) > 0
            assert f.distribution_distance is not None
            assert f.distribution_distance > 0.01
        elif f.name in ["num_cat", "num_cat_Int64", "small_cat"]:
            assert f.training_field_features.type == FieldType.CATEGORICAL
            assert f.synthetic_field_features.type == FieldType.CATEGORICAL
            assert f.training_field_features.unique_count < 10
            assert f.training_distribution is not None
            assert len(f.training_distribution) > 0
            assert f.distribution_distance is not None
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
            assert f.training_distribution is not None
            assert len(f.training_distribution) > 0
            assert f.distribution_distance is not None
            assert f.distribution_distance > 0.01
        elif f.name in ["text"]:
            assert f.training_field_features.type == FieldType.TEXT
            assert f.synthetic_field_features.type == FieldType.TEXT
            assert f.training_field_features.unique_count > 90
            assert f.training_distribution is None
            assert f.distribution_distance is None


def test_from_dataframes_with_sampling(fixture_training_df_5k, fixture_synthetic_df_5k, fixture_test_df):
    evaluation_datasets = EvaluationDatasets.from_dataframes(
        fixture_training_df_5k, fixture_synthetic_df_5k, fixture_test_df, rows=1000
    )

    assert evaluation_datasets is not None
    assert len(evaluation_datasets.training) == 1000
    assert len(evaluation_datasets.synthetic) == 1000
    assert evaluation_datasets.test is not None
    assert len(evaluation_datasets.test) == 100

    assert len(evaluation_datasets.evaluation_fields) == 8


def test_degenerate_input(fixture_synthetic_df_5k, fixture_test_df):
    with pytest.raises(ValueError):
        EvaluationDatasets.from_dataframes(None, fixture_synthetic_df_5k, fixture_test_df)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        EvaluationDatasets.from_dataframes(pd.DataFrame(), fixture_synthetic_df_5k, fixture_test_df)


def test_column_intersection(fixture_training_df, fixture_synthetic_df, fixture_test_df):
    fixture_training_df = fixture_training_df[["num", "num_cat"]]
    fixture_synthetic_df = fixture_synthetic_df[["num", "other"]]
    fixture_test_df = fixture_test_df[["num", "text"]]
    evaluation_datasets = EvaluationDatasets.from_dataframes(fixture_training_df, fixture_synthetic_df, fixture_test_df)
    assert len(evaluation_datasets.evaluation_fields) == 1
    assert evaluation_datasets.evaluation_fields[0].name == "num"


def test_empty_column_intersection(fixture_training_df, fixture_synthetic_df):
    fixture_training_df = fixture_training_df[["num", "num_cat"]]
    fixture_synthetic_df = fixture_synthetic_df[["other", "text"]]
    with pytest.raises(ValueError):
        EvaluationDatasets.from_dataframes(fixture_training_df, fixture_synthetic_df)


def test_empty_testdf_intersection(fixture_training_df, fixture_synthetic_df, fixture_test_df):
    fixture_training_df = fixture_training_df[["num", "num_cat"]]
    fixture_synthetic_df = fixture_synthetic_df[["num", "num_cat"]]
    fixture_test_df = fixture_test_df[["other", "text"]]
    with pytest.raises(ValueError):
        EvaluationDatasets.from_dataframes(fixture_training_df, fixture_synthetic_df, fixture_test_df)


def test_get_columns_of_type(fixture_training_df):
    dataset = EvaluationDatasets.from_dataframes(fixture_training_df, fixture_training_df, fixture_training_df)
    assert set(dataset.get_tabular_columns()) == set(
        ["num", "num_Int64", "num_cat", "num_cat_Int64", "small_cat", "boolean"]
    )
    assert set(dataset.get_nominal_columns()) == set(["num_cat", "num_cat_Int64", "small_cat", "boolean"])
    assert dataset.get_text_columns() == ["text"]


def test_profile_routes_columns_without_reinferring_types():
    training = pd.DataFrame(
        {
            "number": list(range(16)),
            "label": ["a", "b"] * 8,
            "text": [f"several words in this text value {index}" for index in range(16)],
        }
    )
    synthetic = training.copy()
    profile = discover_dataset_profile(training)

    dataset = EvaluationDatasets.from_dataframes(
        training,
        synthetic,
        enable_sampling=False,
        dataset_profile=profile,
    )

    assert dataset.get_tabular_columns() == profile.tabular_columns()
    assert dataset.get_nominal_columns() == profile.nominal_columns()
    assert dataset.get_text_columns() == profile.text_columns()
