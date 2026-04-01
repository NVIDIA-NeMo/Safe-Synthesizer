# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from nemo_safe_synthesizer.artifacts.analyzers.field_features import (
    FieldType,
)
from nemo_safe_synthesizer.evaluation.data_model.evaluation_dataset import (
    EvaluationDataset,
)


def test_from_dataframes_happy_path(train_df, synth_df, test_df):
    evaluation_dataset = EvaluationDataset.from_dataframes(train_df, synth_df, test_df)

    assert evaluation_dataset is not None
    assert len(evaluation_dataset.reference) == 100
    assert len(evaluation_dataset.output) == 100
    assert len(evaluation_dataset.test) == 100

    assert len(evaluation_dataset.evaluation_fields) == 8
    for f in evaluation_dataset.evaluation_fields:
        if f.name in ["num", "num_Int64"]:
            assert f.reference_field_features.type == FieldType.NUMERIC
            assert f.output_field_features.type == FieldType.NUMERIC
            assert len(f.reference_distribution) > 0
            assert f.distribution_distance > 0.01
        elif f.name in ["num_cat", "num_cat_Int64", "small_cat"]:
            assert f.reference_field_features.type == FieldType.CATEGORICAL
            assert f.output_field_features.type == FieldType.CATEGORICAL
            assert f.reference_field_features.unique_count < 10
            assert len(f.reference_distribution) > 0
            assert f.distribution_distance > 0.01
        elif f.name in ["other"]:
            assert f.reference_field_features.type == FieldType.OTHER
            assert f.output_field_features.type == FieldType.OTHER
            assert f.reference_field_features.unique_count > 90
            assert f.reference_distribution is None
            assert f.distribution_distance is None
        elif f.name in ["boolean"]:
            assert f.reference_field_features.type == FieldType.BINARY
            assert f.output_field_features.type == FieldType.BINARY
            assert f.reference_field_features.unique_count == 2
            assert len(f.reference_distribution) > 0
            assert f.distribution_distance > 0.01
        elif f.name in ["text"]:
            assert f.reference_field_features.type == FieldType.TEXT
            assert f.output_field_features.type == FieldType.TEXT
            assert f.reference_field_features.unique_count > 90
            assert f.reference_distribution is None
            assert f.distribution_distance is None


def test_from_dataframes_with_sampling(train_df_5k, synth_df_5k, test_df):
    evaluation_dataset = EvaluationDataset.from_dataframes(train_df_5k, synth_df_5k, test_df, rows=1000)

    assert evaluation_dataset is not None
    assert len(evaluation_dataset.reference) == 1000
    assert len(evaluation_dataset.output) == 1000
    assert len(evaluation_dataset.test) == 100

    assert len(evaluation_dataset.evaluation_fields) == 8


def test_degenerate_input(synth_df_5k, test_df):
    with pytest.raises(ValueError):
        EvaluationDataset.from_dataframes(None, synth_df_5k, test_df)
    with pytest.raises(ValueError):
        EvaluationDataset.from_dataframes(pd.DataFrame(), synth_df_5k, test_df)


def test_column_intersection(train_df, synth_df, test_df):
    train_df = train_df[["num", "num_cat"]]
    synth_df = synth_df[["num", "other"]]
    test_df = test_df[["num", "text"]]
    evaluation_dataset = EvaluationDataset.from_dataframes(train_df, synth_df, test_df)
    assert len(evaluation_dataset.evaluation_fields) == 1
    assert evaluation_dataset.evaluation_fields[0].name == "num"


def test_empty_column_intersection(train_df, synth_df):
    train_df = train_df[["num", "num_cat"]]
    synth_df = synth_df[["other", "text"]]
    with pytest.raises(ValueError):
        EvaluationDataset.from_dataframes(train_df, synth_df)


def test_empty_testdf_intersection(train_df, synth_df, test_df):
    train_df = train_df[["num", "num_cat"]]
    synth_df = synth_df[["num", "num_cat"]]
    test_df = test_df[["other", "text"]]
    with pytest.raises(ValueError):
        EvaluationDataset.from_dataframes(train_df, synth_df, test_df)


def test_get_columns_of_type(train_df):
    dataset = EvaluationDataset.from_dataframes(train_df, train_df, train_df)
    assert set(dataset.get_tabular_columns()) == set(
        ["num", "num_Int64", "num_cat", "num_cat_Int64", "small_cat", "boolean"]
    )
    assert set(dataset.get_nominal_columns()) == set(["num_cat", "num_cat_Int64", "small_cat", "boolean"])
    assert dataset.get_text_columns() == ["text"]
