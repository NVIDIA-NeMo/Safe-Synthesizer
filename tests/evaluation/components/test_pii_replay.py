# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pandas as pd
import pytest

from nemo_safe_synthesizer.evaluation.components.pii_replay import PIIReplay
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets
from nemo_safe_synthesizer.pii_replacer.transform_result import ColumnStatistics

logger = logging.getLogger(__name__)


@pytest.mark.slow
def test_pii_replay(fixture_training_df_5k, fixture_synthetic_df_5k, fixture_test_df, fixture_column_statistics):
    """PII analysis on 5k rows - computationally expensive."""
    evaluation_datasets = EvaluationDatasets.from_dataframes(
        fixture_training_df_5k, fixture_synthetic_df_5k, fixture_test_df, fixture_column_statistics
    )
    pii_replay = PIIReplay.from_evaluation_datasets(evaluation_datasets)

    assert len(pii_replay.pii_replay_data) == 3

    assert pii_replay.pii_replay_data[0].column_name == "small_cat"
    assert pii_replay.pii_replay_data[0].pii_type == "some_cats"
    assert pii_replay.pii_replay_data[0].total_training_data == 1988
    assert pii_replay.pii_replay_data[0].unique_training_data == 2
    assert pii_replay.pii_replay_data[0].total_synthetic_data == 2044
    assert pii_replay.pii_replay_data[0].unique_synthetic_data == 2
    assert pii_replay.pii_replay_data[0].unique_synthetic_data_percentage == 100.0


def test_pii_replay_column_name_with_apostrophe():
    """Regression: a classified column whose name contains an apostrophe must not crash.

    PII replay previously filtered the synthetic column via ``DataFrame.query`` with a
    backtick-quoted column name. Under Python 3.12+ the query parser routes the backtick
    contents through CPython's tokenizer, so a name like ``judge's score`` raised
    ``SyntaxError: Failed to parse backticks`` (unterminated string literal).
    """
    col = "judge's score"
    training = pd.DataFrame({col: ["foo", "bar", "foo", "baz", "bar"]})
    synthetic = pd.DataFrame({col: ["foo", "foo", "bar", "qux", "qux"]})
    column_statistics = {
        col: ColumnStatistics(
            assigned_type="text",
            assigned_entity="name",
            detected_entity_counts={"name": 4},
            detected_entity_values={"name": {"foo", "bar"}},
            is_transformed=True,
            transform_functions={"fake"},
        )
    }

    evaluation_datasets = EvaluationDatasets.from_dataframes(
        training, synthetic, column_statistics=column_statistics, enable_sampling=False
    )
    pii_replay = PIIReplay.from_evaluation_datasets(evaluation_datasets)

    assert len(pii_replay.pii_replay_data) == 1
    datum = pii_replay.pii_replay_data[0]
    assert datum.column_name == col
    assert datum.pii_type == "name"
    assert datum.total_training_data == 4
    assert datum.unique_training_data == 2
    # synthetic values in {"foo", "bar"}: foo, foo, bar -> 3 rows, 2 unique
    assert datum.total_synthetic_data == 3
    assert datum.unique_synthetic_data == 2
    assert datum.unique_synthetic_data_percentage == 100.0
