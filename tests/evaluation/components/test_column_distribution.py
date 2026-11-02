# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from nemo_safe_synthesizer.evaluation.components.column_distribution import ColumnDistribution
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import (
    EvaluationDatasets,
)
from nemo_safe_synthesizer.pii_replacer import ColumnStatistics


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


def _cardholders(counts: dict[str, int]) -> pd.DataFrame:
    """Return one row per transaction, with each cardholder repeated as often as ``counts`` says."""
    names = [name for name, count in counts.items() for _ in range(count)]
    return pd.DataFrame({"cardholder": names, "amount": [float(index % 7) for index in range(len(names))]})


def _cardholder_field(training: pd.DataFrame, synthetic: pd.DataFrame, *, pii: bool):
    statistics = (
        {"cardholder": ColumnStatistics(assigned_type="structured", assigned_entity="full_name")} if pii else None
    )
    datasets = EvaluationDatasets.from_dataframes(training, synthetic, None, column_statistics=statistics)
    return next(field for field in datasets.evaluation_fields if field.name == "cardholder")


@pytest.mark.parametrize("pii", [True, False])
def test_pii_column_is_compared_by_value_shares_not_by_values(pii: bool) -> None:
    training = _cardholders({"Ada Lovelace": 60, "Alan Turing": 30, "Grace Hopper": 10})
    # The same shares under replaced names.
    synthetic = _cardholders({"Janet Ratliff": 60, "Camilo Torres": 30, "Edward Blau": 10})

    field = _cardholder_field(training, synthetic, pii=pii)

    assert field.distribution_distance is not None
    if pii:
        assert field.training_distribution == {"#1": 60.0, "#2": 30.0, "#3": 10.0}
        assert field.distribution_distance == pytest.approx(0.0)
    else:
        assert field.distribution_distance > 0.8


def test_pii_column_with_fewer_values_still_scores_worse() -> None:
    # Ten people with 20 transactions each, so the column is categorical.
    training = _cardholders({f"Person {index}": 20 for index in range(10)})
    collapsed = _cardholders({"Janet Ratliff": 140, "Camilo Torres": 60})
    matching = _cardholders({f"Replaced {index}": 20 for index in range(10)})

    collapsed_field = _cardholder_field(training, collapsed, pii=True)
    matching_field = _cardholder_field(training, matching, pii=True)

    assert matching_field.distribution_distance == pytest.approx(0.0)
    assert collapsed_field.distribution_distance is not None
    assert collapsed_field.distribution_distance > 0.5
