# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
from pydantic import Field

from ...artifacts.analyzers.field_features import FieldType
from ...config.evaluation_report_data import Distribution
from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.components.component import Component
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ...evaluation.data_model.evaluation_field import EvaluationField
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...observability import get_logger
from ...pii_replacer.transform_result import ColumnStatistics
from . import multi_modal_figures as figures

logger = get_logger(__name__)


def distribution_for_field(f: EvaluationField, training: pd.Series, synthetic: pd.Series) -> Distribution | None:
    """Return the values charted for one tabular column, or ``None`` if it has no chart or no numeric training values.

    A numeric training column always gets binned values: a bar per distinct numeric value would name individual
    training values.
    """
    if f.training_field_features.type == FieldType.NUMERIC:
        return figures.numeric_distribution(training, synthetic)
    if f.training_distribution is not None and f.synthetic_distribution is not None:
        return figures.categorical_distribution(f.training_distribution, f.synthetic_distribution, len(training))
    return None


class ColumnDistribution(Component):
    """Column Distribution Stability metric.

    Computes per-column Jensen-Shannon divergence between training and
    synthetic distributions, averages across all tabular columns, and maps
    the result to a 0--10 score.  Also carries data for the per-column
    histogram figures and the Training Data Columns table in the HTML report.
    """

    name: str = Field(default="Column Distribution Stability")
    # Keep a copy to simplify rendering
    column_statistics: dict[str, ColumnStatistics] | None = Field(
        default=None, description="Per-column PII entity and transform metadata."
    )
    evaluation_fields: list[EvaluationField] = Field(
        default=list(), description="Per-column evaluation metadata and distribution scores."
    )

    @staticmethod
    def from_evaluation_datasets(
        evaluation_datasets: EvaluationDatasets, config: SafeSynthesizerParameters | None = None
    ) -> ColumnDistribution:
        """Compute column distribution stability from the evaluation dataset."""
        tabular_columns = set(evaluation_datasets.get_tabular_columns())
        tabular_fields = [f for f in evaluation_datasets.evaluation_fields if f.name in tabular_columns]
        if tabular_fields:
            average_divergence = EvaluationField.get_average_divergence(tabular_fields)
            score = EvaluationField.get_field_distribution_stability(average_divergence)
            return ColumnDistribution(
                score=score,
                column_statistics=evaluation_datasets.column_statistics,
                evaluation_fields=evaluation_datasets.evaluation_fields,
            )
        else:
            return ColumnDistribution(
                score=EvaluationScore(notes="No tabular columns detected."),
                column_statistics=evaluation_datasets.column_statistics,
                evaluation_fields=evaluation_datasets.evaluation_fields,
            )
