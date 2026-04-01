# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property

from pydantic import Field

from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.components.component import Component
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...evaluation.statistics import stats


class DatasetStatistics(Component):
    """Summary statistics for the training and synthetic datasets.

    Reports row/column counts, missing-value percentages, and the number
    of memorized (verbatim-repeated) rows. This component does not produce
    a numeric score -- it provides context for the HTML report.
    """

    name: str = Field(default="Dataset Statistics")
    # Copy these out for rendering convenience
    training_rows: int = Field(
        default=0, ge=0, description="Row count of the training dataframe used for evaluation."
    )
    training_cols: int = Field(
        default=0, ge=0, description="Column count of the training dataframe used for evaluation."
    )
    training_missing: int = Field(
        default=0, ge=0, description="Percentage of missing values in the training dataframe."
    )
    synthetic_rows: int = Field(
        default=0, ge=0, description="Row count of the synthetic dataframe used for evaluation."
    )
    synthetic_cols: int = Field(
        default=0, ge=0, description="Column count of the synthetic dataframe used for evaluation."
    )
    synthetic_missing: int = Field(
        default=0, ge=0, description="Percentage of missing values in the synthetic dataframe."
    )
    memorized_lines: int = Field(
        default=0, ge=0, description="Number of exact row matches between training and synthetic."
    )

    @cached_property
    def jinja_context(self):
        """Template context merging all dataset summary fields into the base context."""
        d = super().jinja_context
        stats = self.model_dump()
        # Dump all the other fields, but don't overwrite pre-prepped stuff in d
        stats.update(d)
        return stats

    @staticmethod
    def from_evaluation_datasets(
        evaluation_datasets: EvaluationDatasets, config: SafeSynthesizerParameters | None = None
    ) -> DatasetStatistics:
        """Compute summary statistics from the evaluation dataset."""
        return DatasetStatistics(
            score=EvaluationScore(),
            training_rows=evaluation_datasets.training_rows,
            training_cols=evaluation_datasets.training_cols,
            training_missing=int(stats.percent_missing(evaluation_datasets.training)),
            synthetic_rows=evaluation_datasets.synthetic_rows,
            synthetic_cols=evaluation_datasets.synthetic_cols,
            synthetic_missing=int(stats.percent_missing(evaluation_datasets.synthetic)),
            memorized_lines=evaluation_datasets.memorized_lines,
        )
