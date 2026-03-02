# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property

from pydantic import Field

from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.components.component import Component
from ...evaluation.data_model.evaluation_dataset import EvaluationDataset
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...evaluation.statistics import stats


class DatasetStatistics(Component):
    """Summary statistics for the reference and output datasets.

    Reports row/column counts, missing-value percentages, and the number
    of memorized (verbatim-repeated) rows. This component does not produce
    a numeric score -- it provides context for the HTML report.
    """

    name: str = Field(default="Dataset Statistics")
    # Copy these out for rendering convenience
    reference_rows: int = Field(default=0, ge=0)
    reference_cols: int = Field(default=0, ge=0)
    reference_missing: int = Field(default=0, ge=0)
    output_rows: int = Field(default=0, ge=0)
    output_cols: int = Field(default=0, ge=0)
    output_missing: int = Field(default=0, ge=0)
    memorized_lines: int = Field(default=0, ge=0)

    @cached_property
    def jinja_context(self):
        """Template context merging all dataset summary fields into the base context."""
        d = super().jinja_context
        stats = self.model_dump()
        # Dump all the other fields, but don't overwrite pre-prepped stuff in d
        stats.update(d)
        return stats

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None = None
    ) -> DatasetStatistics:
        """Compute summary statistics from the evaluation dataset."""
        return DatasetStatistics(
            score=EvaluationScore(),
            reference_rows=evaluation_dataset.reference_rows,
            reference_cols=evaluation_dataset.reference_cols,
            reference_missing=int(stats.percent_missing(evaluation_dataset.reference)),
            output_rows=evaluation_dataset.output_rows,
            output_cols=evaluation_dataset.output_cols,
            output_missing=int(stats.percent_missing(evaluation_dataset.output)),
            memorized_lines=evaluation_dataset.memorized_lines,
        )
