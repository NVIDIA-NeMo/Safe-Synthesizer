# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field

from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.components.component import Component
from ...evaluation.constants import JOB_COUNT
from ...evaluation.data_model.evaluation_dataset import EvaluationDataset
from ...evaluation.data_model.evaluation_field import EvaluationField
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...evaluation.statistics import stats
from ...observability import get_logger
from . import multi_modal_figures as figures

logger = get_logger(__name__)


class Correlation(Component):
    """Column Correlation Stability metric.

    Computes per-column-pair correlations (Pearson, Theil's U, Correlation
    Ratio) for both reference and output dataframes, then scores the mean
    absolute difference.

    Attributes:
        reference_correlation: Correlation matrix for the reference data.
        output_correlation: Correlation matrix for the output data.
        correlation_difference: Element-wise absolute difference of the two matrices.
    """

    name: str = Field(default="Column Correlation Stability")
    reference_correlation: pd.DataFrame | None = Field(default=None)
    output_correlation: pd.DataFrame | None = Field(default=None)
    correlation_difference: pd.DataFrame | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def jinja_context(self):
        """Template context with combined correlation heatmap figure."""
        d = super().jinja_context
        d["anchor_link"] = "#correlation-stability"

        if Component.is_nonempty([self.reference_correlation, self.output_correlation, self.correlation_difference]):
            d["figure"] = figures.generate_combined_correlation_figure(
                reference_correlation=self.reference_correlation,  # ty: ignore[invalid-argument-type]
                output_correlation=self.output_correlation,  # ty: ignore[invalid-argument-type]
                correlation_difference=self.correlation_difference,  # ty: ignore[invalid-argument-type]
            ).to_html(full_html=False, include_plotlyjs=False)
        else:
            d["figure"] = None
        return d

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None = None
    ) -> Correlation:
        """Compute correlation matrices and the correlation stability score."""
        # We only want to use these types for correlation.
        tabular_columns = evaluation_dataset.get_tabular_columns()
        # We use different calculations (Thiel's U) for nominal columns.
        nominal_columns = evaluation_dataset.get_nominal_columns()

        (
            reference_correlation,
            output_correlation,
            correlation_difference,
            mean_absolute_error,
        ) = Correlation._get_correlation_calculations(
            reference=evaluation_dataset.reference[tabular_columns],  # ty: ignore[invalid-argument-type]
            output=evaluation_dataset.output[tabular_columns],  # ty: ignore[invalid-argument-type]
            nominal_columns=nominal_columns,
            fields=evaluation_dataset.evaluation_fields,
        )
        evaluation_score = Correlation._get_field_correlation_stability(mean_absolute_error)
        return Correlation(
            reference_correlation=reference_correlation,
            output_correlation=output_correlation,
            correlation_difference=correlation_difference,
            score=evaluation_score,
        )

    @staticmethod
    def _get_correlation_calculations(
        reference: pd.DataFrame,
        output: pd.DataFrame,
        nominal_columns: list[str],
        fields: list[EvaluationField],
    ):
        """Compute reference and output correlation matrices and their difference.

        Args:
            reference: Tabular reference dataframe.
            output: Tabular output dataframe.
            nominal_columns: Columns to treat as categorical in correlation methods.
            fields: Per-column evaluation metadata (unused, reserved for future use).

        Returns:
            Tuple of (reference_correlation, output_correlation,
            correlation_difference, mean_absolute_error).
        """
        # Legacy code has constant for default value of 4 and function to see if we can go higher.
        # See what we want to do now that we work for a GPU manufacturer.
        job_count = JOB_COUNT

        mean_absolute_error = None
        reference_correlation = None
        output_correlation = None
        correlation_difference = None

        try:
            if reference.shape[1] < 2:
                logger.info("Less than two correlatable columns found. Skipping correlation calculations.")
                return (
                    reference_correlation,
                    output_correlation,
                    correlation_difference,
                    mean_absolute_error,
                )

            reference_correlation = stats.calculate_correlation(
                reference, nominal_columns=nominal_columns, job_count=job_count
            )

            output_correlation = stats.calculate_correlation(
                output, nominal_columns=nominal_columns, job_count=job_count
            )

            correlation_difference = (reference_correlation - output_correlation).abs()

            mean_absolute_error = (reference_correlation - output_correlation).abs().mean().mean()
        except Exception:
            logger.exception("Failure during correlation calculations.")
        return (
            reference_correlation,
            output_correlation,
            correlation_difference,
            mean_absolute_error,
        )

    @staticmethod
    def _get_field_correlation_stability(
        mean_absolute_error: float | None,
    ) -> EvaluationScore:
        """Convert mean absolute correlation error to a graded stability score.

        Args:
            mean_absolute_error: Average absolute difference between reference
                and output correlation matrices. ``None`` if calculation failed.

        Returns:
            A finalized ``EvaluationScore`` for correlation stability.
        """
        if mean_absolute_error is None or np.isnan(mean_absolute_error):
            return EvaluationScore()
        if mean_absolute_error > 0.46:
            score = 0
        elif mean_absolute_error < 0.01:
            score = 10
        else:
            score = 37.562 * mean_absolute_error**2 - 35.643 * mean_absolute_error + 10.056
        return EvaluationScore.finalize_grade(mean_absolute_error, score)
