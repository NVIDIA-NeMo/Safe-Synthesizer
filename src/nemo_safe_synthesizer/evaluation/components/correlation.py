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
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ...evaluation.data_model.evaluation_field import EvaluationField
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...evaluation.statistics import stats
from ...observability import get_logger
from . import multi_modal_figures as figures

logger = get_logger(__name__)


class Correlation(Component):
    """Column Correlation Stability metric.

    Computes per-column-pair correlations (Pearson, Theil's U, Correlation
    Ratio) for both training and synthetic dataframes, then scores the mean
    absolute difference.
    """

    name: str = Field(default="Column Correlation Stability")
    training_correlation: pd.DataFrame | None = Field(
        default=None, description="Correlation matrix for the training data."
    )
    synthetic_correlation: pd.DataFrame | None = Field(
        default=None, description="Correlation matrix for the synthetic data."
    )
    correlation_difference: pd.DataFrame | None = Field(
        default=None, description="Element-wise absolute difference of the two matrices."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def jinja_context(self):
        """Template context with combined correlation heatmap figure."""
        d = super().jinja_context
        d["anchor_link"] = "#correlation-stability"

        if Component.is_nonempty([self.training_correlation, self.synthetic_correlation, self.correlation_difference]):
            d["figure"] = figures.generate_combined_correlation_figure(
                training_correlation_df=self.training_correlation,  # ty: ignore[invalid-argument-type]
                synthetic_correlation_df=self.synthetic_correlation,  # ty: ignore[invalid-argument-type]
                correlation_difference=self.correlation_difference,  # ty: ignore[invalid-argument-type]
            ).to_html(full_html=False, include_plotlyjs=False)
        else:
            d["figure"] = None
        return d

    @staticmethod
    def from_evaluation_datasets(
        evaluation_datasets: EvaluationDatasets, config: SafeSynthesizerParameters | None = None
    ) -> Correlation:
        """Compute correlation matrices and the correlation stability score."""
        # We only want to use these types for correlation.
        tabular_columns = evaluation_datasets.get_tabular_columns()
        # We use different calculations (Theil's U) for nominal columns.
        nominal_columns = evaluation_datasets.get_nominal_columns()

        (
            training_correlation,
            synthetic_correlation,
            correlation_difference,
            mean_absolute_error,
        ) = Correlation._get_correlation_calculations(
            training_df=evaluation_datasets.training[tabular_columns],  # ty: ignore[invalid-argument-type]
            synthetic_df=evaluation_datasets.synthetic[tabular_columns],  # ty: ignore[invalid-argument-type]
            nominal_columns=nominal_columns,
            fields=evaluation_datasets.evaluation_fields,
        )
        evaluation_score = Correlation._get_field_correlation_stability(mean_absolute_error)
        return Correlation(
            training_correlation=training_correlation,
            synthetic_correlation=synthetic_correlation,
            correlation_difference=correlation_difference,
            score=evaluation_score,
        )

    @staticmethod
    def _get_correlation_calculations(
        training_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        nominal_columns: list[str],
        fields: list[EvaluationField],
    ):
        """Compute training and synthetic correlation matrices and their difference.

        Args:
            training_df: Tabular training dataframe.
            synthetic_df: Tabular synthetic dataframe.
            nominal_columns: Columns to treat as categorical in correlation methods.
            fields: Per-column evaluation metadata (unused, reserved for future use).

        Returns:
            Tuple of (training_correlation, synthetic_correlation,
            correlation_difference, mean_absolute_error).
        """
        # Legacy code has constant for default value of 4 and function to see if we can go higher.
        # See what we want to do now that we work for a GPU manufacturer.
        job_count = JOB_COUNT

        mean_absolute_error = None
        training_correlation = None
        synthetic_correlation = None
        correlation_difference = None

        try:
            if training_df.shape[1] < 2:
                logger.info("Less than two correlatable columns found. Skipping correlation calculations.")
                return (
                    training_correlation,
                    synthetic_correlation,
                    correlation_difference,
                    mean_absolute_error,
                )

            training_correlation = stats.calculate_correlation(
                training_df, nominal_columns=nominal_columns, job_count=job_count
            )

            synthetic_correlation = stats.calculate_correlation(
                synthetic_df, nominal_columns=nominal_columns, job_count=job_count
            )

            correlation_difference = (training_correlation - synthetic_correlation).abs()

            mean_absolute_error = (training_correlation - synthetic_correlation).abs().mean().mean()
        except Exception:
            logger.exception("Failure during correlation calculations.")
        return (
            training_correlation,
            synthetic_correlation,
            correlation_difference,
            mean_absolute_error,
        )

    @staticmethod
    def _get_field_correlation_stability(
        mean_absolute_error: float | None,
    ) -> EvaluationScore:
        """Convert mean absolute correlation error to a graded stability score.

        Args:
            mean_absolute_error: Average absolute difference between training
                and synthetic correlation matrices. ``None`` if calculation failed.

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
