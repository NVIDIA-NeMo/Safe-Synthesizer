# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property

import nemo_safe_synthesizer.evaluation.components.multi_modal_figures as figures
import numpy as np
import pandas as pd
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.evaluation.components.component import Component
from nemo_safe_synthesizer.evaluation.constants import JOB_COUNT
from nemo_safe_synthesizer.evaluation.data_model.evaluation_dataset import (
    EvaluationDataset,
)
from nemo_safe_synthesizer.evaluation.data_model.evaluation_field import (
    EvaluationField,
)
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import (
    EvaluationScore,
)
from nemo_safe_synthesizer.evaluation.statistics import stats
from nemo_safe_synthesizer.observability import get_logger
from pydantic import ConfigDict, Field

logger = get_logger(__name__)


class Correlation(Component):
    name: str = Field(default="Column Correlation Stability")
    reference_correlation: pd.DataFrame | None = Field(default=None)
    output_correlation: pd.DataFrame | None = Field(default=None)
    correlation_difference: pd.DataFrame | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def jinja_context(self):
        d = super().jinja_context
        d["anchor_link"] = "#correlation-stability"

        if Component.is_nonempty([self.reference_correlation, self.output_correlation, self.correlation_difference]):
            d["figure"] = figures.generate_combined_correlation_figure(
                reference_correlation=self.reference_correlation,
                output_correlation=self.output_correlation,
                correlation_difference=self.correlation_difference,
            ).to_html(full_html=False, include_plotlyjs=False)
        else:
            d["figure"] = None
        return d

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None = None
    ) -> Correlation:
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
        """
        Calculate everything correlation related -- the actual L and R correlation matrices, their difference
        and the mean absolute error.

        Args:
            reference: pd.DataFrame
            output: pd.DataFrame
            nominal_columns: nominal columns to pass into correlation methods

        Returns: tuple of reference_correlation, output_correlation, correlation_difference, mean_absolute_error

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
        """
        Calculate the field_correlation_stability SQS based on the MAE, one of our correlation calculations.

        Args:
            mean_absolute_error: float, as produced by _get_correlation_calculations

        Returns: the field_correlation_stability SQS

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
