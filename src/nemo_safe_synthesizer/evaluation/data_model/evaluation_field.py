# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from functools import reduce

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
from pydantic import BaseModel, Field

from ...artifacts.analyzers.field_features import (
    FieldFeatures,
    FieldType,
    describe_field,
)
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...evaluation.statistics import stats
from ...observability import get_logger
from ...pii_replacer.transform_result import ColumnStatistics

logger = get_logger(__name__)

HIGHLY_UNIQUE_TYPES = [FieldType.OTHER, FieldType.TEXT, FieldType.EMPTY]


class EvaluationField(BaseModel):
    """Per-column evaluation metadata and distribution scores."""

    name: str = Field(description="Column name from the original dataframe.")
    reference_field_features: FieldFeatures = Field(
        description="Field type and descriptive statistics for the reference column."
    )
    output_field_features: FieldFeatures = Field(
        description="Field type and descriptive statistics for the output column."
    )
    reference_distribution: dict | None = Field(description="Binned distribution dict for the reference column.")
    output_distribution: dict | None = Field(description="Binned distribution dict for the output column.")
    distribution_distance: float | None = Field(description="Jensen-Shannon distance between the two distributions.")
    distribution_stability: EvaluationScore | None = Field(
        description="Graded score derived from the distribution distance."
    )
    column_statistics: ColumnStatistics | None = Field(
        description="PII entity counts and transform metadata, if available."
    )

    @staticmethod
    def from_series(
        name: str,
        reference: pd.Series,
        output: pd.Series,
        column_statistics: ColumnStatistics | None = None,
    ) -> EvaluationField:
        """Build an ``EvaluationField`` from paired reference/output column.

        Normally called internally by ``EvaluationDataset``; direct use is
        rarely needed.

        Args:
            name: Column name.
            reference: Reference column data.
            output: Output (synthetic) column data.
            column_statistics: PII entity metadata to attach, if available.

        Returns:
            A fully populated ``EvaluationField`` with computed distributions
            and stability score.
        """
        reference_field_features = describe_field(name, reference)
        output_field_features = describe_field(name, output)
        # TODO This was a config setting to explicitly force fields to be categorical.
        # if is_categorical:
        #     reference_field_features.type = FieldType.CATEGORICAL
        #     output_field_features.type = FieldType.CATEGORICAL

        # TODO Synthesizer only, but not making conditional until more new config/control is baked up.
        if reference_field_features.type == FieldType.NUMERIC and output_field_features.type == FieldType.NUMERIC:
            bins = stats.get_numeric_distribution_bins(reference, output)
            reference_distribution = stats.get_numeric_field_distribution(reference, bins)
            output_distribution = stats.get_numeric_field_distribution(output, bins)
            distribution_distance = stats.compute_distribution_distance(reference_distribution, output_distribution)
            distribution_stability = EvaluationField.get_field_distribution_stability(distribution_distance)
        else:
            if is_integer_dtype(reference) or is_integer_dtype(output):
                try:
                    # If the other column contains float values or has None values with object dtype,
                    # first cast it to float, round the values, and then convert to pd.Int64Dtype.
                    # This allows missing values to be properly handled and enables meaningful comparisons.
                    reference = reference.astype(float).round().astype(pd.Int64Dtype())
                    output = output.astype(float).round().astype(pd.Int64Dtype())
                except ValueError:
                    # The other column has something weird that is not a float, just keep going.
                    pass
            if (
                reference_field_features.count == 0
                or output_field_features.count == 0
                or reference_field_features.type in HIGHLY_UNIQUE_TYPES
                or output_field_features.type in HIGHLY_UNIQUE_TYPES
            ):
                reference_distribution = None
                output_distribution = None
                distribution_distance = None
                distribution_stability = None
            else:
                reference_distribution = stats.get_categorical_field_distribution(reference)
                output_distribution = stats.get_categorical_field_distribution(output)
                distribution_distance = stats.compute_distribution_distance(reference_distribution, output_distribution)
                distribution_stability = EvaluationField.get_field_distribution_stability(distribution_distance)

        return EvaluationField(
            name=name,
            reference_field_features=reference_field_features,
            output_field_features=output_field_features,
            reference_distribution=reference_distribution,
            output_distribution=output_distribution,
            distribution_distance=distribution_distance,
            distribution_stability=distribution_stability,
            column_statistics=column_statistics,
        )

    @staticmethod
    def get_average_divergence(fields: list[EvaluationField]) -> float:
        """Compute the mean Jensen-Shannon divergence across a list of fields."""
        if len(fields) > 0:
            average_divergence = reduce(
                lambda x, y: x + y,
                [f.distribution_distance for f in fields if f.distribution_distance is not None],
                0.0,  # ENGPROD-6, default accumulator value may be required
            ) / len(fields)
            return average_divergence
        return 0.0

    @staticmethod
    def text_js_scaling_func(average_divergence: float) -> float:
        """Scale average JS divergence for text data using a linear equation.

        Args:
            average_divergence: Mean JS divergence across text fields.

        Returns:
            A score in the range ``[1.5, 10]``.
        """
        # Scaling with linear equation penalizes the lower range scores drastically, setting the lower values to 15 instead of 0.
        # More explained in this doc.
        if average_divergence > 0.8:
            score = 1.5
        elif average_divergence < 0.12:
            score = 10.0
        else:
            score = -12.5 * average_divergence + 11.5
        return score

    @staticmethod
    def tabular_js_scaling_func(average_divergence: float) -> float:
        """Scale average JS divergence for tabular data using a quadratic equation.

        Args:
            average_divergence: Mean JS divergence across tabular fields.

        Returns:
            A score in the range ``[0, 10]``.
        """
        if average_divergence > 0.99:
            score = 0.0
        elif average_divergence < 0.02:
            score = 10.0
        else:
            score = 7.44 * average_divergence**2 - 16.646 * average_divergence + 10.305
        return score

    @staticmethod
    def get_field_distribution_stability(
        average_divergence: float,
        js_scaling_func: Callable[[float], float] | None = None,
    ) -> EvaluationScore:
        """Convert an average JS divergence into a graded ``EvaluationScore``.

        Args:
            average_divergence: Mean JS divergence across fields.
            js_scaling_func: Scaling function mapping divergence to a 0--10
                score. Defaults to ``tabular_js_scaling_func``.

        Returns:
            A finalized ``EvaluationScore`` with grade and scaled score.
        """
        js_scaling_func = js_scaling_func or EvaluationField.tabular_js_scaling_func
        try:
            if np.isnan(average_divergence):
                return EvaluationScore()
            score = js_scaling_func(average_divergence)
            return EvaluationScore.finalize_grade(average_divergence, score)
        except Exception as e:
            logger.exception("Failed to calculate Field Distribution Stability SQS")
            return EvaluationScore(notes=str(e))
