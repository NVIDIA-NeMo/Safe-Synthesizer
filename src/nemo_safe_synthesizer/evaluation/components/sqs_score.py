# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import Field

from ...artifacts.analyzers.field_features import (
    FieldType,
)
from ...evaluation.components.column_distribution import ColumnDistribution
from ...evaluation.components.component import Component
from ...evaluation.components.composite_score import CompositeScore
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...observability import get_logger

logger = get_logger(__name__)


class SQSScore(CompositeScore):
    """Synthetic Quality Score -- weighted aggregate of quality sub-metrics.

    Combines column distribution stability, correlation stability, deep
    structure stability, text semantic similarity, and text structure
    similarity into a single 0--10 score weighted by the number of
    tabular vs. text columns.
    """

    name: str = Field(default="Synthetic Quality Score")

    @staticmethod
    def from_components(components: list[Component] | Component, name: str = "Synthetic Quality Score") -> SQSScore:
        """Compute the SQS from a list of quality sub-metric components."""
        if isinstance(components, Component):
            # wrap with a list and continue
            components = [components]
        if components is None or len(components) == 0:
            return SQSScore(score=EvaluationScore())

        # We need to recover the number of text and tabular fields to get proper weighted sum below.
        text_cols = 0
        tabular_cols = 0
        for c in components:
            # Look through the list of components for this one, it has all the field info.
            # If it is absent, something is really wrong -- we get that field info before even trying to make any components.
            # So if we don't have this, it's not worth trying to get consolation field counts from other components.
            if isinstance(c, ColumnDistribution):
                text_cols = len([f for f in c.evaluation_fields if f.reference_field_features.type == FieldType.TEXT])
                tabular_cols = len(c.evaluation_fields) - text_cols

        if tabular_cols + text_cols == 0:
            logger.warning("Failed to detect text/tabular columns for SQS.")
            return SQSScore(score=EvaluationScore())

        # Make it easier to pick out components by name
        component_dict = {c.name: c.score.score for c in components}

        score = SQSScore.get_overall_synthetic_data_quality_score(
            field_correlation_stability=component_dict.get("Column Correlation Stability"),
            principal_component_stability=component_dict.get("Deep Structure Stability"),
            field_distribution_stability=component_dict.get("Column Distribution Stability"),
            text_semantic_similarity=component_dict.get("Text Semantic Similarity"),
            text_structure_similarity=component_dict.get("Text Structure Similarity"),
            tabular_cols=tabular_cols,
            text_cols=text_cols,
        )
        return SQSScore(score=score)

    @staticmethod
    def get_overall_synthetic_data_quality_score(
        field_correlation_stability: float | None,
        principal_component_stability: float | None,
        field_distribution_stability: float | None,
        text_semantic_similarity: float | None,
        text_structure_similarity: float | None,
        tabular_cols: int,
        text_cols: int,
    ) -> EvaluationScore:
        """Compute the overall SQS from individual sub-metric scores.

        The tabular SQS is a weighted combination of correlation, distribution,
        and PCA stability.  The text SQS blends semantic and structural
        similarity.  The final score weights tabular and text SQS by their
        respective column counts.

        Args:
            field_correlation_stability: Correlation stability sub-score (0--10).
            principal_component_stability: PCA stability sub-score (0--10).
            field_distribution_stability: Distribution stability sub-score (0--10).
            text_semantic_similarity: Semantic similarity sub-score (0--10).
            text_structure_similarity: Structural similarity sub-score (0--10).
            tabular_cols: Number of tabular columns in the dataset.
            text_cols: Number of text columns in the dataset.

        Returns:
            A finalized ``EvaluationScore`` for the overall SQS.
        """
        # Compute SQS for tabular fields.
        tabular_sqs = None
        try:
            score = (
                0.84 * (principal_component_stability or 0)
                + 0.86 * (field_distribution_stability or 0)
                + (field_correlation_stability or 0)
            )
            denominator = (
                (0.84 if principal_component_stability is not None else 0)
                + (0.86 if field_distribution_stability is not None else 0)
                + (1.0 if field_correlation_stability is not None else 0)
            )
            if denominator != 0:
                tabular_sqs = round(score / denominator, 3)
        except Exception:
            logger.exception("Error calculating tabular SQS")

        # Compute text SQS... for text fields.
        text_sqs = None
        try:
            if text_semantic_similarity is None or text_structure_similarity is None:
                text_sqs = text_semantic_similarity or text_structure_similarity
            else:
                text_sqs = 0.7 * text_semantic_similarity + 0.3 * text_structure_similarity
        except Exception:
            logger.exception("Error calculating text SQS")

        if tabular_sqs is None or text_sqs is None:
            score = tabular_sqs or text_sqs or None
        else:
            score = (tabular_cols * tabular_sqs + text_cols * text_sqs) / (tabular_cols + text_cols)
        return EvaluationScore.finalize_grade(score, score)
