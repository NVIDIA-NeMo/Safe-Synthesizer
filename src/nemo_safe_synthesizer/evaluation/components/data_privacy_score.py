# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import Field

from ...observability import get_logger
from ..data_model.evaluation_score import (
    EvaluationScore,
    PrivacyGrade,
)
from .component import Component
from .composite_score import CompositeScore

logger = get_logger(__name__)


class DataPrivacyScore(CompositeScore):
    """Aggregate privacy score -- mean of membership and attribute inference protection."""

    name: str = Field(default="Data Privacy Score")

    @staticmethod
    def from_components(components: list[Component] | Component, name: str = "Data Privacy Score") -> CompositeScore:
        """Compute the Data Privacy Score from privacy sub-metric components."""
        if isinstance(components, Component):
            return CompositeScore(score=components.score, name=name)
        if (
            components is None
            or len(components) == 0
            or all(
                [
                    c.score is None or c.score.score is None or c.score.grade is PrivacyGrade.UNAVAILABLE
                    for c in components
                ]
            )
        ):
            return DataPrivacyScore(score=EvaluationScore())

        # Take the mean
        total = 0.0
        total_components = 0
        for component in components:
            if component.score.score:
                total += component.score.score
                total_components += 1
        if total_components > 0:
            score = total / total_components
            return DataPrivacyScore(score=EvaluationScore.finalize_grade(raw_score=score, score=score, is_privacy=True))
        else:
            return DataPrivacyScore(score=EvaluationScore())
