# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from ...observability import get_logger

logger = get_logger(__name__)


class Grade(Enum):
    """Qualitative quality grade for a synthetic data metric."""

    UNAVAILABLE = "Unavailable"
    VERY_POOR = "Very Poor"
    POOR = "Poor"
    MODERATE = "Moderate"
    GOOD = "Good"
    EXCELLENT = "Excellent"


class PrivacyGrade(Enum):
    """Qualitative privacy grade for a privacy metric."""

    UNAVAILABLE = "Unavailable"
    POOR = "Poor"
    MODERATE = "Moderate"
    GOOD = "Good"
    VERY_GOOD = "Very Good"
    EXCELLENT = "Excellent"


class EvaluationScore(BaseModel):
    """Numeric and qualitative score for a single evaluation metric.

    Carries the raw measurement, a 0--10 scaled score, a qualitative grade,
    and optional notes (warnings or error messages). Use ``finalize_grade``
    to construct a fully populated instance from raw and scaled values.
    """

    raw_score: float | None = Field(
        description="The raw score, None if the score failed to be calculated.", default=None, ge=0
    )
    grade: Grade | PrivacyGrade = Field(
        description="The qualitative grade for this field ('Good', etc).", default=Grade.UNAVAILABLE
    )
    score: float | None = Field(
        description="The scaled score (0 to 10), None if the score failed to be calculated.", default=None, ge=0, le=10
    )
    notes: str | None = Field(description="A string field for relaying warning or error messages.", default=None)
    # FIXME someone reviewing this let me know if this is a good idea or too much
    # traceback: TracebackType | None = Field(
    #     description="Traceback object if score fails to render.",
    #     default=None
    # )

    @staticmethod
    def round_raw_score(raw_score: float | None) -> float | None:
        """Round the raw score to four decimal places."""
        if raw_score is None:
            return None
        return round(raw_score, 4)

    @staticmethod
    def clip_score(score: float | None) -> float | None:
        """Clip and round a score to one decimal in [0, 10]."""
        if score is None:
            return None
        # Cast score to a float with 1 decimal and limit to [0,10]
        return max(0.0, min(10.0, round(score, 1)))

    @staticmethod
    def score_to_grade(score: float | None, is_privacy=False) -> Grade | PrivacyGrade:
        """Map a 0--10 numeric score to a qualitative grade."""
        if score is None:
            return PrivacyGrade.UNAVAILABLE if is_privacy else Grade.UNAVAILABLE
        idx = int(score) // 2
        # Constrain to [0,4]
        idx = max(0, min(4, idx))
        # Get all the (Privacy)Grades that are not UNAVAILABLE
        if is_privacy:
            grades = [g for g in PrivacyGrade][1:]
        else:
            grades = [g for g in Grade][1:]
        return grades[idx]

    @staticmethod
    def finalize_grade(raw_score: float | None, score: float | None, is_privacy=False) -> EvaluationScore:
        """Build a complete ``EvaluationScore`` from raw and scaled values.

        Rounds, clips, and maps to a grade in one step. Returns a default
        (unavailable) score on failure.
        """
        default_score = EvaluationScore()
        try:
            raw_score = EvaluationScore.round_raw_score(raw_score)
            score = EvaluationScore.clip_score(score)
            grade = EvaluationScore.score_to_grade(score, is_privacy=is_privacy)
            return EvaluationScore(raw_score=raw_score, grade=grade, score=score)
        except Exception as e:
            logger.exception("Could not finalize grade.")
            default_score.notes = str(e)
        return default_score
