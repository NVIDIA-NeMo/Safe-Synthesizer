# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum

from nemo_safe_synthesizer.observability import get_logger

# from types import TracebackType
from pydantic import BaseModel, Field

logger = get_logger(__name__)


class Grade(Enum):
    UNAVAILABLE = "Unavailable"
    VERY_POOR = "Very Poor"
    POOR = "Poor"
    MODERATE = "Moderate"
    GOOD = "Good"
    EXCELLENT = "Excellent"


class PrivacyGrade(Enum):
    UNAVAILABLE = "Unavailable"
    POOR = "Poor"
    MODERATE = "Moderate"
    GOOD = "Good"
    VERY_GOOD = "Very Good"
    EXCELLENT = "Excellent"


class EvaluationScore(BaseModel):
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
        if raw_score is None:
            return None
        return round(raw_score, 4)

    @staticmethod
    def clip_score(score: float | None) -> float | None:
        if score is None:
            return None
        # Cast score to a float with 1 decimal and limit to [0,10]
        return max(0.0, min(10.0, round(score, 1)))

    @staticmethod
    def score_to_grade(score: float | None, is_privacy=False) -> Grade | PrivacyGrade:
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
