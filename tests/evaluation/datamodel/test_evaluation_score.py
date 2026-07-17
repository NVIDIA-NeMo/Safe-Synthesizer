# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import EvaluationScore, Grade, PrivacyGrade


def test_raw_scores_are_rounded():
    raw = 1.23456789
    score = EvaluationScore.finalize_grade(raw_score=raw, score=raw)
    assert score.raw_score == 1.2346
    assert score.score == 1.2
    # Why can't you be more like test_scores_are_clipped
    assert score.grade == Grade.POOR
    assert score.notes is None


def test_scores_are_clipped():
    s = 10.5
    score = EvaluationScore.finalize_grade(raw_score=s, score=s)
    assert score.raw_score == 10.5
    assert score.score == 10.0
    assert score.grade == Grade.EXCELLENT

    # Negative number actually catches validation, we have ge=0.
    s = -1.0
    score = EvaluationScore.finalize_grade(raw_score=s, score=s)
    assert score.raw_score is None
    assert score.score is None
    assert score.grade == Grade.UNAVAILABLE
    assert score.notes is not None and "1 validation error for EvaluationScore" in score.notes


def test_nones_okay():
    s = None
    score = EvaluationScore.finalize_grade(raw_score=s, score=s)
    # Get thee to a Nonery
    assert score.raw_score is None
    assert score.score is None
    assert score.grade == Grade.UNAVAILABLE
    assert score.notes is None


def test_privacy_grade():
    s = None
    score = EvaluationScore.finalize_grade(raw_score=s, score=s, is_privacy=True)
    assert score.raw_score is None
    assert score.score is None
    assert score.grade == PrivacyGrade.UNAVAILABLE

    raw = 1.23456789
    score = EvaluationScore.finalize_grade(raw_score=raw, score=raw, is_privacy=True)
    assert score.raw_score == 1.2346
    assert score.score == 1.2
    # Still crappy but with more privacy.
    assert score.grade == PrivacyGrade.POOR


@pytest.mark.parametrize(
    ("score", "expected_quality", "expected_privacy"),
    [
        (0.0, Grade.POOR, PrivacyGrade.POOR),
        (1.9, Grade.POOR, PrivacyGrade.POOR),
        (2.0, Grade.MODERATE, PrivacyGrade.MODERATE),
        (3.9, Grade.MODERATE, PrivacyGrade.MODERATE),
        (4.0, Grade.GOOD, PrivacyGrade.GOOD),
        (5.9, Grade.GOOD, PrivacyGrade.GOOD),
        (6.0, Grade.VERY_GOOD, PrivacyGrade.VERY_GOOD),
        (7.9, Grade.VERY_GOOD, PrivacyGrade.VERY_GOOD),
        (8.0, Grade.EXCELLENT, PrivacyGrade.EXCELLENT),
        (10.0, Grade.EXCELLENT, PrivacyGrade.EXCELLENT),
    ],
)
def test_quality_and_privacy_grades_follow_documented_ranges(
    score: float,
    expected_quality: Grade,
    expected_privacy: PrivacyGrade,
) -> None:
    assert EvaluationScore.score_to_grade(score) is expected_quality
    assert EvaluationScore.score_to_grade(score, is_privacy=True) is expected_privacy
