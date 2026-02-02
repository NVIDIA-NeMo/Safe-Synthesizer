# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

# Skip all tests in this module if sentence_transformers is not available
pytest.importorskip(
    "sentence_transformers",
    reason="sentence_transformers is required for these tests (install with: uv sync --extra cpu)",
)

# from nemo_safe_synthesizer.evaluation.components.attribute_inference_protection.attribute_inference_protection import AttributeInferenceProtection
from nemo_safe_synthesizer.evaluation.components.column_distribution import ColumnDistribution
from nemo_safe_synthesizer.evaluation.components.correlation import Correlation
from nemo_safe_synthesizer.evaluation.components.deep_structure import DeepStructure

# from nemo_safe_synthesizer.evaluation.components.composite_score.data_privacy_score import DataPrivacyScore
from nemo_safe_synthesizer.evaluation.components.sqs_score import SQSScore
from nemo_safe_synthesizer.evaluation.components.text_semantic_similarity import TextSemanticSimilarity
from nemo_safe_synthesizer.evaluation.components.text_structure_similarity import TextStructureSimilarity

# from nemo_safe_synthesizer.evaluation.components.membership_inference_protection.membership_inference_protection import MembershipInferenceProtection
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import EvaluationScore, Grade


@pytest.fixture
def column_correlation_stability():
    return Correlation(
        score=EvaluationScore(name="Column Correlation Stability", raw_score=5.0, score=5.0, grade=Grade.UNAVAILABLE)
    )


@pytest.fixture
def deep_structure_stability():
    return DeepStructure(
        score=EvaluationScore(name="Deep Structure Stability", raw_score=2.0, score=2.0, grade=Grade.UNAVAILABLE)
    )


@pytest.fixture
def column_distribution_stability(evaluation_dataset_5k):
    return ColumnDistribution.from_evaluation_dataset(evaluation_dataset_5k)


@pytest.fixture
def text_semantic_stability():
    return TextSemanticSimilarity(
        score=EvaluationScore(name="Text Semantic Similarity", raw_score=7.0, score=7.0, grade=Grade.UNAVAILABLE)
    )


@pytest.fixture
def text_structure_stability():
    return TextStructureSimilarity(
        score=EvaluationScore(name="Text Structure Similarity", raw_score=3.0, score=3.0, grade=Grade.UNAVAILABLE)
    )


# @pytest.fixture
# def aia():
#     return AttributeInferenceProtection(score=EvaluationScore(
#         name="aia",
#         raw_score=2.0,
#         score=2.0,
#         grade=PrivacyGrade.UNAVAILABLE
#     ))


# @pytest.fixture
# def mia():
#     return MembershipInferenceProtection(score=EvaluationScore(
#         name="mia",
#         raw_score=8.0,
#         score=8.0,
#         grade=PrivacyGrade.UNAVAILABLE
#     ))


def test_sqs_score_tabular(column_correlation_stability, deep_structure_stability, column_distribution_stability):
    # Order agnostic, we check the names
    sqs = SQSScore.from_components(
        [deep_structure_stability, column_distribution_stability, column_correlation_stability]
    )
    assert sqs.name == "Synthetic Quality Score"
    assert sqs.score.raw_score == 5.627
    assert sqs.score.score == 5.6
    assert sqs.score.grade == Grade.MODERATE
    assert sqs.score.notes is None


def test_sqs_score_needs_columns(column_correlation_stability, column_distribution_stability):
    # This one fails to yield a proper score, no column info
    sqs = SQSScore.from_components([column_correlation_stability])
    assert sqs.name == "Synthetic Quality Score"
    assert sqs.score.raw_score is None
    assert sqs.score.score is None
    assert sqs.score.grade == Grade.UNAVAILABLE
    assert sqs.score.notes is None

    # This one will work
    sqs = SQSScore.from_components([column_distribution_stability])
    assert sqs.name == "Synthetic Quality Score"
    assert sqs.score.raw_score == 9.9
    assert sqs.score.score == 9.9
    assert sqs.score.grade == Grade.EXCELLENT
    assert sqs.score.notes is None


def test_sqs_score_from_nothing():
    # With absolutely nothing we get an Unavailable score.
    sqs = SQSScore.from_components([])
    assert sqs.name == "Synthetic Quality Score"
    assert sqs.score.raw_score is None
    assert sqs.score.score is None
    assert sqs.score.grade == Grade.UNAVAILABLE
    assert sqs.score.notes is None


def test_sqs_score_from_everything(
    column_correlation_stability,
    deep_structure_stability,
    column_distribution_stability,
    text_semantic_stability,
    text_structure_stability,
):
    # All 5 subscores present
    sqs = SQSScore.from_components(
        [
            column_correlation_stability,
            deep_structure_stability,
            column_distribution_stability,
            text_semantic_stability,
            text_structure_stability,
        ]
    )
    assert sqs.name == "Synthetic Quality Score"
    assert sqs.score.raw_score == 5.6486
    assert sqs.score.score == 5.6
    assert sqs.score.grade == Grade.MODERATE
    assert sqs.score.notes is None


def test_double_partial(column_distribution_stability, text_semantic_stability):
    # One tabular score and one text score, we still get an SQS
    sqs = SQSScore.from_components([column_distribution_stability, text_semantic_stability])
    assert sqs.name == "Synthetic Quality Score"
    assert sqs.score.raw_score == 9.5375
    # Break it down
    assert sqs.score.raw_score == (7 * 9.9 + 1 * 7.0) / (7 + 1)
    assert sqs.score.score == 9.5
    assert sqs.score.grade == Grade.EXCELLENT
    assert sqs.score.notes is None


# def test_privacy_score(aia, mia):
#     privacy_score = DataPrivacyScore.from_components([aia, mia])
#     assert privacy_score.name == "Data Privacy Score"
#     assert privacy_score.score.raw_score == 5.0
#     assert privacy_score.score.score == 5.0
#     assert privacy_score.score.grade == PrivacyGrade.GOOD
#     assert privacy_score.score.notes is None
