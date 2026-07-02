# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.evaluate import EvaluationParameters
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.evaluation.components.text_semantic_similarity import TextSemanticSimilarity
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import EvaluationScore, Grade
from nemo_safe_synthesizer.evaluation.reports.multimodal import multimodal_report as multimodal_report_module
from nemo_safe_synthesizer.evaluation.reports.multimodal.multimodal_report import MultimodalReport


@pytest.fixture(autouse=True)
def stub_text_semantic_similarity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep report assembly tests independent of sentence-transformer inference."""

    class StubTextSemanticSimilarity:
        @staticmethod
        def from_evaluation_datasets(*_args, **_kwargs) -> TextSemanticSimilarity:
            return TextSemanticSimilarity(score=EvaluationScore.finalize_grade(9.0, 9.0))

    monkeypatch.setattr(
        multimodal_report_module,
        "TextSemanticSimilarity",
        StubTextSemanticSimilarity,
    )


def _minimal_multimodal_report() -> MultimodalReport:
    training_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    synthetic_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    datasets = EvaluationDatasets(training=training_df, synthetic=synthetic_df)
    return MultimodalReport(evaluation_datasets=datasets, components=[])


def test_jinja_context_job_id_none_when_nemo_job_id_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEMO_JOB_ID", raising=False)
    report = _minimal_multimodal_report()
    ctx = report.jinja_context
    assert ctx["job_id"] is None


def test_jinja_context_job_id_set_when_nemo_job_id_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_JOB_ID", "cluster-job-abc123")
    report = _minimal_multimodal_report()
    ctx = report.jinja_context
    assert ctx["job_id"] == "cluster-job-abc123"


def test_from_dataframes_applies_sqs_report_config(fixture_training_df, fixture_synthetic_df, fixture_test_df) -> None:
    """``sqs_report_rows`` / ``sqs_report_columns`` from config drive the actual subsampling.

    Regression: previously the multimodal report looked up ``sqs_rows`` /
    ``sqs_columns`` -- keys that do not exist on ``EvaluationParameters`` --
    so user-supplied row/column limits were silently ignored.
    """
    target_rows = 37
    target_cols = 3
    config = SafeSynthesizerParameters(
        evaluation=EvaluationParameters(
            mia_enabled=False,
            aia_enabled=False,
            pii_replay_enabled=False,
            sqs_report_rows=target_rows,
            sqs_report_columns=target_cols,
        ),
    )

    report = MultimodalReport.from_dataframes(
        training=fixture_training_df,
        synthetic=fixture_synthetic_df,
        test=fixture_test_df,
        config=config,
    )

    assert report.evaluation_datasets is not None
    assert report.evaluation_datasets.training_rows == target_rows
    assert report.evaluation_datasets.synthetic_rows == target_rows
    assert report.evaluation_datasets.training_cols == target_cols
    assert report.evaluation_datasets.synthetic_cols == target_cols


def test_multimodal_report(
    fixture_training_df, fixture_synthetic_df, fixture_test_df, fixture_skip_privacy_metrics_config
):
    report = MultimodalReport.from_dataframes(
        training=fixture_training_df,
        synthetic=fixture_synthetic_df,
        test=fixture_test_df,
        config=fixture_skip_privacy_metrics_config,
    )

    assert len(report.components) == 11
    assert report.components[-1].name == "Synthetic Quality Score"
    assert report.components[-1].score.grade == Grade.EXCELLENT

    report_dict = report.get_dict()
    assert len(report_dict) == 6
    assert report_dict["Text Semantic Similarity"] == {
        "raw_score": 9.0,
        "grade": "Excellent",
        "score": 9.0,
        "notes": None,
    }
    assert report_dict["Synthetic Quality Score"]["grade"] == "Excellent"
    assert report_dict["Synthetic Quality Score"]["score"] > 0

    report_json = json.loads(report.get_json())
    assert report_json["Text Semantic Similarity"] == report_dict["Text Semantic Similarity"]
    assert report_json["Synthetic Quality Score"] == report_dict["Synthetic Quality Score"]
