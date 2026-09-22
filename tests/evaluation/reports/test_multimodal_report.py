# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pandas as pd
import pytest

from nemo_safe_synthesizer.config.evaluate import EvaluationParameters, TimeSeriesEvaluationParameters
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters
from nemo_safe_synthesizer.evaluation.components.autocorrelation_similarity import (
    AutocorrelationProfile,
    AutocorrelationSimilarity,
)
from nemo_safe_synthesizer.evaluation.components.text_semantic_similarity import TextSemanticSimilarity
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import EvaluationScore, Grade
from nemo_safe_synthesizer.evaluation.render import render_report
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


def _time_series_config(
    *,
    report_rows: int = 5000,
    evaluation_enabled: bool = True,
) -> SafeSynthesizerParameters:
    return SafeSynthesizerParameters(
        time_series=TimeSeriesParameters(is_timeseries=True, timestamp_column="time"),
        evaluation=EvaluationParameters(
            enabled=evaluation_enabled,
            mia_enabled=False,
            aia_enabled=False,
            pii_replay_enabled=False,
            sqs_report_rows=report_rows,
            time_series=TimeSeriesEvaluationParameters(enabled=True),
        ),
    )


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


def test_time_series_metric_is_absent_when_feature_gate_is_disabled() -> None:
    frame = pd.DataFrame({"value": [0.0, 1.0, 3.0, 2.0, 4.0, 1.0]})

    report = MultimodalReport.from_dataframes(frame, frame.copy(), config=SafeSynthesizerParameters())

    assert not any(isinstance(component, AutocorrelationSimilarity) for component in report.components)
    assert report.jinja_context["with_time_series"] is False


def test_global_evaluation_gate_overrides_time_series_evaluation_gate() -> None:
    frame = pd.DataFrame({"time": range(6), "value": [0.0, 1.0, 3.0, 2.0, 4.0, 1.0]})

    report = MultimodalReport.from_dataframes(
        frame,
        frame.copy(),
        config=_time_series_config(evaluation_enabled=False),
    )

    assert not any(isinstance(component, AutocorrelationSimilarity) for component in report.components)


def test_time_series_metric_uses_unsampled_ordered_data_and_renders_profiles() -> None:
    row_count = 5001
    time = np.arange(row_count)
    training = pd.DataFrame({"time": time, "value": np.sin(time / 11)})
    synthetic = training.sample(frac=1.0, random_state=17).reset_index(drop=True)

    report = MultimodalReport.from_dataframes(
        training,
        synthetic,
        config=_time_series_config(report_rows=100),
    )

    component = next(item for item in report.components if isinstance(item, AutocorrelationSimilarity))
    assert component.score.score == 10
    assert report.evaluation_datasets.training_rows == 100
    assert report.jinja_context["autocorrelation_similarity"]["evaluated_profile_count"] == 1
    output = render_report(report)
    assert output is not None
    assert "Time-Series Evaluation" in output
    assert "Training ACF" in output
    assert "Synthetic ACF" in output


def test_enabled_unavailable_time_series_metric_renders_actionable_reason() -> None:
    frame = pd.DataFrame({"time": range(6), "value": [1.0] * 6})

    report = MultimodalReport.from_dataframes(frame, frame.copy(), config=_time_series_config())

    component = next(item for item in report.components if isinstance(item, AutocorrelationSimilarity))
    assert component.score.score is None
    assert component.score.notes == "No usable group/column autocorrelation profiles."
    output = render_report(report)
    assert output is not None
    assert "No usable group/column autocorrelation profiles." in output


def test_time_series_report_limits_charts_to_lowest_scoring_profiles() -> None:
    report = _minimal_multimodal_report()
    profiles: list[AutocorrelationProfile] = [
        {
            "group": str(index),
            "column": "value",
            "lags": [1],
            "effective_max_lag": 1,
            "evaluated_lags": 1,
            "error": 0.0,
            "training_acf": [0.5],
            "synthetic_acf": [0.5],
            "similarity": index / 13,
            "training_pair_support": [10],
            "synthetic_pair_support": [10],
        }
        for index in range(13)
    ]
    report.components = [
        AutocorrelationSimilarity(
            score=EvaluationScore.finalize_grade(raw_score=0.5, score=5.0),
            details={"profiles": profiles},
        )
    ]

    context = report.jinja_context["autocorrelation_similarity"]

    assert context["evaluated_profile_count"] == 13
    assert context["displayed_profile_count"] == 12
    assert len(context["figures"]) == 12
    assert all("group 12" not in figure["title"] for figure in context["figures"])
