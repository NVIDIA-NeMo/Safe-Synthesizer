# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

# Skip all tests in this module if sentence_transformers is not available
pytest.importorskip(
    "sentence_transformers",
    reason="sentence_transformers is required for these tests (install with: uv sync --extra cpu)",
)

from nemo_safe_synthesizer.config.evaluate import EvaluationParameters
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import Grade
from nemo_safe_synthesizer.evaluation.reports.multimodal.multimodal_report import MultimodalReport


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
    fixture_training_df_5k, fixture_synthetic_df_5k, fixture_test_df, fixture_skip_privacy_metrics_config
):
    report = MultimodalReport.from_dataframes(
        training=fixture_training_df_5k,
        synthetic=fixture_synthetic_df_5k,
        test=fixture_test_df,
        config=fixture_skip_privacy_metrics_config,
    )

    assert len(report.components) == 11
    assert report.components[-1].name == "Synthetic Quality Score"
    assert report.components[-1].score.grade == Grade.EXCELLENT

    report_dict = report.get_dict()
    assert len(report_dict) == 6
    assert report_dict["Synthetic Quality Score"] == {
        "raw_score": 9.7935,
        "grade": "Excellent",
        "score": 9.8,
        "notes": None,
    }

    report_json = report.get_json()
    assert (
        '"Synthetic Quality Score": {"raw_score": 9.7935, "grade": "Excellent", "score": 9.8, "notes": null}}'
        in report_json
    )
