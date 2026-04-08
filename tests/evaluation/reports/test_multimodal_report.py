# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

# Skip all tests in this module if sentence_transformers is not available
pytest.importorskip(
    "sentence_transformers",
    reason="sentence_transformers is required for these tests (install with: uv sync --extra cpu)",
)

from nemo_safe_synthesizer.evaluation.data_model.evaluation_dataset import EvaluationDataset
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import Grade
from nemo_safe_synthesizer.evaluation.reports.multimodal.multimodal_report import MultimodalReport


def _minimal_multimodal_report() -> MultimodalReport:
    reference = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    output = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    dataset = EvaluationDataset(reference=reference, output=output)
    return MultimodalReport(evaluation_dataset=dataset, components=[])


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


@pytest.mark.skip(reason="Times out")
def test_multimodal_report(training_df_5k, synthetic_df_5k, test_df, skip_privacy_metrics_config):
    report = MultimodalReport.from_dataframes(
        training=training_df_5k, synthetic=synthetic_df_5k, test=test_df, config=skip_privacy_metrics_config
    )

    assert len(report.components) == 11
    assert report.components[-1].name == "Synthetic Quality Score"
    assert report.components[-1].score.grade == Grade.EXCELLENT

    report_dict = report.get_dict()
    assert len(report_dict) == 6
    assert report_dict["Synthetic Quality Score"] == {
        "raw_score": 9.8215,
        "grade": "Excellent",
        "score": 9.8,
        "notes": None,
    }

    report_json = report.get_json()
    assert (
        '"Synthetic Quality Score": {"raw_score": 9.8215, "grade": "Excellent", "score": 9.8, "notes": null}}'
        in report_json
    )
