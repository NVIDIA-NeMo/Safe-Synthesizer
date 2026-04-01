# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

# Skip all tests in this module if sentence_transformers is not available
pytest.importorskip(
    "sentence_transformers",
    reason="sentence_transformers is required for these tests (install with: uv sync --extra cpu)",
)

from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import Grade
from nemo_safe_synthesizer.evaluation.reports.multimodal.multimodal_report import MultimodalReport


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
