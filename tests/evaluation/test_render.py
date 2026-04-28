# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

# Skip all tests in this module if sentence_transformers is not available
pytest.importorskip(
    "sentence_transformers",
    reason="sentence_transformers is required for these tests (install with: uv sync --extra cpu)",
)

from nemo_safe_synthesizer.evaluation.render import render_report
from nemo_safe_synthesizer.evaluation.reports.multimodal.multimodal_report import MultimodalReport


@pytest.mark.slow
def test_render(
    fixture_training_df_10k,
    fixture_synthetic_df_10k,
    fixture_test_df,
    fixture_skip_privacy_metrics_config,
    fixture_column_statistics,
):
    report = MultimodalReport.from_dataframes(
        training=fixture_training_df_10k,
        synthetic=fixture_synthetic_df_10k,
        test=fixture_test_df,
        config=fixture_skip_privacy_metrics_config,
        column_statistics=fixture_column_statistics,
    )
    output = render_report(report, "multi_modal_report.j2")
    assert output is not None
    # output = render_report(report, "multi_modal_report.j2", "/tmp/test_mm_report.html")
    assert len(output) > 0

    # Section headings rendered (catch wholesale template breakage)
    assert "Dataset Statistics" in output
    assert "Synthetic Quality Score" in output
    assert "Training Data Columns" in output

    # Dynamic values from Pydantic models made it into HTML (catch silent blanks
    # from Jinja variable typos -- default Undefined renders as empty string)
    assert "10000" in output
    assert "Missing %" in output


@pytest.mark.slow
def test_render_dp_enabled(
    fixture_training_df_5k,
    fixture_synthetic_df_5k,
    fixture_test_df,
    fixture_dp_enabled_config,
    fixture_column_statistics,
):
    report = MultimodalReport.from_dataframes(
        training=fixture_training_df_5k,
        synthetic=fixture_synthetic_df_5k,
        test=fixture_test_df,
        config=fixture_dp_enabled_config,
        column_statistics=fixture_column_statistics,
    )
    output = render_report(report, "multi_modal_report.j2")
    # output = render_report(report, "multi_modal_report.j2", "/tmp/test_mm_report_dp_enabled.html")
    assert output is not None
    assert len(output) > 0

    assert "Dataset Statistics" in output
    assert "Synthetic Quality Score" in output
    assert "Data Privacy Score" in output
    assert "5000" in output


@pytest.mark.slow
def test_render_dp_not_enabled(
    fixture_training_df_5k,
    fixture_synthetic_df_5k,
    fixture_test_df,
    fixture_dp_not_enabled_config,
    fixture_column_statistics,
):
    report = MultimodalReport.from_dataframes(
        training=fixture_training_df_5k,
        synthetic=fixture_synthetic_df_5k,
        test=fixture_test_df,
        config=fixture_dp_not_enabled_config,
        column_statistics=fixture_column_statistics,
    )
    output = render_report(report, "multi_modal_report.j2")
    # output = render_report(report, "multi_modal_report.j2", "/tmp/test_mm_report_dp_not_enabled.html")
    assert output is not None
    assert len(output) > 0

    assert "Dataset Statistics" in output
    assert "Synthetic Quality Score" in output
    assert "Data Privacy Score" in output
    assert "5000" in output
