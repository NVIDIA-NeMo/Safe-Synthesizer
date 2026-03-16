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
def test_render(train_df_10k, synth_df_10k, test_df, skip_privacy_metrics_config, column_statistics):
    report = MultimodalReport.from_dataframes(
        reference=train_df_10k,
        output=synth_df_10k,
        test=test_df,
        config=skip_privacy_metrics_config,
        column_statistics=column_statistics,
    )
    output = render_report(report, "multi_modal_report.j2")
    assert output is not None
    # output = render_report(report, "multi_modal_report.j2", "/tmp/test_mm_report.html")
    assert len(output) > 0


@pytest.mark.slow
def test_render_dp_enabled(train_df_5k, synth_df_5k, test_df, dp_enabled_config, column_statistics):
    report = MultimodalReport.from_dataframes(
        reference=train_df_5k,
        output=synth_df_5k,
        test=test_df,
        config=dp_enabled_config,
        column_statistics=column_statistics,
    )
    output = render_report(report, "multi_modal_report.j2")
    # output = render_report(report, "multi_modal_report.j2", "/tmp/test_mm_report_dp_enabled.html")
    assert output is not None
    assert len(output) > 0


@pytest.mark.slow
def test_render_dp_not_enabled(train_df_5k, synth_df_5k, test_df, dp_not_enabled_config, column_statistics):
    report = MultimodalReport.from_dataframes(
        reference=train_df_5k,
        output=synth_df_5k,
        test=test_df,
        config=dp_not_enabled_config,
        column_statistics=column_statistics,
    )
    output = render_report(report, "multi_modal_report.j2")
    # output = render_report(report, "multi_modal_report.j2", "/tmp/test_mm_report_dp_not_enabled.html")
    assert output is not None
    assert len(output) > 0
