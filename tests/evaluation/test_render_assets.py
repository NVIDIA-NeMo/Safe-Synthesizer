# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_safe_synthesizer.evaluation.render import _get_template


def test_evaluation_report_uses_versioned_plotly_cdn() -> None:
    template = _get_template("jinja/reports/multi_modal_report.j2")

    assert template is not None
    assert "fonts.googleapis.com" not in template
    assert 'src="https://cdn.plot.ly/plotly-3.3.1.min.js"' in template


def test_evaluation_report_themes_charts_in_report_assets() -> None:
    template = _get_template("jinja/reports/multi_modal_report.j2")
    gauge = _get_template("jinja/components/score_gauge.j2")
    metric_card = _get_template("jinja/components/metric_card.j2")
    stylesheet = _get_template("css/multi_modal_report.css")
    javascript = _get_template("js/multi_modal_toggle.js")

    assert template is not None
    assert gauge is not None
    assert metric_card is not None
    assert stylesheet is not None
    assert javascript is not None
    assert "score_ring(ctx.synthetic_quality_score.score)" in template
    assert "score-ring-canvas" in gauge
    assert "brand-assets.cne.ngc.nvidia.com/assets/fonts/nvidia-sans" in stylesheet
    assert "data-metric-toggle" in metric_card
    assert "initializeGradientScoreRing" in javascript
    assert "themePlotlyCharts" in javascript
    assert "themeMembershipPlot" in javascript
    assert "rebuildDistributionCharts" in javascript
    assert "Plotly.relayout" in javascript
    assert "toggleColumns" in javascript


def test_evaluation_report_uses_reference_icon_geometry() -> None:
    icons = _get_template("jinja/components/icon.j2")

    assert icons is not None
    assert 'viewBox="0 0 16 16"' in icons
    assert "stroke-width" not in icons
    assert 'd="M8 5H7V3.707' in icons  # ChartPerformance
    assert 'd="M13 5.293V14H3V2h6.707z' in icons  # Document
    assert 'd="m11.871 1.295.509 2.325' in icons  # CheckmarkBadge
    assert 'd="M5 5a3 3 0 0 1 6 0v2h1v7H4V7h1z' in icons  # LockClosed
