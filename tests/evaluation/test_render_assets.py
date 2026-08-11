# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from jinja2 import Environment, FunctionLoader, select_autoescape

from nemo_safe_synthesizer.evaluation.render import _get_template


def _render_template(name: str, **context: object) -> str:
    env = Environment(
        loader=FunctionLoader(_get_template),
        autoescape=select_autoescape(["html", "xml"]),
    )
    return env.get_template(name).render(**context)


def test_evaluation_report_uses_versioned_plotly_cdn() -> None:
    template = _get_template("jinja/reports/multi_modal_report.j2")

    assert template is not None
    assert "fonts.googleapis.com" not in template
    assert 'src="https://cdn.plot.ly/plotly-3.3.1.min.js"' in template


def test_evaluation_report_themes_charts_in_report_assets() -> None:
    template = _get_template("jinja/reports/multi_modal_report.j2")
    dataset_statistics = _get_template("jinja/components/dataset_statistics.j2")
    gauge = _get_template("jinja/components/score_gauge.j2")
    metric_card = _get_template("jinja/components/metric_card.j2")
    stylesheet = _get_template("css/multi_modal_report.css")
    training_columns = _get_template("jinja/components/training_columns.j2")
    text_semantic_similarity = _get_template("jinja/components/text_semantic_similarity.j2")
    text_structure_similarity = _get_template("jinja/components/text_structure_similarity.j2")
    javascript = _get_template("js/multi_modal_toggle.js")

    assert template is not None
    assert dataset_statistics is not None
    assert gauge is not None
    assert metric_card is not None
    assert stylesheet is not None
    assert training_columns is not None
    assert text_semantic_similarity is not None
    assert text_structure_similarity is not None
    assert javascript is not None
    assert "score_ring(ctx.synthetic_quality_score.score)" in template
    assert "score-ring-canvas" in gauge
    assert "brand-assets.cne.ngc.nvidia.com/assets/fonts/nvidia-sans" in stylesheet
    assert "data-metric-toggle" in metric_card
    assert "initializeGradientScoreRing" in javascript
    assert "initializeScoreLabels" in javascript
    assert "themePlotlyCharts" in javascript
    assert "themeMembershipPlot" in javascript
    assert "themeDeepStructurePlot" in javascript
    assert "[TRAINING_COLOR, SYNTHETIC_COLOR]" in javascript
    assert "rebuildDistributionCharts" in javascript
    assert "--distribution-chart-height: 120px" in stylesheet
    assert "grid-auto-rows: var(--distribution-chart-height)" in stylesheet
    assert "height: var(--distribution-chart-height)" in stylesheet
    assert "decodePlotlyArray" in javascript
    assert "makePlotResponsive" in javascript
    assert "delete layout.width" in javascript
    assert "_fullData" not in javascript
    assert "Plotly.relayout" in javascript
    assert "toggleColumns" in javascript
    assert template.count("#low-sqs-scores") == 1
    assert "memorization-status--warning" in dataset_statistics
    assert 'class="metric-card-toggle" type="button"' in metric_card
    assert 'role="button"' not in metric_card
    assert "{{ tooltip | trim }}" in metric_card
    assert "white-space: normal" in stylesheet
    assert "white-space: pre-line" not in stylesheet
    assert "column.training_field_features.type" in training_columns
    assert "column.synthetic_field_features.type" not in training_columns
    assert "columns-table--with-transform" in training_columns
    assert 'class="entity-tags"' in training_columns
    assert "Entities (Count)" in training_columns
    assert '{{ entity }} ({{ "{:,}".format(count) }})' in training_columns
    assert ".columns-table--with-transform" in stylesheet
    assert ".transform-function-cell" in stylesheet
    assert 'class="chart-subheading"' in text_semantic_similarity
    assert "{{ row.title }}" in text_semantic_similarity
    assert "{{ row.html }}" in text_semantic_similarity
    assert 'class="chart-subheading"' in text_structure_similarity
    assert "{{ row.title }}" in text_structure_similarity
    assert "{{ row.html }}" in text_structure_similarity
    assert "#dps-interpretation" in template
    assert 'aria-label="Report sections"' in template
    assert '<div class="sidebar-label">' not in template
    assert template.count("data-score-label") == 2
    assert "grid-template-columns: minmax(0, 1fr)" in stylesheet
    assert ".show-columns::before" in stylesheet
    assert "inset: calc(var(--header-height) + 56px) 0 0 0" in stylesheet


def test_training_columns_render_distribution_links_grades_and_entity_counts() -> None:
    ctx = {
        "with_synthesizer": True,
        "with_transform": True,
        "column_distribution_stability": {
            "evaluation_fields": [
                {
                    "name": "Review Text",
                    "training_field_features": {
                        "unique_count": 1_234,
                        "missing_count": 0,
                        "avg_str_length": 42.25,
                        "type": "text",
                    },
                    "distribution_stability": {"grade": "Very Good"},
                    "column_statistics": {
                        "detected_entity_counts": {"PERSON": 12},
                        "is_transformed": True,
                        "transform_methods": ["fake_name"],
                    },
                }
            ]
        },
    }

    rendered = _render_template("jinja/components/training_columns.j2", ctx=ctx)

    assert "<th>Distribution</th>" in rendered
    assert '<a class="column-name-link" href="#Review%20Text">Review Text</a>' in rendered
    assert '<span class="score-label">Very Good</span>' in rendered
    assert "<th>Entities (Count)</th>" in rendered
    assert "PERSON (12)" in rendered
    assert "fake_name" in rendered


def test_score_guidance_renders_recommendations_for_the_current_grade() -> None:
    quality = _render_template("jinja/components/score_guidance.j2", kind="quality", grade="Good")
    privacy = _render_template("jinja/components/score_guidance.j2", kind="privacy", grade="Excellent")

    assert quality.count("Not recommended") == 2
    assert quality.count("Suitable") == 2
    assert privacy.count("Not recommended") == 0
    assert privacy.count("Suitable") == 4


def test_distribution_chart_deep_links_survive_chart_rebuild() -> None:
    javascript = _get_template("js/multi_modal_toggle.js")

    assert javascript is not None
    assert 'container.querySelectorAll(":scope > span[id]")' in javascript
    assert "wrapper.id = anchorIds[chartIndex]" in javascript
    assert "const currentTarget = document.getElementById(targetId)" in javascript
    assert 'currentTarget.scrollIntoView({block: "start"})' in javascript


def test_evaluation_report_uses_reference_icon_geometry() -> None:
    icons = _get_template("jinja/components/icon.j2")

    assert icons is not None
    assert 'viewBox="0 0 16 16"' in icons
    assert "stroke-width" not in icons
    assert 'd="M8 5H7V3.707' in icons  # ChartPerformance
    assert 'd="M13 5.293V14H3V2h6.707z' in icons  # Document
    assert 'd="m11.871 1.295.509 2.325' in icons  # CheckmarkBadge
    assert 'd="M5 5a3 3 0 0 1 6 0v2h1v7H4V7h1z' in icons  # LockClosed
