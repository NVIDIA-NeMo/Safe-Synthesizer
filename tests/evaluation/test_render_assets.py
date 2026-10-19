# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
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
    autocorrelation_similarity = _get_template("jinja/components/autocorrelation_similarity.j2")
    dataset_statistics = _get_template("jinja/components/dataset_statistics.j2")
    gauge = _get_template("jinja/components/score_gauge.j2")
    metric_card = _get_template("jinja/components/metric_card.j2")
    stylesheet = _get_template("css/multi_modal_report.css")
    training_columns = _get_template("jinja/components/training_columns.j2")
    text_semantic_similarity = _get_template("jinja/components/text_semantic_similarity.j2")
    text_structure_similarity = _get_template("jinja/components/text_structure_similarity.j2")
    javascript = _get_template("js/multi_modal_toggle.js")

    assert template is not None
    assert autocorrelation_similarity is not None
    assert dataset_statistics is not None
    assert gauge is not None
    assert metric_card is not None
    assert stylesheet is not None
    assert training_columns is not None
    assert text_semantic_similarity is not None
    assert text_structure_similarity is not None
    assert javascript is not None
    assert "score_ring(ctx.synthetic_quality_score.score)" in template
    assert 'data-report-view="time-series"' not in template
    assert template.index("Text Metrics") < template.index("Time-Series Metrics")
    assert template.index("Time-Series Metrics") < template.index('id="data-privacy"')
    assert "ctx.autocorrelation_similarity.figures" in autocorrelation_similarity
    assert "{{ row.html | safe }}" in autocorrelation_similarity
    assert 'class="autocorrelation-charts"' in autocorrelation_similarity
    assert ".autocorrelation-charts" in stylesheet
    assert "data-autocorrelation-column-select" in autocorrelation_similarity
    assert ".autocorrelation-column-select" in stylesheet
    assert autocorrelation_similarity.count('class="autocorrelation-chart-heading"') == 1
    assert ".autocorrelation-chart-heading" in stylesheet
    assert "selectAutocorrelationColumn" in javascript
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
                        "transform_functions": ["fake_name"],
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


def _autocorrelation_context(
    column_count: int, pair_score_column_count: int, constant_synthetic_profile_count: int = 0
) -> dict[str, object]:
    columns = [f"column_{index}" for index in range(column_count)]
    return {
        "autocorrelation_similarity": {
            "score": {"notes": None},
            "details": {},
            "evaluated_profile_count": column_count,
            "evaluated_group_count": 1,
            "evaluated_column_count": column_count,
            "constant_synthetic_profile_count": constant_synthetic_profile_count,
            "pair_score_column_count": pair_score_column_count,
            "summary_columns": columns,
            "summary_traces_per_column": 4,
            "figures": [
                {
                    "title": "Typical autocorrelation",
                    "html": "<div>summary</div>",
                    "column_select": True,
                    "tooltip_id": "autocorrelation-typical-info",
                    "tooltip": "Lines should overlap.",
                },
                {
                    "title": "Pair scores",
                    "html": "<div>pairs</div>",
                    "column_select": False,
                    "tooltip_id": "autocorrelation-pair-scores-info",
                    "tooltip": "Dots should be near 10.",
                },
            ],
        }
    }


def test_autocorrelation_renders_column_select_and_pair_score_limit_note() -> None:
    rendered = _render_template(
        "jinja/components/autocorrelation_similarity.j2",
        ctx=_autocorrelation_context(column_count=10, pair_score_column_count=8),
    )

    assert rendered.count("data-autocorrelation-column-select") == 1
    assert 'data-traces-per-column="4"' in rendered
    assert '<option value="9">column_9</option>' in rendered
    assert "Pair scores shows the 8 lowest-scoring columns." in rendered
    assert "<div>summary</div>" in rendered
    assert " ".join(rendered.split()).count("Summarizing 10 group-column pairs (1 group, 10 numeric columns).") == 1


def test_autocorrelation_renders_info_tooltip_for_each_chart() -> None:
    rendered = _render_template(
        "jinja/components/autocorrelation_similarity.j2",
        ctx=_autocorrelation_context(column_count=2, pair_score_column_count=2),
    )

    assert rendered.count("data-tooltip-toggle") == 2
    assert 'aria-controls="autocorrelation-typical-info"' in rendered
    assert 'id="autocorrelation-pair-scores-info" role="tooltip" hidden' in rendered
    assert "Dots should be near 10." in rendered


def test_autocorrelation_omits_column_select_and_note_for_single_column() -> None:
    rendered = _render_template(
        "jinja/components/autocorrelation_similarity.j2",
        ctx=_autocorrelation_context(column_count=1, pair_score_column_count=1),
    )

    assert "data-autocorrelation-column-select" not in rendered
    assert "lowest-scoring columns" not in rendered
    assert "constant synthetic values" not in rendered


@pytest.mark.parametrize(
    ("constant_count", "expected"),
    [
        (1, "1 pair has constant synthetic values."),
        (3, "3 pairs have constant synthetic values."),
    ],
)
def test_autocorrelation_notes_constant_synthetic_pairs_missing_from_profile_charts(
    constant_count: int, expected: str
) -> None:
    rendered = _render_template(
        "jinja/components/autocorrelation_similarity.j2",
        ctx=_autocorrelation_context(
            column_count=2, pair_score_column_count=2, constant_synthetic_profile_count=constant_count
        ),
    )

    assert expected in rendered


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


def test_rolling_stats_card_assets_are_wired_after_autocorrelation() -> None:
    template = _get_template("jinja/reports/multi_modal_report.j2")
    rolling_stats_similarity = _get_template("jinja/components/rolling_stats_similarity.j2")
    stylesheet = _get_template("css/multi_modal_report.css")
    javascript = _get_template("js/multi_modal_toggle.js")

    assert template is not None
    assert rolling_stats_similarity is not None
    assert stylesheet is not None
    assert javascript is not None
    assert template.index('"autocorrelation-similarity"') < template.index('"rolling-stats-similarity"')
    assert template.index('"rolling-stats-similarity"') < template.index('id="data-privacy"')
    assert "ctx.rolling_stats_similarity.figures" in rolling_stats_similarity
    assert 'class="rolling-stats-charts"' in rolling_stats_similarity
    assert rolling_stats_similarity.count('class="rolling-stats-chart-heading"') == 1
    assert ".rolling-stats-charts" in stylesheet
    assert ".rolling-stats-column-select" in stylesheet
    assert "[data-rolling-stats-column-select]" in javascript


def _rolling_stats_context(
    column_count: int, pair_score_column_count: int, constant_synthetic_profile_count: int = 0
) -> dict[str, object]:
    columns = [f"column_{index}" for index in range(column_count)]
    return {
        "rolling_stats_similarity": {
            "score": {"notes": None},
            "details": {},
            "evaluated_profile_count": column_count,
            "evaluated_group_count": 1,
            "evaluated_column_count": column_count,
            "constant_synthetic_profile_count": constant_synthetic_profile_count,
            "pair_score_column_count": pair_score_column_count,
            "summary_columns": columns,
            "summary_traces_per_column": 8,
            "figures": [
                {
                    "title": "Average and spread",
                    "html": "<div>profile</div>",
                    "column_select": True,
                    "tooltip_id": "rolling-stats-profile-info",
                    "tooltip": "Lines should overlap.",
                },
                {
                    "title": "Difference by segment",
                    "html": "<div>difference</div>",
                    "column_select": False,
                    "tooltip_id": "rolling-stats-difference-info",
                    "tooltip": "Bars should be short.",
                },
                {
                    "title": "Pair scores",
                    "html": "<div>pairs</div>",
                    "column_select": False,
                    "tooltip_id": "rolling-stats-pair-scores-info",
                    "tooltip": "Dots should be near 10.",
                },
            ],
        }
    }


def test_rolling_stats_renders_column_selects_and_pair_score_limit_note() -> None:
    rendered = _render_template(
        "jinja/components/rolling_stats_similarity.j2",
        ctx=_rolling_stats_context(column_count=10, pair_score_column_count=8),
    )

    assert rendered.count("data-rolling-stats-column-select") == 1
    assert 'data-traces-per-column="8"' in rendered
    assert '<option value="9">column_9</option>' in rendered
    assert "Pair scores shows the 8 lowest-scoring columns." in rendered
    assert " ".join(rendered.split()).count("Summarizing 10 group-column pairs (1 group, 10 numeric columns).") == 1


def test_rolling_stats_renders_info_tooltip_for_each_chart() -> None:
    rendered = _render_template(
        "jinja/components/rolling_stats_similarity.j2",
        ctx=_rolling_stats_context(column_count=2, pair_score_column_count=2),
    )

    assert rendered.count("data-tooltip-toggle") == 3
    assert 'aria-controls="rolling-stats-difference-info"' in rendered
    assert 'id="rolling-stats-pair-scores-info" role="tooltip" hidden' in rendered


def test_rolling_stats_omits_column_select_and_note_for_single_column() -> None:
    rendered = _render_template(
        "jinja/components/rolling_stats_similarity.j2",
        ctx=_rolling_stats_context(column_count=1, pair_score_column_count=1),
    )

    assert "data-rolling-stats-column-select" not in rendered
    assert "lowest-scoring columns" not in rendered
    assert "constant synthetic values" not in rendered


@pytest.mark.parametrize(
    ("constant_count", "expected"),
    [
        (1, "1 pair has constant synthetic values."),
        (3, "3 pairs have constant synthetic values."),
    ],
)
def test_rolling_stats_notes_constant_synthetic_pairs_missing_from_profile_charts(
    constant_count: int, expected: str
) -> None:
    rendered = _render_template(
        "jinja/components/rolling_stats_similarity.j2",
        ctx=_rolling_stats_context(
            column_count=2, pair_score_column_count=2, constant_synthetic_profile_count=constant_count
        ),
    )

    assert expected in rendered


def test_window_group_mia_card_assets_are_wired_after_row_level_membership_inference() -> None:
    template = _get_template("jinja/reports/multi_modal_report.j2")
    window_group_mia = _get_template("jinja/components/window_group_mia.j2")
    stylesheet = _get_template("css/multi_modal_report.css")
    javascript = _get_template("js/multi_modal_toggle.js")

    assert template is not None
    assert window_group_mia is not None
    assert stylesheet is not None
    assert javascript is not None
    assert template.index('id="data-privacy"') < template.index('"window-group-mia"')
    assert (
        template.index('metric_card("mia"') < template.index('"window-group-mia"') < template.index('metric_card("aia"')
    )
    assert 'class="window-group-mia-charts"' in window_group_mia
    assert ".window-group-mia-charts" in stylesheet
    assert ".window-group-mia-column-select" in stylesheet
    assert ".legend-window-group-mia-holdout" in stylesheet
    assert "[data-window-group-mia-column-select]" in javascript


def _window_group_mia_context(*, full: bool, columns: int = 2, sampled: bool = True) -> dict[str, object]:
    counts = {
        "training_windows": 1000,
        "training_windows_total": 1180 if sampled else 1000,
        "holdout_windows": 118 if full else 0,
        "holdout_windows_total": 118 if full else 0,
        "holdout_groups_with_windows": 1 if full else 0,
        "synthetic_windows": 5820,
        "synthetic_windows_total": 5820,
    }
    details: dict[str, object] = {
        "evaluation_mode": "full" if full else "copy_diagnostics_only",
        "counts": counts,
        "columns": [f"column_{index}" for index in range(columns)],
        "window_size": 10,
        "per_group": [{"group": "a"}, {"group": "b"}],
        "exact_copy_rate": 0.114,
    }
    if full:
        details["attack"] = {"training_flag_rate": 0.165, "holdout_flag_rate": 0.051}
    charts = [
        ("Distance to synthetic data", False, "window-group-mia-distance-info"),
        ("Groups flagged most", False, "window-group-mia-groups-info"),
        ("Closest match", True, "window-group-mia-closest-info"),
    ]
    figures = [
        {"title": title, "html": "<div>chart</div>", "column_select": select, "tooltip_id": tooltip_id, "tooltip": "?"}
        for title, select, tooltip_id in (charts if full else [charts[0], charts[2]])
    ]
    return {
        "window_membership_inference_protection": {
            "score": {"notes": None},
            "details": details,
            "closest_traces_per_column": 2,
            "figures": figures,
        }
    }


def test_window_group_mia_renders_summary_flag_rates_and_column_select() -> None:
    rendered = _render_template("jinja/components/window_group_mia.j2", ctx=_window_group_mia_context(full=True))
    text = " ".join(rendered.split())

    assert (
        "Compared 1,000 training windows from 2 groups and 118 holdout windows from 1 holdout group with "
        "5,820 synthetic windows (10 steps across 2 numeric columns)."
    ) in text
    assert "Windows were sampled reproducibly from 1,180 training windows." in text
    assert "16.5% of training windows and 5.1% of holdout windows are flagged as close to a synthetic window." in text
    assert "11.4% of training windows have an exact copy in the synthetic data." in text
    assert "Holdout Data" in rendered
    assert rendered.count("data-tooltip-toggle") == 3
    assert rendered.count("data-window-group-mia-column-select") == 1
    assert 'data-traces-per-column="2"' in rendered
    assert '<option value="1">column_1</option>' in rendered


def test_window_group_mia_copy_diagnostics_render_without_holdout_details() -> None:
    rendered = _render_template(
        "jinja/components/window_group_mia.j2",
        ctx=_window_group_mia_context(full=False, columns=1, sampled=False),
    )
    text = " ".join(rendered.split())

    assert "holdout" not in text.lower()
    assert "flagged" not in text
    assert "sampled reproducibly" not in text
    assert "(10 steps across 1 numeric column)." in text
    assert "11.4% of training windows have an exact copy" in text
    assert "data-window-group-mia-column-select" not in rendered
