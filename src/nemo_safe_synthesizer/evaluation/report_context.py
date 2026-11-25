# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the HTML report's template context from ``EvaluationReportData``.

The HTML report shows only what the data holds; this module adds display names, tooltips, and charts drawn
from the data's values.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..config.evaluation_report_data import (
    AttributeInferenceMetric,
    ColumnResult,
    CorrelationMetric,
    DeepStructureMetric,
    EvaluationReportData,
    MembershipInferenceMetric,
    Metric,
    MetricScore,
    PIIReplacement,
    PIIReplayMetric,
    Points,
    Projection,
    TextSemanticMetric,
    TextStructureMetric,
)
from .assets.text.multi_modal_tooltips import tooltips
from .components import multi_modal_figures as figures
from .components.attribute_inference_protection import AttributeInferenceProtection
from .components.column_distribution import ColumnDistribution
from .components.component import Component
from .components.correlation import Correlation
from .components.data_privacy_score import DataPrivacyScore
from .components.deep_structure import DeepStructure
from .components.membership_inference_protection import MembershipInferenceProtection
from .components.sqs_score import SQSScore
from .components.text_semantic_similarity import TextSemanticSimilarity
from .components.text_structure_similarity import TextStructureSimilarity
from .reports.multimodal.multimodal_report import TIME_SERIES_COMPONENTS

MetricT = TypeVar("MetricT", bound=Metric)


def report_context(data: EvaluationReportData) -> dict[str, Any]:
    """Return the HTML report template's context for ``data``."""
    metrics = data.metrics
    time_series_cards = [
        component.report_context(metric)
        for component in TIME_SERIES_COMPONENTS
        if (metric := getattr(metrics, component.metric_key)) is not None
    ]
    replay = metrics.pii_replay
    return {
        "tooltips": tooltips,
        "job_id": data.job_id,
        "with_synthesizer": _available(data.scores.synthetic_quality),
        "with_data_privacy": _available(data.scores.data_privacy),
        "with_transform": data.any_column_transformed,
        "dp_enabled": data.differential_privacy.enabled,
        "epsilon": data.differential_privacy.epsilon,
        "delta": data.differential_privacy.delta,
        "synthetic_quality_score": _score_context(SQSScore, data.scores.synthetic_quality),
        "data_privacy_score": _score_context(DataPrivacyScore, data.scores.data_privacy),
        "dataset_statistics": data.dataset,
        "column_correlation_stability": _metric_context(
            Correlation, metrics.column_correlation_stability, figure=_correlation_figure
        ),
        "deep_structure_stability": _metric_context(
            DeepStructure, metrics.deep_structure_stability, figure=_deep_structure_figure
        ),
        "column_distribution_stability": {
            **_metric_context(ColumnDistribution, metrics.column_distribution_stability),
            "figures": _column_distribution_rows(data.columns),
            "columns": data.columns,
        },
        "text_semantic_similarity": {
            **_metric_context(TextSemanticSimilarity, metrics.text_semantic_similarity),
            "figures": _text_semantic_figures(metrics.text_semantic_similarity),
        },
        "text_structure_similarity": {
            **_metric_context(TextStructureSimilarity, metrics.text_structure_similarity),
            "figures": _text_structure_figures(metrics.text_structure_similarity),
        },
        "membership_inference_protection": {
            **_metric_context(
                MembershipInferenceProtection, metrics.membership_inference_protection, figure=_mia_figure
            ),
            "excluded_columns": (
                metrics.membership_inference_protection.excluded_columns
                if metrics.membership_inference_protection is not None
                else []
            ),
        },
        "attribute_inference_protection": _metric_context(
            AttributeInferenceProtection, metrics.attribute_inference_protection, figure=_aia_figure
        ),
        "pii_replacement": _pii_context(data.pii_replacement, replay),
        "time_series_cards": [card for card in time_series_cards if card["report_section"] == "time_series"],
        "data_privacy_time_series_cards": [
            card for card in time_series_cards if card["report_section"] == "data_privacy"
        ],
        "with_time_series": any(card["report_section"] == "time_series" for card in time_series_cards),
    }


def _available(score: MetricScore | None) -> bool:
    return score is not None and score.grade != "Unavailable"


def _display_name(component: type[Component]) -> str:
    return component.model_fields["name"].default


def _unavailable() -> MetricScore:
    return MetricScore(raw_score=None, score=None, grade="Unavailable", notes=None)


def _score_context(component: type[Component], score: MetricScore | None) -> dict[str, Any]:
    return {"name": _display_name(component), "score": score or _unavailable()}


def _metric_context(
    component: type[Component],
    metric: MetricT | None,
    figure: Callable[[MetricT], go.Figure | None] | None = None,
) -> dict[str, Any]:
    chart = figure(metric) if figure is not None and metric is not None else None
    return {
        "name": _display_name(component),
        "score": metric.score if metric is not None else _unavailable(),
        "figure": _html(chart) if chart is not None else None,
    }


def _html(figure: go.Figure) -> str:
    return figure.to_html(full_html=False, include_plotlyjs=False)


def _matrix(columns: list[str], values: list[list[float | None]]) -> pd.DataFrame:
    return pd.DataFrame(np.array(values, dtype=float), index=columns, columns=columns)


def _correlation_figure(metric: CorrelationMetric) -> go.Figure | None:
    matrices = metric.correlations
    if matrices is None:
        return None
    return figures.generate_combined_correlation_figure(
        training_correlation_df=_matrix(matrices.columns, matrices.training),
        synthetic_correlation_df=_matrix(matrices.columns, matrices.synthetic),
        correlation_difference=_matrix(matrices.columns, matrices.difference),
    )


def _narrowest_floats(values: list[float | None]) -> np.ndarray:
    # Plotly embeds float64 at twice the size, and float32 embeddings come back from the data as float64.
    wide = np.asarray(values, dtype=np.float64)
    narrow = wide.astype(np.float32)
    return narrow if np.array_equal(narrow, wide, equal_nan=True) else wide


def _points(projection: Projection, which: str) -> pd.DataFrame:
    points: Points = getattr(projection, which)
    return pd.DataFrame(
        {projection.axes[0]: _narrowest_floats(points.x), projection.axes[1]: _narrowest_floats(points.y)}
    )


def _deep_structure_figure(metric: DeepStructureMetric) -> go.Figure | None:
    if metric.projection is None:
        return None
    return figures.structure_stability_figure(
        training_df=_points(metric.projection, "training"), synthetic_df=_points(metric.projection, "synthetic")
    )


def _text_semantic_figures(metric: TextSemanticMetric | None) -> list[dict[str, str]]:
    if metric is None:
        return []
    rows = []
    for column, projection in metric.projections.items():
        figure = figures.generate_text_semantic_similarity_figures(
            _points(projection, "training"), _points(projection, "synthetic")
        )
        if figure is not None:
            rows.append({"title": column, "html": _html(figure)})
    return rows


def _text_structure_figures(metric: TextStructureMetric | None) -> list[dict[str, str]]:
    if metric is None:
        return []
    return [
        {"title": column, "html": _html(figures.generate_text_structure_similarity_figures(distributions))}
        for column, distributions in metric.columns.items()
    ]


def _mia_figure(metric: MembershipInferenceMetric) -> go.Figure | None:
    if not metric.outcomes:
        return None
    frame = pd.DataFrame(
        {
            "Protection": [outcome.grade for outcome in metric.outcomes],
            "Attack Percentage": [outcome.percentage for outcome in metric.outcomes],
        }
    )
    if all(outcome.count is not None for outcome in metric.outcomes):
        frame.insert(1, "Attack Count", [outcome.count for outcome in metric.outcomes])
    return figures.generate_mia_figure(df=frame)


def _aia_figure(metric: AttributeInferenceMetric) -> go.Figure | None:
    if not metric.columns:
        return None
    return figures.generate_aia_figure(
        pd.DataFrame(
            {
                "Column": [column.column for column in metric.columns],
                "Risk": [column.protection for column in metric.columns],
                "Protection": [column.grade for column in metric.columns],
            }
        )
    )


def _column_figure(column: ColumnResult) -> go.Figure | None:
    if column.distribution is not None:
        return figures.distribution_figure(column.distribution)
    if column.training.type == "numeric":
        return figures.binned_distribution_figure(None)
    return None


def _column_distribution_rows(columns: list[ColumnResult]) -> list[dict[str, Any]]:
    """Pair the per-column charts into side-by-side rows, in column order."""
    charted = [(column.name, figure) for column in columns if (figure := _column_figure(column)) is not None]
    rows = []
    for i in range(0, len(charted), 2):
        pair = charted[i : i + 2]
        names = [name for name, _ in pair]
        figure = figures.combine_subplots(
            figures=[figure for _, figure in pair],
            titles=names,
            shared_xaxes=False,
            shared_yaxes=False,
            height=500,
            margin={"t": 36, "b": 0, "l": 0, "r": 0},
        )
        figure.update_xaxes(showticklabels=False, visible=False)
        rows.append({"name1": names[0], "name2": names[1] if len(names) > 1 else None, "figure": _html(figure)})
    return rows


def _pii_context(pii: PIIReplacement | None, replay: PIIReplayMetric | None) -> dict[str, Any] | None:
    if pii is None:
        return None
    replay_data = [entry.model_dump() for entry in replay.entries] if replay is not None else []
    rows = [
        {
            **column.model_dump(exclude={"pattern_symbols"}),
            "pattern_symbols": [(symbol.symbol, symbol.meaning) for symbol in column.pattern_symbols],
        }
        for column in pii.columns
    ]
    groups = _pii_column_groups(rows, replay_data)
    replay_note = _pii_replay_note(replay)
    return {
        "replacement_off": not pii.replaced,
        "rows": rows,
        "kept_columns": pii.kept_columns,
        "sampler": pii.sampler,
        "locale": pii.locale,
        "seed": pii.seed,
        "mappings": pii.value_mappings,
        "groups": groups,
        "replay_note": replay_note,
        "headline": _pii_headline(rows, groups, replay_note),
    }


def _pii_column_groups(rows: list[dict], replay_data: list[dict]) -> list[dict]:
    """Join each replaced column with its PII replay results, with one entry per PII type found in it.

    A free-text column has an entry for every PII type found in its text, most frequent first, so the
    report can show the column once beside all of them. Replay fields are ``None`` when PII replay did
    not run. Replay results for columns outside the plan, such as every column when PII replacement was
    off, get their own group with no replacement description.
    """
    replay_by_column: dict[str, list[dict]] = {}
    for datum in replay_data:
        replay_by_column.setdefault(datum["column_name"], []).append(datum)

    groups = []
    for row in rows:
        replayed = {datum["pii_type"]: datum for datum in replay_by_column.pop(row["column_name"], [])}
        counts = row["entity_counts"] or {row["entity_type"]: 0}
        entities = [
            _pii_entity(pii_type, found, replayed.get(pii_type, {}))
            for pii_type, found in sorted(counts.items(), key=lambda item: -item[1])
        ]
        groups.append({**row, "entities": entities})
    for column_name, data in replay_by_column.items():
        entities = [_pii_entity(datum["pii_type"], datum["total_training_data"], datum) for datum in data]
        groups.append({"column_name": column_name, "replaced_with": None, "entities": entities})
    return groups


def _pii_replay_note(replay: PIIReplayMetric | None) -> str | None:
    """Return why PII replay results are missing or partial, or ``None`` when replay ran in full."""
    if replay is None:
        return None
    if reason := replay.skipped_reason:
        if replay.entries:
            return f"Only the columns in evaluation.pii_replay_columns were checked: {reason}."
        return f"PII replay was not run: {reason}."
    if replay.enabled is False:
        return "PII replay was turned off with evaluation.pii_replay_enabled."
    return None


def _pii_headline(rows: list[dict], groups: list[dict], replay_note: str | None) -> str:
    """Return the short summary shown on the collapsed PII card, such as ``9 columns replaced``.

    With replacement on, it counts the replaced columns; with replacement off, the columns PII was found in.
    """
    if rows:
        return f"{len(rows)} column{'' if len(rows) == 1 else 's'} replaced"
    if groups:
        return f"PII in {len(groups)} column{'' if len(groups) == 1 else 's'}, not replaced"
    return "Replay not run" if replay_note else "No PII found"


def _pii_entity(pii_type: str, found: int, replay: dict) -> dict:
    """Return one PII type's training count and replay counts, which are ``None`` without a replay result."""
    return {
        "pii_type": pii_type,
        "found": found,
        "unique": replay.get("unique_training_data"),
        "synthetic": replay.get("total_synthetic_data"),
        "replayed": replay.get("unique_synthetic_data"),
        "replay_percentage": replay.get("unique_synthetic_data_percentage"),
    }
