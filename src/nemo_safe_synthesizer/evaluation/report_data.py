# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build ``EvaluationReportData`` from a completed evaluation report.

Each card copies only the fields the HTML report displays, listed explicitly, so values a
component keeps for computation never reach the export. Charts are the components'
``report_figures``, the same objects the HTML report embeds.
"""

from __future__ import annotations

import datetime
import json
import math
from collections.abc import Callable, Mapping
from decimal import Decimal
from enum import Enum
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel

from ..artifacts.base.fields import FieldFeatures
from ..config.evaluation_report_data import REPORT_DATA_SCHEMA_VERSION, EvaluationReportData
from ..observability import get_logger
from ..package_info import __version__
from ..pii_replacer.transform_result import ColumnStatistics
from .assets.text.multi_modal_tooltips import tooltips
from .components.attribute_inference_protection import AttributeInferenceProtection
from .components.autocorrelation_similarity import AutocorrelationSimilarity
from .components.column_distribution import ColumnDistribution
from .components.component import Component
from .components.correlation import Correlation
from .components.data_privacy_score import DataPrivacyScore
from .components.dataset_statistics import DatasetStatistics
from .components.deep_structure import DeepStructure
from .components.membership_inference_protection import MembershipInferenceProtection
from .components.pii_replay import PIIReplay
from .components.sqs_score import SQSScore
from .components.text_semantic_similarity import TextSemanticSimilarity
from .components.text_structure_similarity import TextStructureSimilarity
from .components.time_series_common import TimeSeriesComponent
from .components.window_group_mia import WindowGroupMIA
from .data_model.evaluation_report import EvaluationReport
from .data_model.evaluation_score import EvaluationScore

logger = get_logger(__name__)

_ComponentT = TypeVar("_ComponentT", bound=Component)

_SKIPPED_COMPARISONS_SHOWN = 12


def build_report_data(report: EvaluationReport) -> EvaluationReportData:
    """Return the data the HTML report for ``report`` displays.

    Raises:
        TypeError: If a displayed value cannot be represented as JSON.
    """
    context = report.jinja_context
    pii_replacement = context.get("pii_replacement")
    payload = {
        "schema_version": REPORT_DATA_SCHEMA_VERSION,
        "library_version": __version__,
        "job_id": context.get("job_id"),
        "sections": {
            "synthetic_quality": bool(context.get("with_synthesizer")),
            "data_privacy": bool(context.get("with_data_privacy")),
            "transformed_columns": bool(context.get("with_transform")),
            "time_series": bool(context.get("with_time_series")),
        },
        "differential_privacy": {
            "enabled": bool(context.get("dp_enabled")),
            "epsilon": context.get("epsilon"),
            "delta": context.get("delta"),
        },
        "tooltips": {str(key): str(value).strip() for key, value in tooltips.items()},
        "synthetic_quality_score": _maybe(_card, _find(report, SQSScore)),
        "data_privacy_score": _maybe(_card, _find(report, DataPrivacyScore)),
        "dataset_statistics": _maybe(_dataset_statistics_card, _find(report, DatasetStatistics)),
        "column_correlation_stability": _maybe(_figure_card, _find(report, Correlation)),
        "deep_structure_stability": _maybe(_figure_card, _find(report, DeepStructure)),
        "column_distribution_stability": _maybe(_column_distribution_card, _find(report, ColumnDistribution)),
        "text_semantic_similarity": _maybe(_figure_card, _find(report, TextSemanticSimilarity)),
        "text_structure_similarity": _maybe(_figure_card, _find(report, TextStructureSimilarity)),
        "membership_inference_protection": _maybe(
            _membership_inference_card, _find(report, MembershipInferenceProtection)
        ),
        "attribute_inference_protection": _maybe(_figure_card, _find(report, AttributeInferenceProtection)),
        "autocorrelation_similarity": _maybe(_autocorrelation_card, _find(report, AutocorrelationSimilarity)),
        "window_membership_inference_protection": _maybe(_window_mia_card, _find(report, WindowGroupMIA)),
        "pii_replay": _maybe(_pii_replay_card, _find(report, PIIReplay)),
        "pii_replacement": _pii_replacement(pii_replacement) if pii_replacement else None,
    }
    return EvaluationReportData.model_validate(to_json_safe(payload))


def figure_to_json(figure: go.Figure) -> dict[str, Any]:
    """Serialize a figure with Plotly's encoder, without the theme template frontends supply themselves."""
    data: dict[str, Any] = json.loads(figure.to_json(validate=False))
    layout = data.get("layout")
    if isinstance(layout, dict):
        layout.pop("template", None)
    return data


def to_json_safe(value: Any) -> Any:
    """Convert ``value`` to plain JSON types.

    NaN and infinities, ``NaT``, and ``None`` become ``None``; numpy scalars and arrays become Python numbers and
    lists; enums become their values; sets become sorted lists; dates and times become ISO 8601 strings, and
    durations ISO 8601 durations such as ``P0DT0H1M30S``.

    Raises:
        TypeError: For any other type.
    """
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, BaseModel):
        return to_json_safe(value.model_dump())
    if isinstance(value, Mapping):
        return {str(to_json_safe(key)): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, set | frozenset):
        return sorted((to_json_safe(item) for item in value), key=str)
    if isinstance(value, list | tuple):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, np.datetime64 | np.timedelta64):
        return to_json_safe(pd.Timestamp(value) if isinstance(value, np.datetime64) else pd.Timedelta(value))
    if isinstance(value, datetime.timedelta):
        return pd.Timedelta(value).isoformat()
    if isinstance(value, datetime.date | datetime.time):
        return value.isoformat()
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, Enum):
        return to_json_safe(value.value)
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str | int | bool):
        return value
    raise TypeError(f"Cannot represent {type(value).__name__} as JSON in evaluation report data")


def _maybe(build: Callable[[_ComponentT], dict[str, Any]], component: _ComponentT | None) -> dict[str, Any] | None:
    return None if component is None else build(component)


def _find(report: EvaluationReport, component_type: type[_ComponentT]) -> _ComponentT | None:
    return next((c for c in report.components if isinstance(c, component_type)), None)


def _score(score: EvaluationScore) -> dict[str, Any]:
    return {"raw_score": score.raw_score, "score": score.score, "grade": score.grade, "notes": score.notes}


def _card(component: Component) -> dict[str, Any]:
    return {"name": component.name, "score": _score(component.score)}


def _figure_card(component: Component) -> dict[str, Any]:
    figures = [{"title": title, "figure": figure_to_json(figure)} for title, figure in component.report_figures.items()]
    return {**_card(component), "figures": figures}


def _dataset_statistics_card(component: DatasetStatistics) -> dict[str, Any]:
    counts = component.model_dump(
        include={
            "training_rows",
            "training_cols",
            "training_missing",
            "synthetic_rows",
            "synthetic_cols",
            "synthetic_missing",
            "memorized_lines",
        }
    )
    return {**_card(component), **counts}


def _membership_inference_card(component: MembershipInferenceProtection) -> dict[str, Any]:
    return {**_figure_card(component), "excluded_columns": list(component.excluded_columns)}


def _column_profile(features: FieldFeatures) -> dict[str, Any]:
    return {
        "type": features.type,
        "count": features.count,
        "unique_count": features.unique_count,
        "unique_percent": features.unique_percent,
        "missing_count": features.missing_count,
        "missing_percent": features.missing_percent,
        "avg_str_length": features.avg_str_length,
    }


def _column_pii(statistics: ColumnStatistics | None) -> dict[str, Any] | None:
    if statistics is None:
        return None
    return {
        "assigned_type": statistics.assigned_type,
        "assigned_entity": statistics.assigned_entity,
        "detected_entity_counts": dict(statistics.detected_entity_counts),
        "is_transformed": statistics.is_transformed,
        "transform_functions": sorted(statistics.transform_functions),
    }


def _column_distribution_card(component: ColumnDistribution) -> dict[str, Any]:
    columns = [
        {
            "name": field.name,
            "training": _column_profile(field.training_field_features),
            "synthetic": _column_profile(field.synthetic_field_features),
            "distribution_distance": field.distribution_distance,
            "distribution_stability": (
                _score(field.distribution_stability) if field.distribution_stability is not None else None
            ),
            "pii": _column_pii(field.column_statistics),
        }
        for field in component.evaluation_fields
    ]
    return {**_figure_card(component), "columns": columns}


def _pii_replay_card(component: PIIReplay) -> dict[str, Any]:
    return {
        **_card(component),
        "training_total_records": component.training_total_records,
        "synthetic_total_records": component.synthetic_total_records,
        "entries": [datum.model_dump() for datum in component.pii_replay_data],
        "skipped_reason": component.skipped_reason,
    }


def _pii_replacement(pii: Mapping[str, Any]) -> dict[str, Any]:
    columns = [
        {
            "column_name": group["column_name"],
            "replaced_with": group.get("replaced_with"),
            "pattern": group.get("pattern"),
            "pattern_symbols": [
                {"symbol": symbol, "meaning": meaning} for symbol, meaning in group.get("pattern_symbols", ())
            ],
            "depends_on": [
                {"column_name": dependency["column_name"], "entity_type": dependency["entity_type"]}
                for dependency in group.get("depends_on", [])
            ],
            "found_by": list(group.get("found_by", [])),
            "entities": [
                {
                    key: entity[key]
                    for key in ("pii_type", "found", "unique", "synthetic", "replayed", "replay_percentage")
                }
                for entity in group["entities"]
            ],
        }
        for group in pii.get("groups", [])
    ]
    return {
        "replacement_off": bool(pii.get("replacement_off")),
        "headline": pii["headline"],
        "sampler": pii.get("sampler"),
        "locale": pii.get("locale"),
        "seed": pii.get("seed"),
        "replay_note": pii.get("replay_note"),
        "columns": columns,
        "kept_columns": list(pii.get("kept_columns", [])),
        "value_mappings": pii.get("mappings", {}),
    }


def _time_series_card(component: TimeSeriesComponent) -> dict[str, Any]:
    context = component.jinja_context
    details = component.details
    figures = [
        {
            "title": title,
            "figure": figure_to_json(figure),
            "column_select": column_select,
            "tooltip": str(tooltips[tooltip_key]).strip(),
        }
        for title, figure, column_select, tooltip_key in component.figure_specs
    ]
    skipped = [
        {
            "column": item["column"],
            "group": None if item.get("group") is None else str(item["group"]),
            "reason": item["reason"],
        }
        for item in details.get("skipped", [])[:_SKIPPED_COMPARISONS_SHOWN]
    ]
    return {
        **_card(component),
        "section": component.report_section,
        "description": component.description,
        "score_colored_traces": component.score_colored_traces,
        "figures": figures,
        "summary_columns": list(context.get("summary_columns", [])),
        "summary_traces_per_column": context.get("summary_traces_per_column", 1),
        "skipped": skipped,
        "groups_only_in_training": len(details.get("groups_only_in_training", [])),
        "groups_only_in_synthetic": len(details.get("groups_only_in_synthetic", [])),
    }


def _autocorrelation_card(component: AutocorrelationSimilarity) -> dict[str, Any]:
    context = component.jinja_context
    counts = {
        key: context[key]
        for key in (
            "evaluated_profile_count",
            "evaluated_group_count",
            "evaluated_column_count",
            "pair_score_column_count",
            "constant_synthetic_profile_count",
        )
    }
    return {**_time_series_card(component), **counts}


def _window_mia_card(component: WindowGroupMIA) -> dict[str, Any]:
    details = component.details
    attack = details.get("attack")
    closest_match = details.get("closest_match")
    return {
        **_time_series_card(component),
        "evaluation_mode": details.get("evaluation_mode"),
        "window_size": details.get("window_size"),
        "training_group_count": len(details.get("per_group", [])),
        "counts": details.get("counts"),
        "exact_copy_rate": details.get("exact_copy_rate"),
        "attack": (
            {"training_flag_rate": attack["training_flag_rate"], "holdout_flag_rate": attack["holdout_flag_rate"]}
            if attack
            else None
        ),
        "closest_match_exact_copy": closest_match["exact_copy"] if closest_match else None,
        "closest_match_distance_per_value": component.jinja_context.get("closest_match_distance_per_value"),
    }
