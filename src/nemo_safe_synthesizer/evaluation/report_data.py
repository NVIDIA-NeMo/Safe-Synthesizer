# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build ``EvaluationReportData`` from a completed evaluation report.

Each value is copied from an explicit list of fields, so values a component keeps for computation
never reach the export. Charts are the components' ``report_figures``, the same objects the HTML
report embeds.
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
from ..package_info import __version__
from ..pii_replacer.transform_result import ColumnStatistics
from .components.attribute_inference_protection import AttributeInferenceProtection
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
from .data_model.evaluation_report import EvaluationReport
from .data_model.evaluation_score import EvaluationScore

_ComponentT = TypeVar("_ComponentT", bound=Component)


def build_report_data(report: EvaluationReport) -> EvaluationReportData:
    """Return the evaluation results of ``report`` as structured data.

    Raises:
        TypeError: If a value cannot be represented as JSON.
    """
    context = report.jinja_context
    distribution = _find(report, ColumnDistribution)
    payload = {
        "schema_version": REPORT_DATA_SCHEMA_VERSION,
        "library_version": __version__,
        "job_id": context.get("job_id"),
        "scores": {
            "synthetic_quality": _maybe(_component_score, _find(report, SQSScore)),
            "data_privacy": _maybe(_component_score, _find(report, DataPrivacyScore)),
        },
        "metrics": {
            "column_correlation_stability": _maybe(_metric, _find(report, Correlation)),
            "deep_structure_stability": _maybe(_metric, _find(report, DeepStructure)),
            "column_distribution_stability": _maybe(_metric, distribution),
            "text_semantic_similarity": _maybe(_metric, _find(report, TextSemanticSimilarity)),
            "text_structure_similarity": _maybe(_metric, _find(report, TextStructureSimilarity)),
            "membership_inference_protection": _maybe(
                _membership_inference_metric, _find(report, MembershipInferenceProtection)
            ),
            "attribute_inference_protection": _maybe(_metric, _find(report, AttributeInferenceProtection)),
            "pii_replay": _maybe(_pii_replay_metric, _find(report, PIIReplay)),
        },
        "dataset": _maybe(_dataset_counts, _find(report, DatasetStatistics)),
        "columns": _columns(distribution) if distribution is not None else [],
        "differential_privacy": {
            "enabled": bool(context.get("dp_enabled")),
            "epsilon": context.get("epsilon"),
            "delta": context.get("delta"),
        },
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


def _maybe(build: Callable[[_ComponentT], Any], component: _ComponentT | None) -> Any:
    return None if component is None else build(component)


def _find(report: EvaluationReport, component_type: type[_ComponentT]) -> _ComponentT | None:
    return next((c for c in report.components if isinstance(c, component_type)), None)


def _score(score: EvaluationScore) -> dict[str, Any]:
    return {"raw_score": score.raw_score, "score": score.score, "grade": score.grade, "notes": score.notes}


def _component_score(component: Component) -> dict[str, Any]:
    return _score(component.score)


def _metric(component: Component) -> dict[str, Any]:
    figures = [{"title": title, "figure": figure_to_json(figure)} for title, figure in component.report_figures.items()]
    return {"score": _score(component.score), "figures": figures}


def _membership_inference_metric(component: MembershipInferenceProtection) -> dict[str, Any]:
    return {**_metric(component), "excluded_columns": list(component.excluded_columns)}


def _pii_replay_metric(component: PIIReplay) -> dict[str, Any]:
    return {
        **_metric(component),
        "training_total_records": component.training_total_records,
        "synthetic_total_records": component.synthetic_total_records,
        "entries": [datum.model_dump() for datum in component.pii_replay_data],
        "skipped_reason": component.skipped_reason,
    }


def _dataset_counts(component: DatasetStatistics) -> dict[str, Any]:
    return component.model_dump(
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


def _columns(component: ColumnDistribution) -> list[dict[str, Any]]:
    return [
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
