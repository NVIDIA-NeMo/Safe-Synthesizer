# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build ``EvaluationReportData`` from a completed evaluation report.

Each value is copied from an explicit list of fields, so values a component keeps for computation
never reach the export. Chart values come from the same helpers the HTML report charts them with.
"""

from __future__ import annotations

import datetime
import math
import os
from collections.abc import Callable, Mapping
from decimal import Decimal
from enum import Enum
from functools import partial
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel

from ..artifacts.base.fields import FieldFeatures
from ..config.evaluation_report_data import REPORT_DATA_SCHEMA_VERSION, EvaluationReportData
from ..config.parameters import SafeSynthesizerParameters
from ..package_info import __version__
from ..pii_replacer.planning.patterns import describe_pattern
from ..pii_replacer.replacement.descriptions import SAMPLER_LABELS, describe_replacement
from ..pii_replacer.replacement.types import DetectionSource
from ..pii_replacer.transform_result import ColumnStatistics
from .components import multi_modal_figures as figures
from .components.attribute_inference_protection import AttributeInferenceProtection
from .components.autocorrelation_similarity import AutocorrelationSimilarity
from .components.column_distribution import ColumnDistribution, distribution_for_field
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
from .data_model.evaluation_datasets import EvaluationDatasets
from .data_model.evaluation_report import EvaluationReport
from .data_model.evaluation_score import EvaluationScore, PrivacyGrade
from .reports.multimodal.multimodal_report import MultimodalReport

_ComponentT = TypeVar("_ComponentT", bound=Component)


def build_report_data(report: MultimodalReport) -> EvaluationReportData:
    """Return the evaluation results of ``report`` as structured data.

    Raises:
        TypeError: If a value cannot be represented as JSON.
        pydantic.ValidationError: If a value does not fit the schema, such as a required number that is NaN.
    """
    config = report.config
    dp_enabled = bool(config is not None and config.get("dp_enabled"))
    distribution = _find(report, ColumnDistribution)
    datasets = report.evaluation_datasets
    payload = {
        "schema_version": REPORT_DATA_SCHEMA_VERSION,
        "library_version": __version__,
        "job_id": (os.environ.get("NEMO_JOB_ID") or "").strip() or None,
        "scores": {
            "synthetic_quality": _maybe(_component_score, _find(report, SQSScore)),
            "data_privacy": _maybe(_component_score, _find(report, DataPrivacyScore)),
        },
        "metrics": {
            "column_correlation_stability": _maybe(_correlation_metric, _find(report, Correlation)),
            "deep_structure_stability": _maybe(_deep_structure_metric, _find(report, DeepStructure)),
            "column_distribution_stability": _maybe(_metric, distribution),
            "text_semantic_similarity": _maybe(_text_semantic_metric, _find(report, TextSemanticSimilarity)),
            "text_structure_similarity": _maybe(_text_structure_metric, _find(report, TextStructureSimilarity)),
            "membership_inference_protection": _maybe(
                _membership_inference_metric, _find(report, MembershipInferenceProtection)
            ),
            "attribute_inference_protection": _maybe(
                _attribute_inference_metric, _find(report, AttributeInferenceProtection)
            ),
            "pii_replay": _maybe(partial(_pii_replay_metric, config=config), _find(report, PIIReplay)),
            "autocorrelation_similarity": _maybe(_time_series_metric, _find(report, AutocorrelationSimilarity)),
            "window_membership_inference_protection": _maybe(_time_series_metric, _find(report, WindowGroupMIA)),
        },
        "dataset": _maybe(_dataset_counts, _find(report, DatasetStatistics)),
        "columns": _columns(distribution, datasets) if distribution is not None else [],
        "any_column_transformed": any(s.is_transformed for s in (datasets.column_statistics or {}).values()),
        "differential_privacy": {
            "enabled": dp_enabled,
            "epsilon": config.get("epsilon") if dp_enabled and config is not None else None,
            "delta": config.get("delta") if dp_enabled and config is not None else None,
        },
        "pii_replacement": _pii_replacement(config, datasets, _find(report, PIIReplay)),
    }
    return EvaluationReportData.model_validate(to_json_safe(payload))


def to_json_safe(value: object) -> Any:
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
        return to_json_safe(np.asarray(value).tolist())
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
    return {"score": _score(component.score)}


def _points(df: pd.DataFrame) -> dict[str, Any]:
    return {"x": df.iloc[:, 0].tolist(), "y": df.iloc[:, 1].tolist()}


def _projection(training: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any] | None:
    if training.shape[1] < 2 or synthetic.shape[1] < 2:
        return None
    return {
        "axes": [str(c) for c in training.columns[:2]],
        "training": _points(training),
        "synthetic": _points(synthetic),
    }


def _correlation_metric(component: Correlation) -> dict[str, Any]:
    matrices = [
        m
        for m in (component.training_correlation, component.synthetic_correlation, component.correlation_difference)
        if m is not None
    ]
    correlations = None
    if len(matrices) == 3 and not any(m.empty for m in matrices):
        columns = list(matrices[0].columns)
        training, synthetic, difference = (m.reindex(index=columns, columns=columns).to_numpy() for m in matrices)
        correlations = {"columns": columns, "training": training, "synthetic": synthetic, "difference": difference}
    return {**_metric(component), "correlations": correlations}


def _deep_structure_metric(component: DeepStructure) -> dict[str, Any]:
    training, synthetic = component.training_pca, component.synthetic_pca
    projection = None
    if training is not None and synthetic is not None and Component.is_nonempty([training, synthetic]):
        projection = _projection(
            figures.sample_points(training[["pc1", "pc2"]]), figures.sample_points(synthetic[["pc1", "pc2"]])
        )
    return {**_metric(component), "projection": projection}


def _text_semantic_metric(component: TextSemanticSimilarity) -> dict[str, Any]:
    projections = {
        column: _projection(datum.training_pca, datum.synthetic_pca)
        for column, datum in component.text_semantic_similarity_dict.items()
    }
    return {**_metric(component), "projections": {k: v for k, v in projections.items() if v is not None}}


_TEXT_STATISTICS = {
    "sentence_count": "sentence_count",
    "words_per_sentence": "average_words_per_sentence",
    "characters_per_word": "average_characters_per_word",
}


def _text_structure_metric(component: TextStructureSimilarity) -> dict[str, Any]:
    columns = {}
    for column, training in component.training_statistics.items():
        training_records = training.per_record_statistics
        synthetic_records = component.synthetic_statistics[column].per_record_statistics
        if training_records.empty or synthetic_records.empty:
            continue
        columns[column] = {
            name: figures.numeric_distribution(training_records[key], synthetic_records[key])
            for name, key in _TEXT_STATISTICS.items()
        }
    return {**_metric(component), "columns": columns}


def _membership_inference_metric(component: MembershipInferenceProtection) -> dict[str, Any]:
    outcomes = []
    df = component.attack_sum_df
    if df is not None and not df.empty:
        rows = {row["Protection"]: row for row in df.to_dict("records")}
        for grade in reversed(PrivacyGrade):
            row = rows.get(grade.value)
            if grade is PrivacyGrade.UNAVAILABLE or row is None:
                continue
            outcomes.append(
                {"grade": grade.value, "count": row.get("Attack Count"), "percentage": row.get("Attack Percentage")}
            )
    return {**_metric(component), "outcomes": outcomes, "excluded_columns": list(component.excluded_columns)}


def _attribute_inference_metric(component: AttributeInferenceProtection) -> dict[str, Any]:
    df = component.col_accuracy_df
    columns = []
    if df is not None and not df.empty:
        columns = [
            {"column": row["Column"], "protection": row["Risk"], "grade": row["Protection"]}
            for row in df.sort_values(by="Risk").to_dict("records")
        ]
    return {**_metric(component), "columns": columns}


def _pii_replay_metric(component: PIIReplay, config: SafeSynthesizerParameters | None) -> dict[str, Any]:
    return {
        **_metric(component),
        "enabled": bool(config.get("pii_replay_enabled")) if config is not None else None,
        "training_total_records": component.training_total_records,
        "synthetic_total_records": component.synthetic_total_records,
        "entries": [datum.model_dump() for datum in component.pii_replay_data],
        "skipped_reason": component.skipped_reason,
    }


def _dataset_counts(component: DatasetStatistics) -> dict[str, Any]:
    return {
        "training_rows": component.training_rows,
        "training_cols": component.training_cols,
        "training_missing_percent": component.training_missing,
        "synthetic_rows": component.synthetic_rows,
        "synthetic_cols": component.synthetic_cols,
        "synthetic_missing_percent": component.synthetic_missing,
        "memorized_lines": component.memorized_lines,
    }


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


def _columns(component: ColumnDistribution, datasets: EvaluationDatasets) -> list[dict[str, Any]]:
    tabular_columns = set(datasets.get_tabular_columns())
    return [
        {
            "name": field.name,
            "training": _column_profile(field.training_field_features),
            "synthetic": _column_profile(field.synthetic_field_features),
            "distribution_distance": field.distribution_distance,
            "distribution_stability": (
                _score(field.distribution_stability) if field.distribution_stability is not None else None
            ),
            "distribution": (
                distribution_for_field(field, datasets.training[field.name], datasets.synthetic[field.name])
                if field.name in tabular_columns
                else None
            ),
            "pii": _column_pii(field.column_statistics),
        }
        for field in component.evaluation_fields
    ]


def _time_series_metric(component: TimeSeriesComponent) -> dict[str, Any]:
    return {**_metric(component), "details": component.details}


_DETECTION_LABELS = {
    DetectionSource.GLINER: "GLiNER",
    DetectionSource.REGEX: "validated patterns",
    DetectionSource.STRUCTURED_VALUE: "values from replaced columns",
    DetectionSource.DETECTED_VALUE: "values detected elsewhere in the same record or group",
}


def _pii_replacement(
    config: SafeSynthesizerParameters | None, datasets: EvaluationDatasets, replay: PIIReplay | None
) -> dict[str, Any] | None:
    """Return the applied PII replacement plan, or a summary with ``replaced`` false when PII was only detected.

    Counts come from the replacement run's column statistics, which describe the training split only.
    """
    settings = config.replace_pii if config is not None else None
    # A plan is applied only once it is resolved inline; a path or auto-discovery means none was.
    if settings is None or settings.inline_plan is None:
        detected = replay is not None and (replay.pii_replay_data or replay.skipped_reason)
        if not detected and datasets.column_statistics is None:
            return None
        return {
            "replaced": False,
            "columns": [],
            "kept_columns": [],
            "sampler": None,
            "locale": None,
            "seed": None,
            "value_mappings": {},
        }
    plan = settings.inline_plan
    statistics = datasets.column_statistics or {}
    planned = {column.column_name: column for column in plan.columns_to_replace}
    training_columns = list(datasets.training.columns)
    order = {name: index for index, name in enumerate(training_columns)}

    columns = []
    for name, column in sorted(planned.items(), key=lambda item: order.get(item[0], len(order))):
        column_statistics = statistics.get(name)
        functions = column_statistics.transform_functions if column_statistics else set()
        symbols = describe_pattern(column.entity_type, column.pattern) if column.pattern else ()
        columns.append(
            {
                "column_name": name,
                "entity_type": str(column.entity_type.value),
                "replaced_with": describe_replacement(column, settings),
                "pattern": column.pattern,
                "pattern_symbols": [{"symbol": symbol, "meaning": meaning} for symbol, meaning in symbols],
                "depends_on": [
                    {
                        "column_name": dependency.column_name,
                        "entity_type": dependency.entity_type.value if dependency.entity_type else None,
                    }
                    for dependency in column.depends_on
                ],
                "found_by": [label for source, label in _DETECTION_LABELS.items() if source in functions],
                "entity_counts": dict(column_statistics.detected_entity_counts) if column_statistics else {},
            }
        )
    mapping = plan.data_to_sampler_value_mapping or {}
    return {
        "replaced": True,
        "columns": columns,
        "kept_columns": [name for name in training_columns if name not in planned],
        "sampler": SAMPLER_LABELS.get(settings.sampler.backend, settings.sampler.backend.value),
        "locale": settings.locale,
        "seed": settings.seed,
        "value_mappings": {column: dict(values) for column, values in mapping.items()},
    }
