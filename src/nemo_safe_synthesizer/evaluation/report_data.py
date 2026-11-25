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
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from enum import Enum
from functools import partial
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel

from ..artifacts.base.fields import FieldFeatures
from ..config.evaluation_report_data import (
    REPORT_DATA_SCHEMA_VERSION,
    AttackOutcome,
    AttributeInferenceMetric,
    AutocorrelationCounts,
    AutocorrelationDetails,
    AutocorrelationProfile,
    AutocorrelationSimilarityMetric,
    ClosestMatch,
    ColumnPIIStatistics,
    ColumnProfile,
    ColumnProtection,
    ColumnResult,
    ColumnSimilarity,
    CorrelationMatrices,
    CorrelationMetric,
    DatasetCounts,
    DeepStructureMetric,
    DifferentialPrivacySettings,
    EvaluationReportData,
    GroupSelection,
    GroupSimilarity,
    MembershipInferenceMetric,
    Metric,
    Metrics,
    MetricScore,
    PatternSymbol,
    PIIDependency,
    PIIReplacement,
    PIIReplacementColumn,
    PIIReplayEntry,
    PIIReplayMetric,
    Points,
    Projection,
    Scores,
    SkippedComparison,
    TextSemanticMetric,
    TextStructureDistributions,
    TextStructureMetric,
    WindowAttack,
    WindowCounts,
    WindowDistances,
    WindowEvaluationMode,
    WindowGroupSummary,
    WindowMembershipInferenceDetails,
    WindowMembershipInferenceMetric,
)
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
from .components.window_group_mia import WindowGroupMIA
from .data_model.evaluation_datasets import EvaluationDatasets
from .data_model.evaluation_report import EvaluationReport
from .data_model.evaluation_score import EvaluationScore, PrivacyGrade
from .reports.multimodal.multimodal_report import MultimodalReport

_ComponentT = TypeVar("_ComponentT", bound=Component)
_ResultT = TypeVar("_ResultT")

# Group-level rows are capped at what the HTML report shows, so no other group is named in the export.
SKIPPED_MAX_ROWS = 12
GROUP_MAX_ROWS = 12


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
    return EvaluationReportData(
        schema_version=REPORT_DATA_SCHEMA_VERSION,
        library_version=__version__,
        job_id=(os.environ.get("NEMO_JOB_ID") or "").strip() or None,
        scores=Scores(
            synthetic_quality=_maybe(_component_score, _find(report, SQSScore)),
            data_privacy=_maybe(_component_score, _find(report, DataPrivacyScore)),
        ),
        metrics=Metrics(
            column_correlation_stability=_maybe(_correlation_metric, _find(report, Correlation)),
            deep_structure_stability=_maybe(_deep_structure_metric, _find(report, DeepStructure)),
            column_distribution_stability=_maybe(_metric, distribution),
            text_semantic_similarity=_maybe(_text_semantic_metric, _find(report, TextSemanticSimilarity)),
            text_structure_similarity=_maybe(_text_structure_metric, _find(report, TextStructureSimilarity)),
            membership_inference_protection=_maybe(
                _membership_inference_metric, _find(report, MembershipInferenceProtection)
            ),
            attribute_inference_protection=_maybe(
                _attribute_inference_metric, _find(report, AttributeInferenceProtection)
            ),
            pii_replay=_maybe(partial(_pii_replay_metric, config=config), _find(report, PIIReplay)),
            autocorrelation_similarity=_maybe(_autocorrelation_metric, _find(report, AutocorrelationSimilarity)),
            window_membership_inference_protection=_maybe(
                window_membership_inference_metric, _find(report, WindowGroupMIA)
            ),
        ),
        dataset=_maybe(_dataset_counts, _find(report, DatasetStatistics)),
        columns=_columns(distribution, datasets) if distribution is not None else [],
        any_column_transformed=any(s.is_transformed for s in (datasets.column_statistics or {}).values()),
        differential_privacy=DifferentialPrivacySettings(
            enabled=dp_enabled,
            epsilon=to_json_safe(config.get("epsilon")) if dp_enabled and config is not None else None,
            delta=to_json_safe(config.get("delta")) if dp_enabled and config is not None else None,
        ),
        pii_replacement=_pii_replacement(config, datasets, _find(report, PIIReplay)),
    )


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


def _maybe(build: Callable[[_ComponentT], _ResultT], component: _ComponentT | None) -> _ResultT | None:
    return None if component is None else build(component)


def _find(report: EvaluationReport, component_type: type[_ComponentT]) -> _ComponentT | None:
    return next((c for c in report.components if isinstance(c, component_type)), None)


def _score(score: EvaluationScore) -> MetricScore:
    return MetricScore(
        raw_score=to_json_safe(score.raw_score),
        score=to_json_safe(score.score),
        grade=to_json_safe(score.grade),
        notes=score.notes,
    )


def _component_score(component: Component) -> MetricScore:
    return _score(component.score)


def _metric(component: Component) -> Metric:
    return Metric(score=_score(component.score))


def _points(df: pd.DataFrame) -> Points:
    return Points(x=to_json_safe(df.iloc[:, 0].to_numpy()), y=to_json_safe(df.iloc[:, 1].to_numpy()))


def _projection(training: pd.DataFrame, synthetic: pd.DataFrame) -> Projection | None:
    if training.shape[1] < 2 or synthetic.shape[1] < 2:
        return None
    return Projection(
        axes=[str(c) for c in training.columns[:2]],
        training=_points(training),
        synthetic=_points(synthetic),
    )


def _correlation_metric(component: Correlation) -> CorrelationMetric:
    matrices = [
        m
        for m in (component.training_correlation, component.synthetic_correlation, component.correlation_difference)
        if m is not None
    ]
    correlations = None
    if len(matrices) == 3 and not any(m.empty for m in matrices):
        columns = [str(c) for c in matrices[0].columns]
        training, synthetic, difference = (
            to_json_safe(m.reindex(index=matrices[0].columns, columns=matrices[0].columns).to_numpy()) for m in matrices
        )
        correlations = CorrelationMatrices(
            columns=columns, training=training, synthetic=synthetic, difference=difference
        )
    return CorrelationMetric(score=_score(component.score), correlations=correlations)


def _deep_structure_metric(component: DeepStructure) -> DeepStructureMetric:
    training, synthetic = component.training_pca, component.synthetic_pca
    projection = None
    if training is not None and synthetic is not None and Component.is_nonempty([training, synthetic]):
        projection = _projection(
            figures.sample_points(training[["pc1", "pc2"]]), figures.sample_points(synthetic[["pc1", "pc2"]])
        )
    return DeepStructureMetric(score=_score(component.score), projection=projection)


def _text_semantic_metric(component: TextSemanticSimilarity) -> TextSemanticMetric:
    projections = {
        str(column): _projection(datum.training_pca, datum.synthetic_pca)
        for column, datum in component.text_semantic_similarity_dict.items()
    }
    return TextSemanticMetric(
        score=_score(component.score),
        projections={column: projection for column, projection in projections.items() if projection is not None},
    )


def _text_structure_metric(component: TextStructureSimilarity) -> TextStructureMetric:
    columns = {}
    for column, training in component.training_statistics.items():
        training_records = training.per_record_statistics
        synthetic_records = component.synthetic_statistics[column].per_record_statistics
        if training_records.empty or synthetic_records.empty:
            continue
        columns[str(column)] = TextStructureDistributions(
            sentence_count=figures.numeric_distribution(
                training_records["sentence_count"], synthetic_records["sentence_count"]
            ),
            words_per_sentence=figures.numeric_distribution(
                training_records["average_words_per_sentence"], synthetic_records["average_words_per_sentence"]
            ),
            characters_per_word=figures.numeric_distribution(
                training_records["average_characters_per_word"], synthetic_records["average_characters_per_word"]
            ),
        )
    return TextStructureMetric(score=_score(component.score), columns=columns)


def _membership_inference_metric(component: MembershipInferenceProtection) -> MembershipInferenceMetric:
    outcomes = []
    df = component.attack_sum_df
    if df is not None and not df.empty:
        rows = {row["Protection"]: row for row in df.to_dict("records")}
        for grade in reversed(PrivacyGrade):
            row = rows.get(grade.value)
            if grade is PrivacyGrade.UNAVAILABLE or row is None:
                continue
            outcomes.append(
                AttackOutcome(
                    grade=grade.value,
                    count=to_json_safe(row.get("Attack Count")),
                    percentage=to_json_safe(row.get("Attack Percentage")),
                )
            )
    return MembershipInferenceMetric(
        score=_score(component.score), outcomes=outcomes, excluded_columns=list(component.excluded_columns)
    )


def _attribute_inference_metric(component: AttributeInferenceProtection) -> AttributeInferenceMetric:
    df = component.col_accuracy_df
    columns = []
    if df is not None and not df.empty:
        columns = [
            ColumnProtection(
                column=str(row["Column"]),
                protection=to_json_safe(row["Risk"]),
                grade=to_json_safe(row["Protection"]),
            )
            for row in df.sort_values(by="Risk").to_dict("records")
        ]
    return AttributeInferenceMetric(score=_score(component.score), columns=columns)


def _pii_replay_metric(component: PIIReplay, config: SafeSynthesizerParameters | None) -> PIIReplayMetric:
    return PIIReplayMetric(
        score=_score(component.score),
        enabled=bool(config.get("pii_replay_enabled")) if config is not None else None,
        training_total_records=component.training_total_records,
        synthetic_total_records=component.synthetic_total_records,
        entries=[PIIReplayEntry.model_validate(to_json_safe(datum)) for datum in component.pii_replay_data],
        skipped_reason=component.skipped_reason,
    )


def _dataset_counts(component: DatasetStatistics) -> DatasetCounts:
    return DatasetCounts(
        training_rows=component.training_rows,
        training_cols=component.training_cols,
        training_missing_percent=to_json_safe(component.training_missing),
        synthetic_rows=component.synthetic_rows,
        synthetic_cols=component.synthetic_cols,
        synthetic_missing_percent=to_json_safe(component.synthetic_missing),
        memorized_lines=component.memorized_lines,
    )


def _column_profile(features: FieldFeatures) -> ColumnProfile:
    return ColumnProfile(
        type=to_json_safe(features.type),
        count=to_json_safe(features.count),
        unique_count=to_json_safe(features.unique_count),
        unique_percent=to_json_safe(features.unique_percent),
        missing_count=to_json_safe(features.missing_count),
        missing_percent=to_json_safe(features.missing_percent),
        avg_str_length=to_json_safe(features.avg_str_length),
    )


def _column_pii(statistics: ColumnStatistics | None) -> ColumnPIIStatistics | None:
    if statistics is None:
        return None
    return ColumnPIIStatistics(
        assigned_type=statistics.assigned_type,
        assigned_entity=statistics.assigned_entity,
        detected_entity_counts=dict(statistics.detected_entity_counts),
        is_transformed=statistics.is_transformed,
        transform_functions=sorted(statistics.transform_functions),
    )


def _columns(component: ColumnDistribution, datasets: EvaluationDatasets) -> list[ColumnResult]:
    tabular_columns = set(datasets.get_tabular_columns())
    return [
        ColumnResult(
            name=field.name,
            training=_column_profile(field.training_field_features),
            synthetic=_column_profile(field.synthetic_field_features),
            distribution_distance=to_json_safe(field.distribution_distance),
            distribution_stability=(
                _score(field.distribution_stability) if field.distribution_stability is not None else None
            ),
            distribution=(
                distribution_for_field(field, datasets.training[field.name], datasets.synthetic[field.name])
                if field.name in tabular_columns
                else None
            ),
            pii=_column_pii(field.column_statistics),
        )
        for field in component.evaluation_fields
    ]


def _autocorrelation_metric(component: AutocorrelationSimilarity) -> AutocorrelationSimilarityMetric:
    details = to_json_safe(component.details)
    counts, selection = details.get("counts"), details.get("group_selection")
    return AutocorrelationSimilarityMetric(
        score=_score(component.score),
        details=AutocorrelationDetails(
            evaluation_mode=details.get("evaluation_mode"),
            timestamp_column=details.get("timestamp_column"),
            group_column=details.get("group_column"),
            max_lag=details.get("max_lag"),
            min_points=details.get("min_points"),
            counts=AutocorrelationCounts.model_validate(counts) if counts is not None else None,
            group_selection=GroupSelection.model_validate(selection) if selection is not None else None,
            per_group=[GroupSimilarity.model_validate(item) for item in details.get("per_group", [])],
            per_column=[ColumnSimilarity.model_validate(item) for item in details.get("per_column", [])],
            profiles=[
                AutocorrelationProfile.model_validate({"reason": None, **item}) for item in details.get("profiles", [])
            ],
            skipped=[SkippedComparison.model_validate(item) for item in details.get("skipped", [])[:SKIPPED_MAX_ROWS]],
            excluded_columns=details.get("excluded_columns", []),
            groups_only_in_training_count=len(details.get("groups_only_in_training", [])),
            groups_only_in_synthetic_count=len(details.get("groups_only_in_synthetic", [])),
        ),
    )


def _most_flagged_groups(per_group: Sequence[Mapping[str, Any]]) -> list[WindowGroupSummary]:
    flagged = [group for group in per_group if group.get("flag_rate") is not None]
    flagged.sort(key=lambda group: (-group["flag_rate"], str(group["group"])))
    return [WindowGroupSummary.model_validate(group) for group in flagged[:GROUP_MAX_ROWS]]


def window_membership_inference_metric(component: WindowGroupMIA) -> WindowMembershipInferenceMetric:
    """Return the exported results of a window membership inference component."""
    details = to_json_safe(component.details)
    counts, match, distances, attack = (details.get(key) for key in ("counts", "closest_match", "distances", "attack"))
    per_group = details.get("per_group", [])
    return WindowMembershipInferenceMetric(
        score=_score(component.score),
        details=WindowMembershipInferenceDetails(
            evaluation_mode=details.get("evaluation_mode", WindowEvaluationMode.UNAVAILABLE),
            columns=details.get("columns", []),
            excluded_columns=details.get("excluded_columns", []),
            window_size=details.get("window_size"),
            counts=WindowCounts.model_validate(counts) if counts is not None else None,
            exact_copy_rate=details.get("exact_copy_rate"),
            closest_match=None
            if match is None
            else ClosestMatch(
                distance=match["distance"],
                exact_copy=match["exact_copy"],
                synthetic_values=match["synthetic_values"],
                training_values=match["training_values"] if match["exact_copy"] else None,
            ),
            distances=None
            if distances is None
            else WindowDistances(training=distances["training"], holdout=distances.get("holdout")),
            attack=WindowAttack.model_validate(attack) if attack is not None else None,
            training_group_count=len(per_group),
            per_group=_most_flagged_groups(per_group),
        ),
    )


_DETECTION_LABELS = {
    DetectionSource.GLINER: "GLiNER",
    DetectionSource.REGEX: "validated patterns",
    DetectionSource.STRUCTURED_VALUE: "values from replaced columns",
    DetectionSource.DETECTED_VALUE: "values detected elsewhere in the same record or group",
}


def _pii_replacement(
    config: SafeSynthesizerParameters | None, datasets: EvaluationDatasets, replay: PIIReplay | None
) -> PIIReplacement | None:
    """Return the applied PII replacement plan, or a summary with ``replaced`` false when PII was only detected.

    Counts come from the replacement run's column statistics, which describe the training split only.
    """
    settings = config.replace_pii if config is not None else None
    # A plan is applied only once it is resolved inline; a path or auto-discovery means none was.
    if settings is None or settings.inline_plan is None:
        detected = replay is not None and (replay.pii_replay_data or replay.skipped_reason)
        if not detected and datasets.column_statistics is None:
            return None
        return PIIReplacement(
            replaced=False, columns=[], kept_columns=[], sampler=None, locale=None, seed=None, value_mappings={}
        )
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
            PIIReplacementColumn(
                column_name=name,
                entity_type=str(column.entity_type.value),
                replaced_with=describe_replacement(column, settings),
                pattern=column.pattern,
                pattern_symbols=[PatternSymbol(symbol=symbol, meaning=meaning) for symbol, meaning in symbols],
                depends_on=[
                    PIIDependency(
                        column_name=dependency.column_name,
                        entity_type=dependency.entity_type.value if dependency.entity_type else None,
                    )
                    for dependency in column.depends_on
                ],
                found_by=[label for source, label in _DETECTION_LABELS.items() if source in functions],
                entity_counts=dict(column_statistics.detected_entity_counts) if column_statistics else {},
            )
        )
    mapping = plan.data_to_sampler_value_mapping or {}
    return PIIReplacement(
        replaced=True,
        columns=columns,
        kept_columns=[name for name in training_columns if name not in planned],
        sampler=SAMPLER_LABELS.get(settings.sampler.backend, settings.sampler.backend.value),
        locale=settings.locale,
        seed=settings.seed,
        value_mappings=to_json_safe({column: dict(values) for column, values in mapping.items()}),
    )
