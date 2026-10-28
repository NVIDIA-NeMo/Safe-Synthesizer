# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from functools import cached_property
from typing import Any

import pandas as pd
from pydantic import Field

from ....config.evaluate import (
    DEFAULT_RECORD_COUNT,
    DEFAULT_SQS_REPORT_COLUMNS,
)
from ....config.parameters import SafeSynthesizerParameters
from ....evaluation.assets.text.multi_modal_tooltips import tooltips
from ....evaluation.components.attribute_inference_protection import AttributeInferenceProtection
from ....evaluation.components.autocorrelation_similarity import AutocorrelationSimilarity
from ....evaluation.components.autocorrelation_similarity_figures import (
    AUTOCORRELATION_PAIR_SCORE_MAX_COLUMNS,
    AUTOCORRELATION_SUMMARY_TRACES_PER_COLUMN,
    generate_autocorrelation_lag_error_figure,
    generate_autocorrelation_pair_score_figure,
    generate_autocorrelation_summary_figure,
    order_autocorrelation_columns,
)
from ....evaluation.components.column_distribution import (
    ColumnDistribution,
    ColumnDistributionPlotRow,
)
from ....evaluation.components.component import Component
from ....evaluation.components.correlation import (
    Correlation,
)
from ....evaluation.components.data_privacy_score import DataPrivacyScore
from ....evaluation.components.dataset_statistics import (
    DatasetStatistics,
)
from ....evaluation.components.deep_structure import (
    DeepStructure,
)
from ....evaluation.components.membership_inference_protection import (
    MembershipInferenceProtection,
)
from ....evaluation.components.pii_replay import PIIReplay
from ....evaluation.components.sqs_score import SQSScore
from ....evaluation.components.text_semantic_similarity import TextSemanticSimilarity
from ....evaluation.components.text_structure_similarity import TextStructureSimilarity
from ....evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ....evaluation.data_model.evaluation_report import EvaluationReport
from ....evaluation.data_model.evaluation_score import (
    EvaluationScore,
    PrivacyGrade,
)
from ....observability import get_logger
from ....pii_replacer.planning.patterns import describe_pattern
from ....pii_replacer.replacement.descriptions import SAMPLER_LABELS, describe_replacement
from ....pii_replacer.transform_result import ColumnStatistics

logger = get_logger(__name__)

# Report names for the free-text detectors recorded in ``ColumnStatistics.transform_functions``, in report order.
_DETECTION_LABELS = {
    "gliner": "GLiNER",
    "structured_value": "values from replaced columns",
    "detected_value": "values detected in other rows",
}


def _pii_column_groups(rows: list[dict], replay_data: list[dict]) -> list[dict]:
    """Join each planned column with its PII replay results, with one entry per PII type found in it.

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


def _pii_replay_note(config: SafeSynthesizerParameters | None, replay: dict) -> str | None:
    """Return why PII replay results are missing, or ``None`` when replay ran."""
    if reason := replay.get("skipped_reason"):
        return f"PII replay was not run: {reason}."
    if config is not None and not config.get("pii_replay_enabled"):
        return "PII replay was turned off with evaluation.pii_replay_enabled."
    return None


def _pii_headline(rows: list[dict], groups: list[dict], replay_note: str | None) -> str:
    """Return the short summary shown on the collapsed PII card, such as ``9 columns replaced``.

    With replacement on, it counts the planned columns; with replacement off, the columns PII was found in.
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


def _pii_replacement_context(
    config: SafeSynthesizerParameters | None, evaluation_datasets: EvaluationDatasets | None
) -> dict | None:
    """Return the PII replacement plan as table rows, or ``None`` when no plan was applied.

    Rows follow the training column order. Counts come from the replacement run's column
    statistics, which describe the training split only; the test split is never replaced.
    """
    settings = config.replace_pii if config is not None else None
    # A plan is applied only once it is resolved inline; a path or auto-discovery means none was.
    if settings is None or settings.inline_plan is None or evaluation_datasets is None:
        return None
    plan = settings.inline_plan
    statistics = evaluation_datasets.column_statistics or {}
    planned = {column.column_name: column for column in plan.columns_to_replace}
    training_columns = list(evaluation_datasets.training.columns)
    order = {name: index for index, name in enumerate(training_columns)}

    rows = []
    for name, column in sorted(planned.items(), key=lambda item: order.get(item[0], len(order))):
        column_statistics = statistics.get(name)
        functions = column_statistics.transform_functions if column_statistics else set()
        rows.append(
            {
                "column_name": name,
                "entity_type": str(column.entity_type.value),
                "replaced_with": describe_replacement(column, settings),
                "pattern": column.pattern,
                "pattern_symbols": describe_pattern(column.entity_type, column.pattern) if column.pattern else (),
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
        "rows": rows,
        "kept_columns": [name for name in training_columns if name not in planned],
        "sampler": SAMPLER_LABELS.get(settings.sampler.backend, settings.sampler.backend.value),
        "locale": settings.locale,
        "seed": settings.seed,
        "mappings": {column: dict(values) for column, values in mapping.items()},
    }


class MultimodalReport(EvaluationReport):
    """Multi-modal evaluation report combining quality and privacy components.

    Assembles all evaluation components (SQS sub-metrics for tabular and/or text columns,
    privacy scores, PII replay, dataset statistics) from paired training/synthetic
    dataframes and renders them into an HTML report via Jinja2 templates.

    Use ``from_dataframes`` to construct a fully populated report.
    """

    config: SafeSynthesizerParameters | None = Field(
        default=None, description="Pipeline configuration parameters used for this evaluation."
    )

    @cached_property
    def jinja_context(self) -> dict:
        """Template context with tooltips, flags, and per-column distribution figures."""
        try:
            ctx = super().jinja_context
            ctx["tooltips"] = tooltips
            # Job ID from the runtime environment (e.g. cluster); omit from HTML when unset.
            raw_job_id = os.environ.get("NEMO_JOB_ID")
            ctx["job_id"] = (raw_job_id or "").strip() or None

            # Shorthands used in template
            ctx["with_synthesizer"] = False
            for c in self.components:
                if c.name == "Synthetic Quality Score" and c.score and c.score.grade.value != "Unavailable":
                    ctx["with_synthesizer"] = True

            ctx["with_data_privacy"] = False
            for c in self.components:
                if c.name == "Data Privacy Score" and c.score and c.score.grade.value != "Unavailable":
                    ctx["with_data_privacy"] = True

            statistics = self.evaluation_datasets.column_statistics if self.evaluation_datasets is not None else None
            # Statistics from detection alone, with replacement off, mark no column transformed and add no columns.
            ctx["with_transform"] = bool(statistics) and any(s.is_transformed for s in statistics.values())

            config = self.config
            ctx["dp_enabled"] = config is not None and config.get("dp_enabled")
            if ctx["dp_enabled"] and config is not None:
                ctx["delta"] = config.get("delta")
                ctx["epsilon"] = config.get("epsilon")

            replay = ctx.get("pii_replay") or {}
            replay_data = replay.get("pii_replay_data") or []
            pii: dict | None = _pii_replacement_context(config, self.evaluation_datasets)
            rows: list[dict] = pii["rows"] if pii is not None else []
            # With replacement off, PII is detected only for replay; the table then comes from replay alone.
            if pii is None and (replay_data or replay.get("skipped_reason") or statistics is not None):
                pii = {"replacement_off": True, "kept_columns": [], "mappings": {}}
            if pii is not None:
                groups = _pii_column_groups(rows, replay_data)
                replay_note = _pii_replay_note(config, replay)
                pii.update(groups=groups, replay_note=replay_note, headline=_pii_headline(rows, groups, replay_note))
            ctx["pii_replacement"] = pii

            # Numeric per-column figures require access to the original data, a little hacky.
            if "column_distribution_stability" in ctx:
                ctx["column_distribution_stability"]["figures"] = ColumnDistributionPlotRow.from_evaluation_datasets(
                    self.evaluation_datasets
                )

            ctx["with_time_series"] = "autocorrelation_similarity" in ctx
            if ctx["with_time_series"]:
                autocorrelation = ctx["autocorrelation_similarity"]
                profiles = autocorrelation["details"].get("profiles", [])
                autocorrelation["figures"] = []
                autocorrelation["summary_columns"] = []
                if profiles:
                    figures = [
                        (
                            "Typical autocorrelation",
                            generate_autocorrelation_summary_figure(profiles),
                            True,
                            "autocorrelation_typical_info",
                        ),
                        (
                            "Difference by lag",
                            generate_autocorrelation_lag_error_figure(profiles),
                            False,
                            "autocorrelation_difference_info",
                        ),
                        (
                            "Pair scores",
                            generate_autocorrelation_pair_score_figure(profiles, autocorrelation["score"]["score"]),
                            False,
                            "autocorrelation_pair_scores_info",
                        ),
                    ]
                    autocorrelation["figures"] = [
                        {
                            "title": title,
                            "html": figure.to_html(full_html=False, include_plotlyjs=False),
                            "column_select": column_select,
                            "tooltip_id": tooltip_key.replace("_", "-"),
                            "tooltip": tooltips[tooltip_key],
                        }
                        for title, figure, column_select, tooltip_key in figures
                    ]
                    autocorrelation["summary_columns"] = order_autocorrelation_columns(profiles)
                autocorrelation["summary_traces_per_column"] = AUTOCORRELATION_SUMMARY_TRACES_PER_COLUMN
                autocorrelation["pair_score_column_count"] = min(
                    len(autocorrelation["summary_columns"]), AUTOCORRELATION_PAIR_SCORE_MAX_COLUMNS
                )
                autocorrelation["evaluated_profile_count"] = len(profiles)
                autocorrelation["evaluated_group_count"] = len({str(item["group"]) for item in profiles})
                autocorrelation["evaluated_column_count"] = len({item["column"] for item in profiles})
                autocorrelation["constant_synthetic_profile_count"] = sum(
                    1 for item in profiles if item["evaluated_lags"] == 0
                )

            return ctx
        except Exception:
            logger.exception("Failed to get jinja context")
            raise

    @staticmethod
    def _get_config_value(param: str, default: Any, config: SafeSynthesizerParameters | None = None) -> Any:
        """Return a config parameter value, falling back to ``default``."""
        if config and config.get(param):
            return config.get(param)
        return default

    @staticmethod
    def from_dataframes(
        training: pd.DataFrame,
        synthetic: pd.DataFrame,
        test: pd.DataFrame | None = None,
        column_statistics: dict[str, ColumnStatistics] | None = None,
        config: SafeSynthesizerParameters | None = None,
        pii_replay_skipped_reason: str | None = None,
    ) -> MultimodalReport:
        """Build a complete multi-modal evaluation report from dataframes.

        Constructs an ``EvaluationDatasets``, runs all enabled evaluation
        components (quality and privacy), and assembles them into a report.

        Args:
            training: Training dataframe.
            synthetic: Synthetic dataframe.
            test: Optional holdout dataframe for privacy metrics.
            column_statistics: Per-column PII entity metadata.
            config: Pipeline configuration controlling which metrics are enabled.
            pii_replay_skipped_reason: Why PII replay could not run, shown in the report, or ``None``.

        Returns:
            A fully populated ``MultimodalReport`` ready for rendering.
        """
        evaluation_datasets = EvaluationDatasets.from_dataframes(
            training=training,
            synthetic=synthetic,
            test=test,
            column_statistics=column_statistics,
            rows=MultimodalReport._get_config_value("sqs_report_rows", DEFAULT_RECORD_COUNT, config),
            cols=MultimodalReport._get_config_value("sqs_report_columns", DEFAULT_SQS_REPORT_COLUMNS, config),
            mandatory_columns=MultimodalReport._get_config_value("mandatory_columns", [], config),
        )

        components: list[Component] = []

        attribute_inference_protection = AttributeInferenceProtection(
            score=EvaluationScore(grade=PrivacyGrade.UNAVAILABLE)
        )
        if config and config.get("aia_enabled"):
            attribute_inference_protection = AttributeInferenceProtection.from_evaluation_datasets(
                evaluation_datasets, config
            )
        components.append(attribute_inference_protection)

        membership_inference_protection = MembershipInferenceProtection(
            score=EvaluationScore(grade=PrivacyGrade.UNAVAILABLE)
        )
        if config and config.get("mia_enabled"):
            membership_inference_protection = MembershipInferenceProtection.from_evaluation_datasets(
                evaluation_datasets, config
            )
        components.append(membership_inference_protection)

        data_privacy_score = DataPrivacyScore(score=EvaluationScore(grade=PrivacyGrade.UNAVAILABLE))
        if attribute_inference_protection or membership_inference_protection:
            data_privacy_score = DataPrivacyScore.from_components(
                [attribute_inference_protection, membership_inference_protection]
            )
        components.append(data_privacy_score)

        pii_replay = PIIReplay(skipped_reason=pii_replay_skipped_reason)
        if column_statistics and config is not None and config.get("pii_replay_enabled"):
            # PII Replay requires full df's. Make a one-off dataset for that call.
            pii_replay = PIIReplay.from_evaluation_datasets(
                EvaluationDatasets.from_dataframes(
                    training=training,
                    synthetic=synthetic,
                    test=test,
                    column_statistics=column_statistics,
                    rows=MultimodalReport._get_config_value("sqs_report_rows", DEFAULT_RECORD_COUNT, config),
                    cols=MultimodalReport._get_config_value("sqs_report_columns", DEFAULT_SQS_REPORT_COLUMNS, config),
                    mandatory_columns=MultimodalReport._get_config_value("mandatory_columns", [], config),
                    enable_sampling=False,
                ),
                config=config,
            )
        components.append(pii_replay)

        dataset_statistics = DatasetStatistics.from_evaluation_datasets(evaluation_datasets)

        column_distribution = ColumnDistribution.from_evaluation_datasets(evaluation_datasets)
        correlation = Correlation.from_evaluation_datasets(evaluation_datasets)
        deep_structure = DeepStructure.from_evaluation_datasets(evaluation_datasets)
        text_semantic_similarity = TextSemanticSimilarity.from_evaluation_datasets(evaluation_datasets)
        text_structure_similarity = TextStructureSimilarity.from_evaluation_datasets(evaluation_datasets)
        sqs_score = SQSScore.from_components(
            [column_distribution, correlation, deep_structure, text_semantic_similarity, text_structure_similarity]
        )

        components += [
            dataset_statistics,
            column_distribution,
            correlation,
            deep_structure,
            text_semantic_similarity,
            text_structure_similarity,
            sqs_score,
        ]

        if config is not None and config.time_series_evaluation_enabled:
            time_series_datasets = EvaluationDatasets.from_dataframes(
                training=training,
                synthetic=synthetic,
                test=test,
                column_statistics=column_statistics,
                enable_sampling=False,
            )
            components.append(AutocorrelationSimilarity.from_evaluation_datasets(time_series_datasets, config))

        report = MultimodalReport(config=config, evaluation_datasets=evaluation_datasets, components=components)
        report.evaluation_datasets = evaluation_datasets
        return report
