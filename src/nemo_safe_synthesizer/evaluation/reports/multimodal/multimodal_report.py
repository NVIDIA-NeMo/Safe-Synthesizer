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
from ....evaluation.components.adf_stationarity import ADFStationarity
from ....evaluation.components.autocorrelation_similarity import AutocorrelationSimilarity
from ....evaluation.components.column_distribution import (
    ColumnDistribution,
    ColumnDistributionPlotRow,
)
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
from ....evaluation.components.drift_similarity import DriftSimilarity
from ....evaluation.components.dtw_similarity import DTWSimilarity
from ....evaluation.components.hurst_similarity import HurstSimilarity
from ....evaluation.components.membership_inference_protection import (
    MembershipInferenceProtection,
)
from ....evaluation.components.pii_replay import PIIReplay
from ....evaluation.components.rolling_stats_similarity import RollingStatsSimilarity
from ....evaluation.components.spectral_similarity import SpectralSimilarity
from ....evaluation.components.sqs_score import SQSScore
from ....evaluation.components.text_semantic_similarity import TextSemanticSimilarity
from ....evaluation.components.text_structure_similarity import TextStructureSimilarity
from ....evaluation.components.time_series_similarity_score import TimeSeriesSimilarityScore
from ....evaluation.data_model.evaluation_dataset import EvaluationDataset
from ....evaluation.data_model.evaluation_report import EvaluationReport
from ....evaluation.data_model.evaluation_score import (
    EvaluationScore,
    PrivacyGrade,
)
from ....observability import get_logger
from ....pii_replacer.transform_result import ColumnStatistics

logger = get_logger(__name__)


class MultimodalReport(EvaluationReport):
    config: SafeSynthesizerParameters | None = Field(default=None)

    @cached_property
    def jinja_context(self):
        try:
            ctx = super().jinja_context
            ctx["tooltips"] = tooltips
            # Get the job ID if it exist or create a new one.
            ctx["job_id"] = os.environ.get("NEMO_JOB_ID") or "N/A"

            # Shorthands used in template
            ctx["with_synthesizer"] = False
            for c in self.components:
                if c.name == "Synthetic Quality Score" and c.score and c.score.grade.value != "Unavailable":
                    ctx["with_synthesizer"] = True

            ctx["with_data_privacy"] = False
            for c in self.components:
                if c.name == "Data Privacy Score" and c.score and c.score.grade.value != "Unavailable":
                    ctx["with_data_privacy"] = True

            ctx["with_transform"] = False
            if (
                self.evaluation_dataset is not None
                and self.evaluation_dataset.column_statistics is not None
                and len(self.evaluation_dataset.column_statistics) > 0
            ):
                ctx["with_transform"] = True

            ctx["dp_enabled"] = self.config and self.config.get("dp_enabled")
            if ctx["dp_enabled"]:
                ctx["delta"] = self.config.get("delta")
                ctx["epsilon"] = self.config.get("epsilon")

            ctx["with_time_series"] = any(
                [
                    ctx.get("autocorrelation_similarity"),
                    ctx.get("drift_similarity"),
                    ctx.get("hurst_similarity"),
                    ctx.get("dtw_similarity"),
                    ctx.get("rolling_stats_similarity"),
                    ctx.get("spectral_similarity"),
                    ctx.get("time_series_similarity_score"),
                ]
            )

            # Numeric per-column figures require access to the original data, a little hacky.
            if "column_distribution_stability" in ctx:
                ctx["column_distribution_stability"]["figures"] = ColumnDistributionPlotRow.from_evaluation_dataset(
                    self.evaluation_dataset
                )

            return ctx
        except Exception:
            logger.exception("Failed to get jinja context")
            raise

    @staticmethod
    def _get_config_value(param: str, default: Any, config: SafeSynthesizerParameters | None = None):
        if config and config.get(param):
            return config.get(param)
        return default

    @staticmethod
    def _collect_time_series_columns(config: SafeSynthesizerParameters | None) -> list[str]:
        if config is None or not getattr(config, "evaluation", None):
            return []
        ts_cfg = getattr(config.evaluation, "time_series", None)
        if not ts_cfg or not ts_cfg.enabled:
            return []

        cols: set[str] = set()

        def _add(values):
            if not values:
                return
            if isinstance(values, str):
                cols.add(values)
            else:
                cols.update([v for v in values if v])

        autocorrelation = getattr(ts_cfg, "autocorrelation", None)
        if autocorrelation and autocorrelation.enabled:
            _add(autocorrelation.value_columns)
            _add(autocorrelation.timestamp_column)
            _add(autocorrelation.group_column)

        return [c for c in cols if c]

    @staticmethod
    def from_dataframes(
        reference: pd.DataFrame,
        output: pd.DataFrame,
        test: pd.DataFrame | None = None,
        column_statistics: dict[str, ColumnStatistics] | None = None,
        config: SafeSynthesizerParameters | None = None,
    ) -> MultimodalReport:
        # Check is_timeseries directly since config.get() doesn't find nested attributes reliably
        is_timeseries = config.time_series.is_timeseries if config and getattr(config, "time_series", None) else False

        evaluation_dataset = EvaluationDataset.from_dataframes(
            reference=reference,
            output=output,
            test=test,
            column_statistics=column_statistics,
            rows=MultimodalReport._get_config_value("sqs_rows", DEFAULT_RECORD_COUNT, config),
            cols=MultimodalReport._get_config_value("sqs_columns", DEFAULT_SQS_REPORT_COLUMNS, config),
            mandatory_columns=list(
                set(
                    (MultimodalReport._get_config_value("mandatory_columns", [], config) or [])
                    + MultimodalReport._collect_time_series_columns(config)
                )
            ),
            enable_sampling=not is_timeseries,
        )

        components = []

        attribute_inference_protection = AttributeInferenceProtection(
            score=EvaluationScore(grade=PrivacyGrade.UNAVAILABLE)
        )
        if config and config.get("enable_synthesis") and config.get("aia_enabled"):
            attribute_inference_protection = AttributeInferenceProtection.from_evaluation_dataset(
                evaluation_dataset, config
            )
        components.append(attribute_inference_protection)

        membership_inference_protection = MembershipInferenceProtection(
            score=EvaluationScore(grade=PrivacyGrade.UNAVAILABLE)
        )
        if config and config.get("enable_synthesis") and config.get("mia_enabled"):
            membership_inference_protection = MembershipInferenceProtection.from_evaluation_dataset(evaluation_dataset)
        components.append(membership_inference_protection)

        data_privacy_score = DataPrivacyScore(score=EvaluationScore(grade=PrivacyGrade.UNAVAILABLE))
        if attribute_inference_protection or membership_inference_protection:
            data_privacy_score = DataPrivacyScore.from_components(
                [attribute_inference_protection, membership_inference_protection]
            )
        components.append(data_privacy_score)

        pii_replay = PIIReplay()
        if column_statistics and config is not None and config.get("pii_replay_enabled"):
            # PII Replay requires full df's. Make a one-off dataset for that call.
            pii_replay = PIIReplay.from_evaluation_dataset(
                EvaluationDataset.from_dataframes(
                    reference=reference,
                    output=output,
                    test=test,
                    column_statistics=column_statistics,
                    rows=MultimodalReport._get_config_value("sqs_rows", DEFAULT_RECORD_COUNT, config),
                    cols=MultimodalReport._get_config_value("sqs_columns", DEFAULT_SQS_REPORT_COLUMNS, config),
                    mandatory_columns=MultimodalReport._get_config_value("mandatory_columns", [], config),
                    enable_sampling=False,
                ),
                config=config,
            )
        # drop columns with assigned type "text" from pii_replay.
        # TODO: This will be removed once text entities is added to the PII Replay section of the evaluation report.
        pii_replay.pii_replay_data = [
            datum for datum in pii_replay.pii_replay_data if datum.column_assigned_type != "text"
        ]
        components.append(pii_replay)

        dataset_statistics = DatasetStatistics.from_evaluation_dataset(evaluation_dataset)

        # column_distribution = ColumnDistribution(score=EvaluationScore())
        column_distribution = ColumnDistribution.from_evaluation_dataset(evaluation_dataset)
        correlation = Correlation(score=EvaluationScore())
        deep_structure = DeepStructure(score=EvaluationScore())
        text_semantic_similarity = TextSemanticSimilarity(score=EvaluationScore())
        text_structure_similarity = TextStructureSimilarity(score=EvaluationScore())
        sqs_score = SQSScore(score=EvaluationScore())
        if config and config.get("enable_synthesis"):
            column_distribution = ColumnDistribution.from_evaluation_dataset(evaluation_dataset)
            correlation = Correlation.from_evaluation_dataset(evaluation_dataset)
            deep_structure = DeepStructure.from_evaluation_dataset(evaluation_dataset)
            text_semantic_similarity = TextSemanticSimilarity.from_evaluation_dataset(evaluation_dataset)
            text_structure_similarity = TextStructureSimilarity.from_evaluation_dataset(evaluation_dataset)
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

        time_series_components: list = []
        ts_params = config.evaluation.time_series if config and getattr(config, "evaluation", None) else None

        if ts_params and ts_params.enabled:
            if ts_params.autocorrelation.enabled:
                time_series_components.append(
                    AutocorrelationSimilarity.from_evaluation_dataset(evaluation_dataset, config)
                )

        if is_timeseries:
            if not any(isinstance(c, AutocorrelationSimilarity) for c in time_series_components):
                acf_component = AutocorrelationSimilarity.from_evaluation_dataset(evaluation_dataset, config)
                if acf_component.score and acf_component.score.score is not None:
                    time_series_components.append(acf_component)

            time_series_components.append(DriftSimilarity.from_evaluation_dataset(evaluation_dataset, config))
            time_series_components.append(HurstSimilarity.from_evaluation_dataset(evaluation_dataset, config))
            time_series_components.append(DTWSimilarity.from_evaluation_dataset(evaluation_dataset, config))
            time_series_components.append(RollingStatsSimilarity.from_evaluation_dataset(evaluation_dataset, config))
            time_series_components.append(SpectralSimilarity.from_evaluation_dataset(evaluation_dataset, config))
            time_series_components.append(ADFStationarity.from_evaluation_dataset(evaluation_dataset, config))

            ts_similarity_score = TimeSeriesSimilarityScore.from_components(time_series_components)
            time_series_components.append(ts_similarity_score)

        components += time_series_components

        report = MultimodalReport(config=config, evaluation_dataset=evaluation_dataset, components=components)
        report.evaluation_dataset = evaluation_dataset
        return report
