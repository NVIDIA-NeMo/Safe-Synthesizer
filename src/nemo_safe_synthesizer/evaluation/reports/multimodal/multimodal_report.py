# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import Field

from ....config.evaluate import (
    DEFAULT_RECORD_COUNT,
    DEFAULT_SQS_REPORT_COLUMNS,
)
from ....config.parameters import SafeSynthesizerParameters
from ....evaluation.components.attribute_inference_protection import AttributeInferenceProtection
from ....evaluation.components.autocorrelation_similarity import AutocorrelationSimilarity
from ....evaluation.components.column_distribution import (
    ColumnDistribution,
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
from ....evaluation.components.time_series_common import TimeSeriesComponent
from ....evaluation.components.window_group_mia import WindowGroupMIA
from ....evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ....evaluation.data_model.evaluation_report import EvaluationReport
from ....evaluation.data_model.evaluation_score import (
    EvaluationScore,
    PrivacyGrade,
)
from ....observability import get_logger
from ....pii_replacer.transform_result import ColumnStatistics

logger = get_logger(__name__)

TIME_SERIES_COMPONENTS: tuple[type[TimeSeriesComponent], ...] = (AutocorrelationSimilarity, WindowGroupMIA)
"""Components run when time-series evaluation is enabled, in report display order."""


class MultimodalReport(EvaluationReport):
    """Multi-modal evaluation report combining quality and privacy components.

    Assembles all evaluation components (SQS sub-metrics for tabular and/or text columns,
    privacy scores, PII replay, dataset statistics) from paired training/synthetic
    dataframes. ``build_report_data`` exports the results, and the HTML report is rendered from that export.

    Use ``from_dataframes`` to construct a fully populated report.
    """

    config: SafeSynthesizerParameters | None = Field(
        default=None, description="Pipeline configuration parameters used for this evaluation."
    )

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
        if (
            config is not None
            and config.get("pii_replay_enabled")
            and (column_statistics or config.get("pii_replay_columns"))
        ):
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
            # Configured columns can run after automatic detection failed; keep the failure in the report.
            if pii_replay_skipped_reason is not None:
                pii_replay = pii_replay.model_copy(update={"skipped_reason": pii_replay_skipped_reason})
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
            components.extend(
                component_cls.from_evaluation_datasets(time_series_datasets, config)
                for component_cls in TIME_SERIES_COMPONENTS
            )

        report = MultimodalReport(config=config, evaluation_datasets=evaluation_datasets, components=components)
        report.evaluation_datasets = evaluation_datasets
        return report
