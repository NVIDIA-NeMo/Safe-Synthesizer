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
from ....evaluation.components.membership_inference_protection import (
    MembershipInferenceProtection,
)
from ....evaluation.components.pii_replay import PIIReplay
from ....evaluation.components.sqs_score import SQSScore
from ....evaluation.components.text_semantic_similarity import TextSemanticSimilarity
from ....evaluation.components.text_structure_similarity import TextStructureSimilarity
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
    """Multi-modal evaluation report combining quality and privacy components.

    Assembles all evaluation components (SQS sub-metrics for tabular and/or text columns,
    privacy scores, PII replay, dataset statistics) from paired reference/output
    dataframes and renders them into an HTML report via Jinja2 templates.

    Use ``from_dataframes`` to construct a fully populated report.
    """

    config: SafeSynthesizerParameters | None = Field(
        default=None, description="Pipeline configuration parameters used for this evaluation."
    )

    @cached_property
    def jinja_context(self):
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

            ctx["with_transform"] = False
            if (
                self.evaluation_dataset is not None
                and self.evaluation_dataset.column_statistics is not None
                and len(self.evaluation_dataset.column_statistics) > 0
            ):
                ctx["with_transform"] = True

            ctx["dp_enabled"] = self.config and self.config.get("dp_enabled")
            if ctx["dp_enabled"]:
                ctx["delta"] = self.config.get("delta")  # ty: ignore[unresolved-attribute]
                ctx["epsilon"] = self.config.get("epsilon")  # ty: ignore[unresolved-attribute]

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
        """Return a config parameter value, falling back to ``default``."""
        if config and config.get(param):
            return config.get(param)
        return default

    @staticmethod
    def from_dataframes(
        reference: pd.DataFrame,
        output: pd.DataFrame,
        test: pd.DataFrame | None = None,
        column_statistics: dict[str, ColumnStatistics] | None = None,
        config: SafeSynthesizerParameters | None = None,
    ) -> MultimodalReport:
        """Build a complete multi-modal evaluation report from dataframes.

        Constructs an ``EvaluationDataset``, runs all enabled evaluation
        components (quality and privacy), and assembles them into a report.

        Args:
            reference: Training (reference) dataframe.
            output: Synthetic (output) dataframe.
            test: Optional holdout dataframe for privacy metrics.
            column_statistics: Per-column PII entity metadata.
            config: Pipeline configuration controlling which metrics are enabled.

        Returns:
            A fully populated ``MultimodalReport`` ready for rendering.
        """
        evaluation_dataset = EvaluationDataset.from_dataframes(
            reference=reference,
            output=output,
            test=test,
            column_statistics=column_statistics,
            rows=MultimodalReport._get_config_value("sqs_rows", DEFAULT_RECORD_COUNT, config),
            cols=MultimodalReport._get_config_value("sqs_columns", DEFAULT_SQS_REPORT_COLUMNS, config),
            mandatory_columns=MultimodalReport._get_config_value("mandatory_columns", [], config),
        )

        components = []

        attribute_inference_protection = AttributeInferenceProtection(
            score=EvaluationScore(grade=PrivacyGrade.UNAVAILABLE)
        )
        if config and config.get("aia_enabled"):
            attribute_inference_protection = AttributeInferenceProtection.from_evaluation_dataset(
                evaluation_dataset, config
            )
        components.append(attribute_inference_protection)

        membership_inference_protection = MembershipInferenceProtection(
            score=EvaluationScore(grade=PrivacyGrade.UNAVAILABLE)
        )
        if config and config.get("mia_enabled"):
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

        report = MultimodalReport(config=config, evaluation_dataset=evaluation_dataset, components=components)
        report.evaluation_dataset = evaluation_dataset
        return report
