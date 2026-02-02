# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from functools import cached_property
from typing import Any

import pandas as pd
from nemo_safe_synthesizer.config.evaluate import (
    DEFAULT_RECORD_COUNT,
    DEFAULT_SQS_REPORT_COLUMNS,
)
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.evaluation.assets.text.multi_modal_tooltips import tooltips
from nemo_safe_synthesizer.evaluation.components.attribute_inference_protection import (
    AttributeInferenceProtection,
)
from nemo_safe_synthesizer.evaluation.components.column_distribution import (
    ColumnDistribution,
    ColumnDistributionPlotRow,
)
from nemo_safe_synthesizer.evaluation.components.correlation import (
    Correlation,
)
from nemo_safe_synthesizer.evaluation.components.data_privacy_score import DataPrivacyScore
from nemo_safe_synthesizer.evaluation.components.dataset_statistics import (
    DatasetStatistics,
)
from nemo_safe_synthesizer.evaluation.components.deep_structure import (
    DeepStructure,
)
from nemo_safe_synthesizer.evaluation.components.membership_inference_protection import (
    MembershipInferenceProtection,
)
from nemo_safe_synthesizer.evaluation.components.pii_replay import PIIReplay
from nemo_safe_synthesizer.evaluation.components.sqs_score import (
    SQSScore,
)
from nemo_safe_synthesizer.evaluation.components.text_semantic_similarity import (
    TextSemanticSimilarity,
)
from nemo_safe_synthesizer.evaluation.components.text_structure_similarity import (
    TextStructureSimilarity,
)
from nemo_safe_synthesizer.evaluation.data_model.evaluation_dataset import (
    EvaluationDataset,
)
from nemo_safe_synthesizer.evaluation.data_model.evaluation_report import (
    EvaluationReport,
)
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import (
    EvaluationScore,
    PrivacyGrade,
)
from nemo_safe_synthesizer.observability import get_logger
from nemo_safe_synthesizer.pii_replacer.transform_result import ColumnStatistics
from pydantic import Field

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
    def from_dataframes(
        reference: pd.DataFrame,
        output: pd.DataFrame,
        test: pd.DataFrame | None = None,
        column_statistics: dict[str, ColumnStatistics] | None = None,
        config: SafeSynthesizerParameters | None = None,
    ) -> MultimodalReport:
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

        report = MultimodalReport(config=config, evaluation_dataset=evaluation_dataset, components=components)
        report.evaluation_dataset = evaluation_dataset
        return report
