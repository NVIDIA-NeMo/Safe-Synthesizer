# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public result models returned by the Safe Synthesizer pipeline."""

import logging
from typing import TYPE_CHECKING, Optional

from pydantic import Field

from .base import NSSBaseModel

__all__ = ["SafeSynthesizerTiming", "SafeSynthesizerSummary"]

if TYPE_CHECKING:
    import wandb


class SafeSynthesizerTiming(NSSBaseModel):
    """Wall-clock durations for each pipeline stage."""

    total_time_sec: float | None = Field(default=None, description="Total end-to-end pipeline duration in seconds.")

    pii_replacer_time_sec: float | None = Field(default=None, description="Time spent on PII replacement.")

    training_time_sec: float | None = Field(default=None, description="Time spent on model training.")

    generation_time_sec: float | None = Field(default=None, description="Time spent generating synthetic records.")

    evaluation_time_sec: float | None = Field(default=None, description="Time spent evaluating synthetic data quality.")

    def log_timing(self, logger: logging.Logger) -> None:
        """Emit all timing fields as a structured table via *logger*."""
        logger.info(
            "Safe Synthesizer timing",
            extra={"ctx": {"render_table": True, "tabular_data": self.model_dump(), "title": "Pipeline Timing"}},
        )

    def log_wandb(self, run: Optional["wandb.Run"] = None) -> None:
        """Log timing metrics to an active Weights & Biases run.

        Args:
            run: W&B run instance. No-op when ``None``.
        """
        if run is not None:
            run.log(
                {
                    "total_time_sec": self.total_time_sec,
                    "pii_replacer_time_sec": self.pii_replacer_time_sec,
                    "training_time_sec": self.training_time_sec,
                    "generation_time_sec": self.generation_time_sec,
                    "evaluation_time_sec": self.evaluation_time_sec if self.evaluation_time_sec else 0,
                }
            )


class SafeSynthesizerSummary(NSSBaseModel):
    """Aggregated quality, privacy, and record-count metrics for a pipeline run."""

    synthetic_data_quality_score: float | None = Field(
        default=None,
        description="Weighted composite of the five sub-scores below (SQS). Higher is better (0--10 scale).",
    )

    column_correlation_stability_score: float | None = Field(
        default=None,
        description="How closely pairwise column correlations in synthetic data match the original for numeric and categorical columns.",
    )

    deep_structure_stability_score: float | None = Field(
        default=None,
        description="PCA-based comparison of multivariate structure between real and synthetic data for numeric and categorical columns.",
    )

    column_distribution_stability_score: float | None = Field(
        default=None,
        description="Per-column Jensen-Shannon distance between training and synthetic distributions averaged across all numeric and categorical columns.",
    )

    text_semantic_similarity_score: float | None = Field(
        default=None,
        description="Embedding-based semantic closeness between real and synthetic free-text columns.",
    )

    text_structure_similarity_score: float | None = Field(
        default=None,
        description="Jensen-Shannon divergence over sentence count, words-per-sentence, and characters-per-word distributions between real and synthetic free-text columns.",
    )

    data_privacy_score: float | None = Field(default=None, description="Composite of MIA and AIA protection scores.")

    membership_inference_protection_score: float | None = Field(
        default=None,
        description="Resistance to attacks that try to determine whether a record was in the training set.",
    )

    attribute_inference_protection_score: float | None = Field(
        default=None,
        description="Resistance to attacks that try to infer sensitive attributes from quasi-identifiers.",
    )

    num_valid_records: int | None = Field(
        default=None, description="Count of synthetic records that passed schema and format validation."
    )

    num_invalid_records: int | None = Field(
        default=None, description="Count of synthetic records filtered out during validation."
    )

    num_prompts: int | None = Field(default=None, description="Total LLM generation prompts issued.")

    valid_record_fraction: float | None = Field(
        default=None,
        description="Ratio of valid records: ``num_valid_records / (num_valid_records + num_invalid_records)``.",
    )

    timing: SafeSynthesizerTiming = Field(description="Per-stage wall-clock durations.")

    def log_summary(self, logger: logging.Logger) -> None:
        """Emit all summary metrics as a structured table via ``logger``."""
        logger.info(
            "Safe Synthesizer Summary",
            extra={"ctx": {"render_table": True, "tabular_data": self.model_dump(), "title": "Quality Metrics"}},
        )

    def log_wandb(self) -> None:
        """Log all summary and timing metrics to the active W&B run."""
        import wandb

        if wandb.run is not None:
            wandb.log(
                {
                    "gen/generation_time_sec": self.timing.generation_time_sec,
                    "gen/evaluation_time_sec": self.timing.evaluation_time_sec,
                    "eval/total_time_sec": self.timing.total_time_sec,
                    "train/pii_replacer_time_sec": self.timing.pii_replacer_time_sec,
                    "train/training_time_sec": self.timing.training_time_sec,
                    "gen/num_valid_records": self.num_valid_records,
                    "gen/num_invalid_records": self.num_invalid_records,
                    "gen/num_prompts": self.num_prompts,
                    "gen/valid_record_fraction": self.valid_record_fraction,
                    "eval/data_privacy_score": self.data_privacy_score,
                    "eval/membership_inference_protection_score": self.membership_inference_protection_score,
                    "eval/attribute_inference_protection_score": self.attribute_inference_protection_score,
                    "eval/synthetic_data_quality_score": self.synthetic_data_quality_score,
                    "eval/column_correlation_stability_score": self.column_correlation_stability_score,
                    "eval/deep_structure_stability_score": self.deep_structure_stability_score,
                    "eval/column_distribution_stability_score": self.column_distribution_stability_score,
                    "eval/text_semantic_similarity_score": self.text_semantic_similarity_score,
                    "eval/text_structure_similarity_score": self.text_structure_similarity_score,
                    "eval/success": 1
                    if self.data_privacy_score is not None
                    and self.synthetic_data_quality_score is not None
                    and self.synthetic_data_quality_score > 0
                    else 0,
                }
            )
