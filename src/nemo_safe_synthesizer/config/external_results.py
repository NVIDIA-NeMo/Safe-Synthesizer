# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import TYPE_CHECKING, Optional

from .base import NSSBaseModel

__all__ = ["SafeSynthesizerTiming", "SafeSynthesizerSummary"]

if TYPE_CHECKING:
    import wandb


class SafeSynthesizerTiming(NSSBaseModel):
    """
    Output object for Safe Synthesizer
    """

    total_time_sec: float | None = None
    pii_replacer_time_sec: float | None = None
    training_time_sec: float | None = None
    generation_time_sec: float | None = None
    evaluation_time_sec: float | None = None

    def log_timing(self, logger: logging.Logger) -> None:
        logger.info(
            "Safe Synthesizer timing",
            extra={"ctx": {"render_table": True, "tabular_data": self.model_dump(), "title": "Pipeline Timing"}},
        )

    def log_wandb(self, run: Optional["wandb.Run"] = None) -> None:
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
    """
    Output object for Safe Synthesizer
    """

    synthetic_data_quality_score: float | None = None
    column_correlation_stability_score: float | None = None
    deep_structure_stability_score: float | None = None
    column_distribution_stability_score: float | None = None
    text_semantic_similarity_score: float | None = None
    text_structure_similarity_score: float | None = None

    time_series_similarity_score: float | None = None
    autocorrelation_similarity_score: float | None = None
    drift_similarity_score: float | None = None
    hurst_similarity_score: float | None = None
    dtw_similarity_score: float | None = None
    rolling_stats_similarity_score: float | None = None
    spectral_similarity_score: float | None = None
    adf_stationarity_score: float | None = None
    tsc_utility_score: float | None = None

    data_privacy_score: float | None = None
    membership_inference_protection_score: float | None = None
    attribute_inference_protection_score: float | None = None

    num_valid_records: int | None = None
    num_invalid_records: int | None = None
    num_prompts: int | None = None
    valid_record_fraction: float | None = None

    timing: SafeSynthesizerTiming

    def log_summary(self, logger: logging.Logger) -> None:
        logger.info(
            "Safe Synthesizer Summary",
            extra={"ctx": {"render_table": True, "tabular_data": self.model_dump(), "title": "Quality Metrics"}},
        )

    def log_wandb(self) -> None:
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
                    "eval/time_series_similarity_score": self.time_series_similarity_score,
                    "eval/autocorrelation_similarity_score": self.autocorrelation_similarity_score,
                    "eval/drift_similarity_score": self.drift_similarity_score,
                    "eval/hurst_similarity_score": self.hurst_similarity_score,
                    "eval/dtw_similarity_score": self.dtw_similarity_score,
                    "eval/rolling_stats_similarity_score": self.rolling_stats_similarity_score,
                    "eval/spectral_similarity_score": self.spectral_similarity_score,
                    "eval/adf_stationarity_score": self.adf_stationarity_score,
                    "eval/tsc_utility_score": self.tsc_utility_score,
                    "eval/success": 1
                    if self.data_privacy_score is not None
                    and self.synthetic_data_quality_score is not None
                    and self.synthetic_data_quality_score > 0
                    else 0,
                }
            )
