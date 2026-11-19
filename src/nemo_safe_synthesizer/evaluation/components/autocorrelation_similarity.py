# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autocorrelation-based fidelity evaluation for numeric time-series columns.

The metric compares lagged self-correlation in training and synthetic
sequences. It evaluates each shared group and numeric value column separately,
skips profiles with unusable training data, treats constant synthetic output
as a fidelity failure, and averages the remaining similarities into a 0--10
score.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field

from ...config.evaluate import AutocorrelationSimilarityParameters
from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from .autocorrelation_similarity_figures import (
    AUTOCORRELATION_FIGURE_HEIGHT,
    AUTOCORRELATION_SUMMARY_TRACES_PER_COLUMN,
    generate_autocorrelation_lag_error_figure,
    generate_autocorrelation_summary_figure,
)
from .time_series_common import (
    FigureSpec,
    PairResult,
    TimeSeriesComponent,
    evaluate_pairs,
    is_effectively_constant,
    prepare_values,
    profile_details,
)
from .time_series_common_figures import PAIR_SCORE_MAX_COLUMNS, generate_pair_score_figure, order_columns_by_score

_MIN_VALID_PAIRS = 3


class AutocorrelationSimilarity(TimeSeriesComponent):
    """Measure fidelity of lagged self-dependence in numeric time-series data.

    The component compares training and synthetic autocorrelation profiles
    independently for each shared group and numeric value column. Every usable
    comparison contributes equally to the final score. Skipped comparisons and
    drill-down summaries remain available in ``details`` so callers can locate
    columns or groups responsible for a mismatch.

    Attributes:
        name: Display name used in serialized evaluation results.
        details: Per-group/column profiles, skipped comparisons, and summaries.
    """

    card_id: ClassVar[str] = "autocorrelation-similarity"
    description: ClassVar[str] = "Measures whether numeric columns change over time like the training data"
    tooltip_key: ClassVar[str] = "autocorrelation_similarity_info"
    template: ClassVar[str] = "jinja/components/autocorrelation_similarity.j2"
    score_colored_traces: ClassVar[bool] = True

    name: str = Field(
        default="Autocorrelation Similarity",
        description="Display name used in serialized evaluation results.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-group/column autocorrelation profiles, skipped comparisons, and summaries.",
    )

    @cached_property
    def jinja_context(self) -> dict[str, Any]:
        """Return score, details, card attributes, and chart figures for report rendering."""
        context = super().jinja_context
        profiles = self.details.get("profiles", [])
        figures: list[FigureSpec] = []
        summary_columns: list[str] = []
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
                    generate_pair_score_figure(profiles, self.score.score, height=AUTOCORRELATION_FIGURE_HEIGHT),
                    False,
                    "autocorrelation_pair_scores_info",
                ),
            ]
            summary_columns = order_columns_by_score(profiles)
        context["figures"] = self.figure_rows(figures)
        context["summary_columns"] = summary_columns
        context["summary_traces_per_column"] = AUTOCORRELATION_SUMMARY_TRACES_PER_COLUMN
        context["pair_score_column_count"] = min(len(summary_columns), PAIR_SCORE_MAX_COLUMNS)
        context["evaluated_profile_count"] = len(profiles)
        context["evaluated_group_count"] = len({str(item["group"]) for item in profiles})
        context["evaluated_column_count"] = len({item["column"] for item in profiles})
        context["constant_synthetic_profile_count"] = sum(1 for item in profiles if item["evaluated_lags"] == 0)
        return context

    @staticmethod
    def from_evaluation_datasets(
        evaluation_datasets: EvaluationDatasets,
        config: SafeSynthesizerParameters | None = None,
    ) -> AutocorrelationSimilarity:
        """Evaluate autocorrelation fidelity for paired time-series datasets.

        Report orchestration controls whether the optional metric runs.
        Calling this component directly always computes it with the supplied
        configuration or isolated defaults. Evaluation failures are returned
        in ``score.notes`` instead of aborting the full evaluation pipeline.

        Args:
            evaluation_datasets: Training and synthetic datasets to compare.
            config: Optional pipeline and metric configuration.

        Returns:
            A component containing the score, diagnostic details, and notes.
        """
        cfg = AutocorrelationSimilarity._resolve_config(config)
        score, details = evaluate_pairs(
            evaluation_datasets,
            config,
            cfg,
            lambda training, synthetic: AutocorrelationSimilarity._profile_score(training, synthetic, cfg),
        )
        return AutocorrelationSimilarity(score=score, details=details)

    @staticmethod
    def _resolve_config(config: SafeSynthesizerParameters | None) -> AutocorrelationSimilarityParameters:
        """Return configured metric parameters or an isolated default model."""
        if config is None:
            return AutocorrelationSimilarityParameters()
        return config.evaluation.time_series.autocorrelation

    @staticmethod
    def _profile_score(
        training: pd.Series,
        synthetic: pd.Series,
        cfg: AutocorrelationSimilarityParameters,
    ) -> PairResult:
        """Compare one training and synthetic autocorrelation profile.

        For each lag, Pearson correlation is computed over positions whose two
        endpoints are finite. Lags are capped at half the shorter finite
        sequence. The error is scored by ``_noise_aware_error``, so differences
        explained by sampling noise do not lower the score on average and losing
        all lag structure scores near zero.

        Args:
            training: Ordered training values for one group and column.
            synthetic: Ordered synthetic values for the same group and column.
            cfg: Parameters controlling minimum length and maximum lag.

        Returns:
            Profile score details and ``None`` on success, or ``None`` and a
            human-readable skip reason when the comparison is unusable.
        """
        training_values = prepare_values(training)
        synthetic_values = prepare_values(synthetic)
        training_count = int(np.sum(np.isfinite(training_values)))
        synthetic_count = int(np.sum(np.isfinite(synthetic_values)))
        n = min(training_count, synthetic_count)
        if n < cfg.min_points:
            return None, f"Each sequence needs at least {cfg.min_points} finite observations."
        if is_effectively_constant(training_values):
            return None, "The training sequence is constant or indistinguishable from constant at its numeric scale."

        # The lag cap bounds work and avoids profiles dominated by the short
        # tail of either sequence. Per-lag support is validated independently.
        effective_max_lag = min(cfg.max_lag, (n - 1) // 2)
        if effective_max_lag < 1:
            return None, "The sequences are too short to evaluate a positive lag."
        training_acf, training_support = AutocorrelationSimilarity._acf_profile(training_values, effective_max_lag)
        if not np.any(np.isfinite(training_acf)):
            return None, f"No training lags have at least {_MIN_VALID_PAIRS} usable endpoint pairs."

        lags = list(range(1, effective_max_lag + 1))
        synthetic_acf, synthetic_support = AutocorrelationSimilarity._acf_profile(synthetic_values, effective_max_lag)
        if is_effectively_constant(synthetic_values):
            return {
                "lags": lags,
                "effective_max_lag": effective_max_lag,
                "evaluated_lags": 0,
                "error": 1.0,
                "similarity": 0.0,
                "reason": "The synthetic sequence is constant or indistinguishable from constant at its numeric scale.",
                "training_acf": profile_details(training_acf),
                "synthetic_acf": [None] * effective_max_lag,
                "training_pair_support": training_support.tolist(),
                "synthetic_pair_support": synthetic_support.tolist(),
            }, None

        shared_valid_lags = np.isfinite(training_acf) & np.isfinite(synthetic_acf)
        if not np.any(shared_valid_lags):
            return None, f"No lags have at least {_MIN_VALID_PAIRS} usable endpoint pairs in both sequences."
        noise = np.hypot(
            AutocorrelationSimilarity._acf_standard_error(training_acf, training_support),
            AutocorrelationSimilarity._acf_standard_error(synthetic_acf, synthetic_support),
        )
        error = AutocorrelationSimilarity._noise_aware_error(
            training_acf[shared_valid_lags],
            synthetic_acf[shared_valid_lags],
            noise[shared_valid_lags],
        )
        return {
            "lags": lags,
            "effective_max_lag": effective_max_lag,
            "evaluated_lags": int(np.sum(shared_valid_lags)),
            "error": error,
            "similarity": 1.0 - error,
            "training_acf": profile_details(training_acf),
            "synthetic_acf": profile_details(synthetic_acf),
            "training_pair_support": training_support.tolist(),
            "synthetic_pair_support": synthetic_support.tolist(),
        }, None

    @staticmethod
    def _acf_standard_error(acf: NDArray[np.float64], support: NDArray[np.int64]) -> NDArray[np.float64]:
        """Return Bartlett's large-sample standard error for each lag of an ACF.

        The variance at lag ``k`` is ``(1 + 2 * sum(r_j ** 2 for j < k)) / n_k``,
        where ``n_k`` is the number of endpoint pairs behind that lag, so lags
        thinned by gaps get a wider allowance. Each measured ``r_j ** 2`` carries
        about ``1 / n_j`` of its own noise, which is removed before summing so
        short series do not overstate their noise. Undefined lags contribute
        nothing to later lags.
        """
        support = np.maximum(support, 1)
        signal = np.maximum(np.nan_to_num(acf) ** 2 - 1.0 / support, 0.0)
        earlier = np.concatenate([[0.0], np.cumsum(signal[:-1])])
        return np.sqrt((1.0 + 2.0 * earlier) / support)

    @staticmethod
    def _noise_aware_error(
        training_acf: NDArray[np.float64],
        synthetic_acf: NDArray[np.float64],
        noise: NDArray[np.float64],
    ) -> float:
        """Return the share of lag structure that differs beyond sampling noise.

        Sampling noise adds its variance to every squared difference, so the
        expected noise variance is subtracted from each squared difference and
        the results are pooled across lags before taking the square root.
        Pooling first lets noise cancel across lags instead of rounding every
        small gap up to a penalty or down to zero, which keeps the expected score
        from depending on sequence length. The remaining difference is divided
        by the larger of the two profile magnitudes, floored at the noise, so
        losing all structure and inventing structure are penalized alike. Lags
        are weighted by inverse noise variance so precisely estimated lags count
        more.

        Args:
            training_acf: Training correlations at lags valid in both profiles.
            synthetic_acf: Synthetic correlations at the same lags.
            noise: Expected sampling noise of the difference at each lag.

        Returns:
            Error in ``[0, 1]``.
        """
        weight = 1.0 / noise**2
        excess = np.sum(weight * ((training_acf - synthetic_acf) ** 2 - noise**2))
        magnitude = np.maximum(np.maximum(np.abs(training_acf), np.abs(synthetic_acf)), noise)
        return float(np.clip(np.sqrt(max(excess, 0.0) / np.sum(weight * magnitude**2)), 0.0, 1.0))

    @staticmethod
    def _acf_profile(
        values: NDArray[np.float64],
        max_lag: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Compute missing-aware per-lag Pearson correlations and support.

        Missing positions are preserved. At each lag, only pairs with two
        finite endpoints contribute, and each endpoint vector is centered and
        scaled independently as required by Pearson correlation.

        Args:
            values: Nonconstant values in temporal order, with gaps represented
                as ``NaN``.
            max_lag: Largest positive lag to include.

        Returns:
            Correlations and valid endpoint-pair counts for each positive lag.
        """
        finite = np.isfinite(values)
        acf = np.full(max_lag, np.nan)
        support = np.zeros(max_lag, dtype=np.int64)
        for lag in range(1, max_lag + 1):
            valid_pairs = finite[:-lag] & finite[lag:]
            pair_count = int(np.sum(valid_pairs))
            support[lag - 1] = pair_count
            if pair_count < _MIN_VALID_PAIRS:
                continue
            earlier = values[:-lag][valid_pairs]
            later = values[lag:][valid_pairs]
            if is_effectively_constant(earlier):
                continue
            if is_effectively_constant(later):
                continue
            correlation = float(np.corrcoef(earlier, later)[0, 1])
            acf[lag - 1] = np.clip(correlation, -1.0, 1.0)
        return acf, support
