# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rolling statistics fidelity evaluation for numeric time-series columns.

The metric compares how the local level (rolling average) and local spread
(rolling standard deviation) change from the beginning to the end of training
and synthetic sequences. It evaluates each shared group and numeric value
column separately, skips pairs with unusable training data, treats constant
synthetic output as a fidelity failure, and averages the remaining similarities
into a 0--10 score.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from pydantic import Field
from scipy.signal import lfilter

from ...config.evaluate import RollingStatsSimilarityParameters
from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...observability import get_logger
from .rolling_stats_similarity_figures import (
    ROLLING_STATS_PAIR_SCORE_MAX_COLUMNS,
    ROLLING_STATS_SEGMENTS,
    ROLLING_STATS_SUMMARY_TRACES_PER_COLUMN,
    generate_rolling_stats_pair_score_figure,
    generate_rolling_stats_profile_figure,
    generate_rolling_stats_segment_difference_figure,
    order_rolling_stats_columns,
)
from .time_series_common import (
    FigureSpec,
    TimeSeriesComponent,
    excluded_columns_note,
    group_frame,
    resolve_group_column,
    select_value_columns,
    shared_groups,
    summaries,
)

logger = get_logger(__name__)

_MIN_WINDOW = 3
_GROUP_SELECTION_SEED = 2112
_CONSTANT_TOLERANCE_FACTOR = 32.0
# Floor for segment noise, in squared training standard deviations, so noise-free
# deterministic sequences still compare cleanly.
_NOISE_FLOOR = 1e-12
_MAX_LAG_ONE_CORRELATION = 0.99
_MIN_LAG_ONE_CORRELATION = -0.95
_MIN_VARIANCE_CORRECTION = 0.5
# Gaps can leave few adjacent pairs, so the lag-one correlation may come from a
# lag up to this one with pairs for at least this share of the finite values.
_MAX_CORRELATION_LAG = 4
_MIN_PAIR_SHARE = 0.25
_CONSTANT_REASON = "The synthetic sequence is constant or indistinguishable from constant at its numeric scale."
_MOSTLY_MISSING_REASON = "The synthetic sequence is mostly missing, so its local average and spread cannot be measured."


@dataclass(frozen=True)
class _SegmentProfile:
    """Segment averages of one standardized sequence and their chance variances."""

    level: NDArray[np.float64]
    spread: NDArray[np.float64]
    level_noise: float
    spread_noise: float
    mean_spread: float


class RollingStatsSimilarity(TimeSeriesComponent):
    """Measure fidelity of local level and spread over time in numeric time-series data.

    The component compares training and synthetic rolling averages and spreads
    over ten aligned segments, from the beginning to the end of each sequence,
    independently for each shared group and numeric value column. Every usable
    comparison contributes equally to the final score. Skipped comparisons and
    drill-down summaries remain available in ``details`` so callers can locate
    columns or groups responsible for a mismatch.

    Attributes:
        name: Display name used in serialized evaluation results.
        details: Per-group/column segment profiles, skipped comparisons, and summaries.
    """

    card_id: ClassVar[str] = "rolling-stats-similarity"
    description: ClassVar[str] = (
        "Measures whether the local average and spread of numeric columns change over time like the training data"
    )
    tooltip_key: ClassVar[str] = "rolling_stats_similarity_info"
    template: ClassVar[str] = "jinja/components/rolling_stats_similarity.j2"

    name: str = Field(
        default="Rolling Stats Similarity",
        description="Display name used in serialized evaluation results.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-group/column segment profiles, skipped comparisons, and summaries.",
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
                    "Average and spread",
                    generate_rolling_stats_profile_figure(profiles),
                    True,
                    "rolling_stats_profile_info",
                ),
                (
                    "Difference by segment",
                    generate_rolling_stats_segment_difference_figure(profiles),
                    False,
                    "rolling_stats_difference_info",
                ),
                (
                    "Pair scores",
                    generate_rolling_stats_pair_score_figure(profiles, self.score.score),
                    False,
                    "rolling_stats_pair_scores_info",
                ),
            ]
            summary_columns = order_rolling_stats_columns(profiles)
        context["figures"] = self.figure_rows(figures)
        context["summary_columns"] = summary_columns
        context["summary_traces_per_column"] = ROLLING_STATS_SUMMARY_TRACES_PER_COLUMN
        context["pair_score_column_count"] = min(len(summary_columns), ROLLING_STATS_PAIR_SCORE_MAX_COLUMNS)
        context["evaluated_profile_count"] = len(profiles)
        context["evaluated_group_count"] = len({str(item["group"]) for item in profiles})
        context["evaluated_column_count"] = len({item["column"] for item in profiles})
        context["constant_synthetic_profile_count"] = sum(
            1 for item in profiles if all(value is None for value in item["synthetic_level"])
        )
        return context

    @staticmethod
    def from_evaluation_datasets(
        evaluation_datasets: EvaluationDatasets,
        config: SafeSynthesizerParameters | None = None,
    ) -> RollingStatsSimilarity:
        """Evaluate rolling statistics fidelity for paired time-series datasets.

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
        cfg = RollingStatsSimilarity._resolve_config(config)
        # Optional metrics must fail independently so one diagnostic cannot
        # prevent the rest of the evaluation report from being produced.
        try:
            return RollingStatsSimilarity._evaluate(evaluation_datasets, cfg, config)
        except Exception as exc:
            logger.exception("Failed to compute Rolling Stats Similarity.")
            return RollingStatsSimilarity(score=EvaluationScore(notes=str(exc)))

    @staticmethod
    def _resolve_config(config: SafeSynthesizerParameters | None) -> RollingStatsSimilarityParameters:
        """Return configured metric parameters or an isolated default model."""
        if config is None:
            return RollingStatsSimilarityParameters()
        return config.evaluation.time_series.rolling_stats

    @staticmethod
    def _evaluate(
        datasets: EvaluationDatasets,
        cfg: RollingStatsSimilarityParameters,
        config: SafeSynthesizerParameters | None,
    ) -> RollingStatsSimilarity:
        """Compute per-group/column segment profiles and aggregate the component score.

        Args:
            datasets: Training and synthetic datasets to compare.
            cfg: Resolved rolling statistics metric parameters.
            config: Optional top-level configuration used for inherited columns.

        Returns:
            A scored component, or an unavailable component with explanatory
            notes when no usable comparison remains.
        """
        timestamp_column = config.time_series.timestamp_column if config is not None else None
        group_column, group_error = resolve_group_column(datasets, config)
        if group_error is not None:
            return RollingStatsSimilarity(score=EvaluationScore(notes=group_error))

        columns, column_error, excluded_columns = select_value_columns(
            datasets, cfg.value_columns, timestamp_column, group_column, "rolling stats"
        )
        if column_error is not None:
            return RollingStatsSimilarity(score=EvaluationScore(notes=column_error))
        if not columns:
            return RollingStatsSimilarity._unavailable("No shared numeric value columns.", excluded_columns)

        groups, missing_training, missing_synthetic, shared_group_count = shared_groups(
            datasets.training,
            datasets.synthetic,
            group_column,
            cfg.max_groups,
            _GROUP_SELECTION_SEED,
        )
        if not groups:
            return RollingStatsSimilarity._unavailable("No shared groups to evaluate.", excluded_columns)
        omitted_groups = shared_group_count - len(groups)
        group_selection = {
            "shared_groups": shared_group_count,
            "evaluated_groups": len(groups),
            "omitted_shared_groups": omitted_groups,
            "policy": "seeded_random_sample" if omitted_groups else "all_shared_groups",
        }

        profiles: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for group_value in groups:
            training_group = group_frame(datasets.training, group_column, group_value, timestamp_column)
            synthetic_group = group_frame(datasets.synthetic, group_column, group_value, timestamp_column)
            group_label = None if group_column is None else str(group_value)
            for column in columns:
                result, reason = RollingStatsSimilarity._pair_score(
                    training_group[column], synthetic_group[column], cfg
                )
                RollingStatsSimilarity._record_pair_result(profiles, skipped, group_label, column, result, reason)

        counts = {
            "groups": len(groups),
            "shared_groups": shared_group_count,
            "omitted_shared_groups": omitted_groups,
            "columns": len(columns),
            "evaluated_profiles": len(profiles),
            "skipped": len(skipped),
        }
        if not profiles:
            unavailable_note = "No usable group/column rolling statistics profiles."
            if omitted_groups:
                unavailable_note += (
                    f" Evaluated {len(groups)} of {shared_group_count} shared groups using a reproducible sample."
                )
            return RollingStatsSimilarity._unavailable(
                unavailable_note,
                excluded_columns,
                {
                    "counts": counts,
                    "group_selection": group_selection,
                    "skipped": skipped,
                    "groups_only_in_training": missing_training,
                    "groups_only_in_synthetic": missing_synthetic,
                },
            )

        similarity = float(np.mean([item["similarity"] for item in profiles]))
        score = EvaluationScore.finalize_grade(raw_score=similarity, score=10.0 * similarity)
        notes: list[str] = []
        if skipped:
            notes.append(f"Skipped {len(skipped)} unusable group/column comparisons.")
        if omitted_groups:
            notes.append(f"Evaluated {len(groups)} of {shared_group_count} shared groups using a reproducible sample.")
        excluded_note = excluded_columns_note(excluded_columns)
        if excluded_note is not None:
            notes.append(excluded_note)
        if notes:
            score.notes = " ".join(notes)

        details = {
            "evaluation_mode": "per_group" if group_column else "global",
            "timestamp_column": timestamp_column,
            "group_column": group_column,
            "window_size": cfg.window_size,
            "min_points": cfg.min_points,
            "segments": ROLLING_STATS_SEGMENTS,
            "counts": counts,
            "group_selection": group_selection,
            "per_group": summaries(profiles, "group"),
            "per_column": summaries(profiles, "column"),
            "profiles": profiles,
            "skipped": skipped,
            "excluded_columns": excluded_columns,
            "groups_only_in_training": missing_training,
            "groups_only_in_synthetic": missing_synthetic,
        }
        return RollingStatsSimilarity(score=score, details=details)

    @staticmethod
    def _unavailable(
        note: str,
        excluded_columns: list[str],
        details: dict[str, Any] | None = None,
    ) -> RollingStatsSimilarity:
        """Return an unscored component whose notes and details still name left-out columns."""
        excluded_note = excluded_columns_note(excluded_columns)
        return RollingStatsSimilarity(
            score=EvaluationScore(notes=note if excluded_note is None else f"{note} {excluded_note}"),
            details={**(details or {}), "excluded_columns": excluded_columns},
        )

    @staticmethod
    def _record_pair_result(
        profiles: list[dict[str, Any]],
        skipped: list[dict[str, Any]],
        group_label: str | None,
        column: str,
        result: dict[str, Any] | None,
        reason: str | None,
    ) -> None:
        """Record one successful or skipped group-and-column comparison."""
        if result is None:
            skipped.append({"group": group_label, "column": column, "reason": reason})
            return
        profiles.append({"group": group_label, "column": column, **result})

    @staticmethod
    def _pair_score(
        training: pd.Series,
        synthetic: pd.Series,
        cfg: RollingStatsSimilarityParameters,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Compare one training and synthetic sequence segment by segment.

        Both sequences are standardized by the training mean and standard
        deviation, so differences in level and scale count. Rolling averages
        and spreads are averaged over ten aligned segments, beginning to end,
        and scored by ``_noise_aware_error`` so differences explained by chance
        do not lower the score on average.

        Args:
            training: Ordered training values for one group and column.
            synthetic: Ordered synthetic values for the same group and column.
            cfg: Parameters controlling minimum length and window size.

        Problems on the training side skip the pair, because there is nothing
        to compare against. Problems on the synthetic side, a constant or
        mostly missing sequence, score 0, because they are fidelity failures.

        Returns:
            Profile details and ``None`` on success, or ``None`` and a
            human-readable skip reason when the comparison is unusable.
        """
        training_values = RollingStatsSimilarity._prepare_values(training)
        synthetic_values = RollingStatsSimilarity._prepare_values(synthetic)
        training_count = int(np.sum(np.isfinite(training_values)))
        synthetic_count = int(np.sum(np.isfinite(synthetic_values)))
        if training_count < cfg.min_points:
            return None, f"The training sequence needs at least {cfg.min_points} finite observations."
        if RollingStatsSimilarity._is_effectively_constant(training_values):
            return None, "The training sequence is constant or indistinguishable from constant at its numeric scale."

        synthetic_missing = synthetic_count < cfg.min_points
        n = training_count if synthetic_missing else min(training_count, synthetic_count)
        window = min(cfg.window_size, max(_MIN_WINDOW, n // 2))
        segments = min(ROLLING_STATS_SEGMENTS, len(training_values) - window + 1)
        if not synthetic_missing:
            segments = min(segments, len(synthetic_values) - window + 1)
        center = float(np.nanmean(training_values))
        scale = float(np.nanstd(training_values))
        training_profile = RollingStatsSimilarity._segment_profile((training_values - center) / scale, window, segments)
        if not RollingStatsSimilarity._has_finite_segment(training_profile):
            return None, "No segment of the training sequence has enough finite values to measure."
        base = {"window": window, "segments": segments}
        if synthetic_missing:
            return RollingStatsSimilarity._synthetic_failure(base, training_profile, _MOSTLY_MISSING_REASON), None
        if RollingStatsSimilarity._is_effectively_constant(synthetic_values):
            return RollingStatsSimilarity._synthetic_failure(base, training_profile, _CONSTANT_REASON), None

        synthetic_profile = RollingStatsSimilarity._segment_profile(
            (synthetic_values - center) / scale, window, segments
        )
        if not RollingStatsSimilarity._has_finite_segment(synthetic_profile):
            return RollingStatsSimilarity._synthetic_failure(base, training_profile, _MOSTLY_MISSING_REASON), None
        if not RollingStatsSimilarity._shares_a_segment(training_profile, synthetic_profile):
            return None, "No segment of the sequence has values in both training and synthetic data."
        error = RollingStatsSimilarity._noise_aware_error(training_profile, synthetic_profile)
        return {
            **base,
            "error": error,
            "similarity": 1.0 - error,
            "level_noise": training_profile.level_noise + synthetic_profile.level_noise,
            "spread_noise": training_profile.spread_noise + synthetic_profile.spread_noise,
            "spread_reference": training_profile.mean_spread,
            "training_level": RollingStatsSimilarity._profile_details(training_profile.level),
            "synthetic_level": RollingStatsSimilarity._profile_details(synthetic_profile.level),
            "training_spread": RollingStatsSimilarity._profile_details(training_profile.spread),
            "synthetic_spread": RollingStatsSimilarity._profile_details(synthetic_profile.spread),
        }, None

    @staticmethod
    def _segment_profile(values: NDArray[np.float64], window: int, segments: int) -> _SegmentProfile:
        """Average rolling statistics over aligned segments and estimate their chance variation.

        The local level is the rolling average over complete windows. The local
        spread is the rolling root mean square of each value's deviation from
        the centered rolling average around it. Both are averaged over
        ``segments`` equal parts of the sequence by relative position.

        Args:
            values: Standardized values in temporal order, with gaps as ``NaN``.
            window: Rolling window length.
            segments: Number of aligned segments.

        Returns:
            Segment averages with their chance variances.
        """
        squares = RollingStatsSimilarity._local_squares(values, window)
        level = RollingStatsSimilarity._rolling_mean(values, window)
        spread = np.sqrt(RollingStatsSimilarity._rolling_mean(squares, window))
        level_noise, spread_noise = RollingStatsSimilarity._segment_noise(values, squares, window, segments)
        return _SegmentProfile(
            level=RollingStatsSimilarity._segment_means(level, segments),
            spread=RollingStatsSimilarity._segment_means(spread, segments),
            level_noise=level_noise,
            spread_noise=spread_noise,
            mean_spread=float(np.nanmean(spread)),
        )

    @staticmethod
    def _rolling_mean(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
        """Average complete windows, leaving windows with under half their values finite as ``NaN``."""
        windows = sliding_window_view(values, window)
        finite = np.isfinite(windows)
        count = np.sum(finite, axis=1)
        total = np.sum(np.where(finite, windows, 0.0), axis=1)
        enough = count >= RollingStatsSimilarity._min_window_count(window)
        return np.where(enough, total / np.maximum(count, 1), np.nan)

    @staticmethod
    def _min_window_count(window: int) -> int:
        """Return the number of finite values a rolling window needs to count."""
        return max(2, (window + 1) // 2)

    @staticmethod
    def _local_squares(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
        """Return each value's squared deviation from the centered rolling average around it.

        Edge windows are truncated rather than padded, missing values stay
        missing, and the result is scaled by ``window / (window - 1)`` like a
        sample variance.
        """
        finite = np.isfinite(values)
        kernel = np.ones(window)
        total = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
        count = np.convolve(finite.astype(float), kernel, mode="same")
        local = total / np.maximum(count, 1.0)
        return np.where(finite, (values - local) ** 2 * window / (window - 1), np.nan)

    @staticmethod
    def _segment_means(profile: NDArray[np.float64], segments: int) -> NDArray[np.float64]:
        """Average a rolling profile over equal segments by relative position."""
        edges = np.linspace(0, len(profile), segments + 1).round().astype(int)
        means = np.full(segments, np.nan)
        for index in range(segments):
            part = profile[edges[index] : edges[index + 1]]
            if np.any(np.isfinite(part)):
                means[index] = float(np.nanmean(part))
        return means

    @staticmethod
    def _segment_noise(
        values: NDArray[np.float64],
        squares: NDArray[np.float64],
        window: int,
        segments: int,
    ) -> tuple[float, float]:
        """Return the chance variance of one segment's average level and average spread.

        Each segment average is a weighted average of the finite values in the
        segment's rolling windows, so its variance follows from those weights
        and the autocovariance of the underlying series. The values are
        linearly detrended, so a steady trend counts as structure and not as
        chance, and fitted with a bias-corrected first-order autoregressive
        model, which gives smooth autocovariances at every lag even for short
        sequences. The spread uses the same algebra on the squared deviations,
        converted with ``Var(s) = Var(s^2) / (4 E[s^2])``. The variances are
        averaged over segments.

        Args:
            values: Standardized values in temporal order, with gaps as ``NaN``.
            squares: Squared deviations from the local rolling average.
            window: Rolling window length.
            segments: Number of aligned segments.

        Returns:
            Chance variances of the segment level and segment spread.
        """
        finite = np.isfinite(values)
        weights = RollingStatsSimilarity._segment_weights(finite, window, segments)
        count = max(int(np.sum(finite)), 1)
        level_variance, level_correlation = RollingStatsSimilarity._ar1_fit(RollingStatsSimilarity._detrend(values))
        mean_square = float(np.nanmean(squares))
        square_variance, square_correlation = RollingStatsSimilarity._ar1_fit(squares - mean_square)
        level_noise = RollingStatsSimilarity._average_variance(level_variance, level_correlation, weights, count)
        square_noise = RollingStatsSimilarity._average_variance(square_variance, square_correlation, weights, count)
        spread_noise = square_noise / (4 * mean_square) if mean_square > 0 else 0.0
        return level_noise, spread_noise

    @staticmethod
    def _segment_weights(finite: NDArray[np.bool_], window: int, segments: int) -> list[NDArray[np.float64]]:
        """Return the weight of each value in each segment average, for segments with a usable window.

        A segment average is the mean of its usable rolling windows, and each
        window is the mean of its finite values. A value's weight is therefore
        the sum, over the usable windows that contain it, of one over the
        window's finite count, divided by the number of usable windows. Gaps
        raise the weight of the values that remain. Each array covers the
        values from the segment's first window to the end of its last window.
        """
        count = np.sum(sliding_window_view(finite, window), axis=1)
        usable = count >= RollingStatsSimilarity._min_window_count(window)
        edges = np.linspace(0, len(count), segments + 1).round().astype(int)
        weights: list[NDArray[np.float64]] = []
        for start, stop in zip(edges[:-1], edges[1:]):
            used = usable[start:stop]
            if not np.any(used):
                continue
            share = np.where(used, 1.0 / np.maximum(count[start:stop], 1), 0.0) / np.sum(used)
            weights.append(np.convolve(share, np.ones(window)) * finite[start : stop + window - 1])
        return weights

    @staticmethod
    def _average_variance(variance: float, correlation: float, weights: list[NDArray[np.float64]], count: int) -> float:
        """Mean variance of weighted sums under AR(1) autocovariances ``variance * correlation^k``.

        For weights ``w``, the variance is ``variance * (sum w_i^2 + 2 sum_i w_i
        sum_{k<i} correlation^(i-k) w_k)``, and the inner sum follows the AR(1)
        recursion, so the cost is linear in the sequence length. Each variance
        is floored at the independent-values variance divided by the sequence
        length, so strongly negative correlations cannot drive it to zero.
        """
        if not weights:
            return 0.0
        totals = []
        for weight in weights:
            earlier = lfilter([0.0, correlation], [1.0, -correlation], weight)
            square = float(np.dot(weight, weight))
            total = variance * (square + 2.0 * float(np.dot(weight, earlier)))
            totals.append(max(total, variance * square / count))
        return float(np.mean(totals))

    @staticmethod
    def _detrend(values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Remove a least-squares straight line from the finite values, keeping gaps as ``NaN``."""
        finite = np.isfinite(values)
        positions = np.arange(len(values), dtype=float)
        slope, intercept = np.polyfit(positions[finite], values[finite], 1)
        return np.where(finite, values - (slope * positions + intercept), np.nan)

    @staticmethod
    def _ar1_fit(centered: NDArray[np.float64]) -> tuple[float, float]:
        """Return the variance and lag-one correlation of a centered series, corrected for short samples.

        Lag-one correlation is biased low by about ``(1 + 3 rho) / n`` (Kendall),
        and the sample variance of a persistent series by a factor ``1 - f / n``
        with ``f = (1 + rho) / (1 - rho)``. Only pairs with two finite
        endpoints contribute to the correlation.
        """
        finite = np.isfinite(centered)
        n = int(np.sum(finite))
        filled = np.where(finite, centered, 0.0)
        variance = float(np.dot(filled, filled)) / n
        if variance <= 0:
            return 0.0, 0.0
        correlation = RollingStatsSimilarity._lag_one_correlation(filled, finite, variance)
        correlation = float(
            np.clip(correlation + (1 + 3 * correlation) / n, _MIN_LAG_ONE_CORRELATION, _MAX_LAG_ONE_CORRELATION)
        )
        persistence = (1 + correlation) / (1 - correlation)
        return variance / max(1 - persistence / n, _MIN_VARIANCE_CORRECTION), correlation

    @staticmethod
    def _lag_one_correlation(filled: NDArray[np.float64], finite: NDArray[np.bool_], variance: float) -> float:
        """Estimate the lag-one correlation of a centered series from pairs of finite values.

        Gaps can leave few adjacent pairs, so the estimate uses the smallest
        lag up to ``_MAX_CORRELATION_LAG`` with pairs for at least
        ``_MIN_PAIR_SHARE`` of the finite values, or the lag with the most
        pairs when none has that many. An AR(1) series has lag-``k``
        correlation ``rho^k``, so the ``k``-th root with its sign kept
        estimates ``rho``. Without gaps this is the lag-one correlation.
        """
        n = int(np.sum(finite))
        lags = range(1, _MAX_CORRELATION_LAG + 1)
        pairs = [int(np.sum(finite[:-lag] & finite[lag:])) for lag in lags]
        enough = [index for index, count in enumerate(pairs) if count >= _MIN_PAIR_SHARE * n]
        index = enough[0] if enough else int(np.argmax(pairs))
        if pairs[index] == 0:
            return 0.0
        lag = lags[index]
        correlation = float(np.dot(filled[:-lag], filled[lag:])) / pairs[index] / variance
        return float(np.sign(correlation) * abs(correlation) ** (1 / lag))

    @staticmethod
    def _noise_aware_error(training: _SegmentProfile, synthetic: _SegmentProfile) -> float:
        """Return the share of segment structure that differs beyond chance.

        Chance adds its variance to every squared segment difference, so the
        expected chance variance is subtracted from each squared difference and
        the results are pooled across segments and both statistics before
        taking the square root, as in Autocorrelation Similarity. The remaining
        difference is divided by the larger of the two segment structures,
        floored at the noise, so losing structure and inventing it are
        penalized alike. Level structure is measured from the training mean and
        spread structure from the training's average spread. Each statistic is
        weighted by its inverse noise variance.

        Args:
            training: Training segment averages and chance variances.
            synthetic: Synthetic segment averages and chance variances.

        Returns:
            Error in ``[0, 1]``.
        """
        excess = 0.0
        magnitude = 0.0
        statistics = (
            (training.level, synthetic.level, training.level_noise + synthetic.level_noise, 0.0),
            (training.spread, synthetic.spread, training.spread_noise + synthetic.spread_noise, training.mean_spread),
        )
        for training_values, synthetic_values, noise, reference in statistics:
            noise = max(noise, _NOISE_FLOOR)
            valid = np.isfinite(training_values) & np.isfinite(synthetic_values)
            difference = training_values[valid] - synthetic_values[valid]
            structure = np.maximum(
                np.abs(training_values[valid] - reference), np.abs(synthetic_values[valid] - reference)
            )
            excess += float(np.sum(difference**2 - noise)) / noise
            magnitude += float(np.sum(np.maximum(structure**2, noise))) / noise
        return float(np.clip(np.sqrt(max(excess, 0.0) / magnitude), 0.0, 1.0))

    @staticmethod
    def _synthetic_failure(base: dict[str, Any], training_profile: _SegmentProfile, reason: str) -> dict[str, Any]:
        """Return zero-similarity details for a synthetic sequence with no usable local average or spread."""
        segments = base["segments"]
        return {
            **base,
            "error": 1.0,
            "similarity": 0.0,
            "reason": reason,
            "training_level": RollingStatsSimilarity._profile_details(training_profile.level),
            "synthetic_level": [None] * segments,
            "training_spread": RollingStatsSimilarity._profile_details(training_profile.spread),
            "synthetic_spread": [None] * segments,
        }

    @staticmethod
    def _has_finite_segment(profile: _SegmentProfile) -> bool:
        """Return whether any segment's level or spread is finite."""
        return bool(np.any(np.isfinite(profile.level)) or np.any(np.isfinite(profile.spread)))

    @staticmethod
    def _shares_a_segment(training: _SegmentProfile, synthetic: _SegmentProfile) -> bool:
        """Return whether any segment's level or spread is finite in both profiles."""
        level = np.isfinite(training.level) & np.isfinite(synthetic.level)
        spread = np.isfinite(training.spread) & np.isfinite(synthetic.spread)
        return bool(np.any(level) or np.any(spread))

    @staticmethod
    def _prepare_values(values: pd.Series) -> NDArray[np.float64]:
        """Convert a series to floats while preserving non-finite positions as gaps."""
        numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(numeric_values), numeric_values, np.nan)

    @staticmethod
    def _profile_details(values: NDArray[np.float64]) -> list[float | None]:
        """Convert segment averages into JSON-safe full-precision details."""
        return [float(value) if np.isfinite(value) else None for value in values]

    @staticmethod
    def _is_effectively_constant(values: NDArray[np.float64]) -> bool:
        """Return whether finite variation is negligible relative to value scale."""
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return True
        scale = float(np.max(np.abs(finite_values)))
        if scale == 0.0:
            return True
        normalized_std = float(np.std(finite_values / scale))
        tolerance = _CONSTANT_TOLERANCE_FACTOR * np.finfo(float).eps
        return bool(normalized_std <= tolerance)
