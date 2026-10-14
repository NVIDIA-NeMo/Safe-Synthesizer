# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rolling statistics fidelity evaluation for numeric time-series columns.

The metric compares how the local level (rolling average) and local spread
(rolling standard deviation) change from the beginning to the end of training
and synthetic sequences. It evaluates each shared group and numeric value
column separately, skips pairs with unusable training data, treats constant
synthetic output as a fidelity failure, and averages the remaining similarities
into a 0--10 score.

Classes:
    RollingStatsSimilarity: Component that computes and summarizes the
        rolling statistics fidelity score.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from pydantic import Field

from ...artifacts.analyzers.field_features import FieldType
from ...config.evaluate import RollingStatsSimilarityParameters
from ...config.parameters import SafeSynthesizerParameters
from ...defaults import PSEUDO_GROUP_COLUMN
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...observability import get_logger
from .component import Component

logger = get_logger(__name__)

ROLLING_STATS_SEGMENTS = 10
_MIN_WINDOW = 3
_GROUP_SELECTION_SEED = 2112
_CONSTANT_TOLERANCE_FACTOR = 32.0
# Floor for segment noise, in squared training standard deviations, so noise-free
# deterministic sequences still compare cleanly.
_NOISE_FLOOR = 1e-12
_MAX_LAG_ONE_CORRELATION = 0.99
_MIN_LAG_ONE_CORRELATION = -0.95
_MIN_VARIANCE_CORRECTION = 0.5


@dataclass(frozen=True)
class _SegmentProfile:
    """Segment averages of one standardized sequence and their chance variances."""

    level: NDArray[np.float64]
    spread: NDArray[np.float64]
    level_noise: float
    spread_noise: float
    mean_spread: float


class RollingStatsSimilarity(Component):
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
        """Return score and diagnostic details for report rendering."""
        context = super().jinja_context
        context["details"] = self.details
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
        group_column = config.data.group_training_examples_by if config is not None else None
        if (
            group_column == PSEUDO_GROUP_COLUMN
            and group_column not in datasets.training
            and group_column not in datasets.synthetic
        ):
            # Training injects this reserved column to reuse grouped sequence
            # infrastructure, but evaluation receives frames with it removed.
            group_column = None
        elif group_column is not None and (
            group_column not in datasets.training or group_column not in datasets.synthetic
        ):
            return RollingStatsSimilarity(
                score=EvaluationScore(notes=f"Configured group column {group_column!r} is missing from a dataset.")
            )

        columns, column_error = RollingStatsSimilarity._numeric_columns(datasets, cfg, timestamp_column, group_column)
        if column_error is not None:
            return RollingStatsSimilarity(score=EvaluationScore(notes=column_error))
        if not columns:
            return RollingStatsSimilarity(score=EvaluationScore(notes="No shared numeric value columns."))

        groups, missing_training, missing_synthetic, shared_group_count = RollingStatsSimilarity._shared_groups(
            datasets.training,
            datasets.synthetic,
            group_column,
            cfg.max_groups,
        )
        if not groups:
            return RollingStatsSimilarity(score=EvaluationScore(notes="No shared groups to evaluate."))
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
            training_group = RollingStatsSimilarity._group_frame(
                datasets.training, group_column, group_value, timestamp_column
            )
            synthetic_group = RollingStatsSimilarity._group_frame(
                datasets.synthetic, group_column, group_value, timestamp_column
            )
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
            return RollingStatsSimilarity(
                score=EvaluationScore(notes=unavailable_note),
                details={
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
            "per_group": RollingStatsSimilarity._summaries(profiles, "group"),
            "per_column": RollingStatsSimilarity._summaries(profiles, "column"),
            "profiles": profiles,
            "skipped": skipped,
            "groups_only_in_training": missing_training,
            "groups_only_in_synthetic": missing_synthetic,
        }
        return RollingStatsSimilarity(score=score, details=details)

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
    def _numeric_columns(
        datasets: EvaluationDatasets,
        cfg: RollingStatsSimilarityParameters,
        timestamp_column: str | None,
        group_column: str | None,
    ) -> tuple[list[str], str | None]:
        """Select shared numeric value columns eligible for evaluation.

        Explicit ``value_columns`` retain their configured order. Automatic
        selection is sorted for deterministic output. Timestamp and grouping
        columns are excluded even when their storage dtype is numeric.

        Args:
            datasets: Training and synthetic datasets with inferred field types.
            cfg: Resolved metric parameters.
            timestamp_column: Column used only to order observations.
            group_column: Column used only to separate sequences.

        Returns:
            Shared numeric value column names and an optional validation error.
        """
        if cfg.value_columns is not None:
            invalid_columns = RollingStatsSimilarity._invalid_value_columns(
                datasets,
                cfg.value_columns,
                timestamp_column,
                group_column,
            )
            if invalid_columns:
                return [], "Invalid rolling statistics value columns: " + "; ".join(invalid_columns)
            return list(dict.fromkeys(cfg.value_columns)), None
        numeric = set(datasets.get_columns_of_type({FieldType.NUMERIC}, based_on="both"))
        numeric.difference_update([timestamp_column, group_column, PSEUDO_GROUP_COLUMN])
        return sorted(numeric), None

    @staticmethod
    def _invalid_value_columns(
        datasets: EvaluationDatasets,
        columns: list[str],
        timestamp_column: str | None,
        group_column: str | None,
    ) -> list[str]:
        """Return actionable validation failures for explicit value columns."""
        invalid: list[str] = []
        reserved = {timestamp_column, group_column, PSEUDO_GROUP_COLUMN}
        for column in columns:
            if column in reserved:
                invalid.append(f"{column!r} is used for timestamp or sequence grouping")
                continue
            missing_from = [
                name
                for name, frame in (("training", datasets.training), ("synthetic", datasets.synthetic))
                if column not in frame
            ]
            if missing_from:
                invalid.append(f"{column!r} is missing from {' and '.join(missing_from)} data")
                continue
            if not pd.api.types.is_numeric_dtype(datasets.training[column]):
                invalid.append(f"{column!r} is not numeric in training data")
                continue
            if not pd.api.types.is_numeric_dtype(datasets.synthetic[column]):
                invalid.append(f"{column!r} is not numeric in synthetic data")
        return invalid

    @staticmethod
    def _shared_groups(
        training: pd.DataFrame,
        synthetic: pd.DataFrame,
        group_column: str | None,
        max_groups: int,
    ) -> tuple[list[Any], list[str], list[str], int]:
        """Find a deterministic, bounded set of groups shared by both datasets.

        A ``None`` sentinel represents one global sequence when grouping is not
        configured. Group labels present in only one dataset are returned for
        diagnostics but do not contribute to the score.

        Args:
            training: Training records containing candidate group labels.
            synthetic: Synthetic records containing candidate group labels.
            group_column: Optional column that identifies independent sequences.
            max_groups: Maximum number of shared groups to evaluate.

        Returns:
            Shared group keys, training-only labels, synthetic-only labels,
            and the total number of shared groups before limiting.
        """
        if group_column is None:
            return [None], [], [], 1
        if group_column not in training or group_column not in synthetic:
            return [], [], [], 0
        training_groups = set(training[group_column].dropna().unique())
        synthetic_groups = set(synthetic[group_column].dropna().unique())
        shared_groups = training_groups & synthetic_groups
        ordered_shared = sorted(shared_groups, key=RollingStatsSimilarity._group_sort_key)
        shared = ordered_shared
        if len(ordered_shared) > max_groups:
            shared = random.Random(_GROUP_SELECTION_SEED).sample(ordered_shared, k=max_groups)
            shared.sort(key=RollingStatsSimilarity._group_sort_key)
        only_training = [str(value) for value in sorted(training_groups - synthetic_groups, key=str)]
        only_synthetic = [str(value) for value in sorted(synthetic_groups - training_groups, key=str)]
        return shared, only_training, only_synthetic, len(shared_groups)

    @staticmethod
    def _group_sort_key(value: Any) -> tuple[str, str]:
        """Return a stable ordering key for heterogeneous group values."""
        type_name = f"{type(value).__module__}.{type(value).__qualname__}"
        return type_name, str(value)

    @staticmethod
    def _group_frame(
        df: pd.DataFrame,
        group_column: str | None,
        group_value: Any,
        timestamp_column: str | None,
    ) -> pd.DataFrame:
        """Return one sequence in deterministic timestamp order.

        ``mergesort`` preserves input order for equal timestamps, which makes
        repeated runs stable without inventing a secondary ordering key.

        Args:
            df: Dataset containing one or more sequences.
            group_column: Optional sequence identifier column.
            group_value: Group to select, or ``None`` for the full dataset.
            timestamp_column: Optional column used to order the selected rows.

        Returns:
            A filtered and time-ordered DataFrame view or copy.
        """
        frame = df if group_column is None else df[df[group_column] == group_value]
        if timestamp_column and timestamp_column in frame:
            frame = frame.sort_values(timestamp_column, kind="mergesort")
        return frame

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

        Returns:
            Profile details and ``None`` on success, or ``None`` and a
            human-readable skip reason when the comparison is unusable.
        """
        training_values = RollingStatsSimilarity._prepare_values(training)
        synthetic_values = RollingStatsSimilarity._prepare_values(synthetic)
        n = min(int(np.sum(np.isfinite(training_values))), int(np.sum(np.isfinite(synthetic_values))))
        if n < cfg.min_points:
            return None, f"Each sequence needs at least {cfg.min_points} finite observations."
        if RollingStatsSimilarity._is_effectively_constant(training_values):
            return None, "The training sequence is constant or indistinguishable from constant at its numeric scale."

        window = min(cfg.window_size, max(_MIN_WINDOW, n // 2))
        segments = min(ROLLING_STATS_SEGMENTS, len(training_values) - window + 1, len(synthetic_values) - window + 1)
        center = float(np.nanmean(training_values))
        scale = float(np.nanstd(training_values))
        training_profile = RollingStatsSimilarity._segment_profile((training_values - center) / scale, window, segments)
        base = {"window": window, "segments": segments}
        if RollingStatsSimilarity._is_effectively_constant(synthetic_values):
            return {
                **base,
                "error": 1.0,
                "similarity": 0.0,
                "reason": "The synthetic sequence is constant or indistinguishable from constant at its numeric scale.",
                "training_level": RollingStatsSimilarity._profile_details(training_profile.level),
                "synthetic_level": [None] * segments,
                "training_spread": RollingStatsSimilarity._profile_details(training_profile.spread),
                "synthetic_spread": [None] * segments,
            }, None

        synthetic_profile = RollingStatsSimilarity._segment_profile(
            (synthetic_values - center) / scale, window, segments
        )
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
        span = len(level) / segments
        level_noise, spread_noise = RollingStatsSimilarity._segment_noise(values, squares, window, span)
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
        enough = count >= max(2, (window + 1) // 2)
        return np.where(enough, total / np.maximum(count, 1), np.nan)

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
        span: float,
    ) -> tuple[float, float]:
        """Return the chance variance of one segment's average level and average spread.

        Each segment average is a weighted average of ``span`` consecutive
        rolling windows, so its variance follows from the autocovariance of
        the underlying series. The values are linearly detrended, so a steady
        trend counts as structure rather than chance, and fitted with a
        bias-corrected first-order autoregressive model, which gives smooth
        autocovariances at every lag even for short sequences. The spread uses
        the same algebra on the squared deviations, converted with
        ``Var(s) = Var(s^2) / (4 E[s^2])``.

        Args:
            values: Standardized values in temporal order, with gaps as ``NaN``.
            squares: Squared deviations from the local rolling average.
            window: Rolling window length.
            span: Number of rolling windows averaged in each segment.

        Returns:
            Chance variances of the segment level and segment spread.
        """
        span_windows = max(int(round(span)), 1)
        weights = np.convolve(np.ones(window), np.ones(span_windows)) / (window * span_windows)
        overlap = np.correlate(weights, weights, mode="full")[len(weights) - 1 :]
        count = max(int(np.sum(np.isfinite(values))), 1)
        level_variance, level_correlation = RollingStatsSimilarity._ar1_fit(RollingStatsSimilarity._detrend(values))
        mean_square = float(np.nanmean(squares))
        square_variance, square_correlation = RollingStatsSimilarity._ar1_fit(squares - mean_square)
        level_noise = RollingStatsSimilarity._average_variance(level_variance, level_correlation, overlap, count)
        square_noise = RollingStatsSimilarity._average_variance(square_variance, square_correlation, overlap, count)
        spread_noise = square_noise / (4 * mean_square) if mean_square > 0 else 0.0
        return level_noise, spread_noise

    @staticmethod
    def _average_variance(variance: float, correlation: float, overlap: NDArray[np.float64], count: int) -> float:
        """Variance of a weighted average under AR(1) autocovariances ``variance * correlation^k``.

        ``overlap[k]`` is the sum of products of averaging weights ``k`` steps
        apart. The result is floored at the independent-values variance divided
        by the sequence length, so strongly negative correlations cannot drive
        it to zero.
        """
        lags = np.arange(len(overlap))
        total = variance * float(np.sum(overlap * correlation**lags * np.where(lags > 0, 2.0, 1.0)))
        return max(total, variance * float(overlap[0]) / count)

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
        pairs = max(int(np.sum(finite[:-1] & finite[1:])), 1)
        correlation = float(np.dot(filled[:-1], filled[1:])) / pairs / variance
        correlation = float(
            np.clip(correlation + (1 + 3 * correlation) / n, _MIN_LAG_ONE_CORRELATION, _MAX_LAG_ONE_CORRELATION)
        )
        persistence = (1 + correlation) / (1 - correlation)
        return variance / max(1 - persistence / n, _MIN_VARIANCE_CORRECTION), correlation

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
        if magnitude <= 0:
            return 0.0
        return float(np.clip(np.sqrt(max(excess, 0.0) / magnitude), 0.0, 1.0))

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

    @staticmethod
    def _summaries(profiles: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        """Average profile similarities by a diagnostic key.

        Args:
            profiles: Successful group-and-column comparison details.
            key: Detail key to group by, such as ``group`` or ``column``.

        Returns:
            Deterministically ordered summaries with similarity and count.
        """
        scores: defaultdict[Any, list[float]] = defaultdict(list)
        for item in profiles:
            scores[item[key]].append(item["similarity"])
        return [
            {key: value, "similarity": float(np.mean(values)), "count": len(values)}
            for value, values in sorted(scores.items(), key=lambda item: str(item[0]))
        ]
