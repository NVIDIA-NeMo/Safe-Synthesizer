# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spectral fidelity evaluation for numeric time-series columns.

The metric compares how training and synthetic sequences spread their
variation across cycle lengths: which repeating cycles they contain, how
strong those cycles are relative to noise, and whether slow or fast movement
dominates. It evaluates each shared group and numeric value column
separately, skips pairs with unusable training data, treats constant
synthetic output as a fidelity failure, and averages the remaining
similarities into a 0--10 score.
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

from ...config.evaluate import SpectralSimilarityParameters
from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...observability import get_logger
from .spectral_similarity_figures import (
    SPECTRAL_PAIR_SCORE_MAX_COLUMNS,
    SPECTRAL_SUMMARY_TRACES_PER_COLUMN,
    generate_spectral_band_difference_figure,
    generate_spectral_band_figure,
    generate_spectral_pair_score_figure,
    order_spectral_columns,
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

_GROUP_SELECTION_SEED = 2112
_CONSTANT_TOLERANCE_FACTOR = 32.0
# Band centers are spaced this many octaves apart in cycle length, starting at
# the shortest observable cycle of two observations.
_BAND_SPACING = 0.5
_SHORTEST_BAND_OCTAVE = 1.0
# Keeps a longest cycle that falls exactly on a center, such as 128, as a band.
_CENTER_TOLERANCE = 1e-9
# The noise level at each frequency is the median over this many frequencies on each side and itself.
_NOISE_HALF_WIDTH = 2


@dataclass(frozen=True)
class _BandProfile:
    """Share of variation in each cycle-length band and the sampling variance of each share."""

    shares: NDArray[np.float64]
    noise: NDArray[np.float64]


class SpectralSimilarity(TimeSeriesComponent):
    """Measure fidelity of variation across cycle lengths in numeric time-series data.

    The component compares the share of variation that training and synthetic
    sequences hold in overlapping cycle-length bands, independently for each
    shared group and numeric value column. Every usable comparison contributes
    equally to the final score. Skipped comparisons and drill-down summaries
    remain available in ``details`` so callers can locate columns or groups
    responsible for a mismatch.

    Attributes:
        name: Display name used in serialized evaluation results.
        details: Per-group/column band profiles, skipped comparisons, and summaries.
    """

    card_id: ClassVar[str] = "spectral-similarity"
    description: ClassVar[str] = (
        "Measures whether numeric columns repeat in cycles of the same lengths as the training data"
    )
    tooltip_key: ClassVar[str] = "spectral_similarity_info"
    template: ClassVar[str] = "jinja/components/spectral_similarity.j2"

    name: str = Field(
        default="Spectral Similarity",
        description="Display name used in serialized evaluation results.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-group/column band profiles, skipped comparisons, and summaries.",
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
                    "Cycle lengths",
                    generate_spectral_band_figure(profiles),
                    True,
                    "spectral_bands_info",
                ),
                (
                    "Difference by cycle length",
                    generate_spectral_band_difference_figure(profiles),
                    False,
                    "spectral_difference_info",
                ),
                (
                    "Pair scores",
                    generate_spectral_pair_score_figure(profiles, self.score.score),
                    False,
                    "spectral_pair_scores_info",
                ),
            ]
            summary_columns = order_spectral_columns(profiles)
        context["figures"] = self.figure_rows(figures)
        context["summary_columns"] = summary_columns
        context["summary_traces_per_column"] = SPECTRAL_SUMMARY_TRACES_PER_COLUMN
        context["pair_score_column_count"] = min(len(summary_columns), SPECTRAL_PAIR_SCORE_MAX_COLUMNS)
        context["evaluated_profile_count"] = len(profiles)
        context["evaluated_group_count"] = len({str(item["group"]) for item in profiles})
        context["evaluated_column_count"] = len({item["column"] for item in profiles})
        context["constant_synthetic_profile_count"] = sum(
            1 for item in profiles if all(value is None for value in item["synthetic_bands"])
        )
        return context

    @staticmethod
    def from_evaluation_datasets(
        evaluation_datasets: EvaluationDatasets,
        config: SafeSynthesizerParameters | None = None,
    ) -> SpectralSimilarity:
        """Evaluate spectral fidelity for paired time-series datasets.

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
        cfg = SpectralSimilarity._resolve_config(config)
        # Optional metrics must fail independently so one diagnostic cannot
        # prevent the rest of the evaluation report from being produced.
        try:
            return SpectralSimilarity._evaluate(evaluation_datasets, cfg, config)
        except Exception as exc:
            logger.exception("Failed to compute Spectral Similarity.")
            return SpectralSimilarity(score=EvaluationScore(notes=str(exc)))

    @staticmethod
    def _resolve_config(config: SafeSynthesizerParameters | None) -> SpectralSimilarityParameters:
        """Return configured metric parameters or an isolated default model."""
        if config is None:
            return SpectralSimilarityParameters()
        return config.evaluation.time_series.spectral

    @staticmethod
    def _evaluate(
        datasets: EvaluationDatasets,
        cfg: SpectralSimilarityParameters,
        config: SafeSynthesizerParameters | None,
    ) -> SpectralSimilarity:
        """Compute per-group/column band profiles and aggregate the component score.

        Args:
            datasets: Training and synthetic datasets to compare.
            cfg: Resolved spectral metric parameters.
            config: Optional top-level configuration used for inherited columns.

        Returns:
            A scored component, or an unavailable component with explanatory
            notes when no usable comparison remains.
        """
        timestamp_column = config.time_series.timestamp_column if config is not None else None
        group_column, group_error = resolve_group_column(datasets, config)
        if group_error is not None:
            return SpectralSimilarity(score=EvaluationScore(notes=group_error))

        columns, column_error, excluded_columns = select_value_columns(
            datasets, cfg.value_columns, timestamp_column, group_column, "spectral"
        )
        if column_error is not None:
            return SpectralSimilarity(score=EvaluationScore(notes=column_error))
        if not columns:
            return SpectralSimilarity._unavailable("No shared numeric value columns.", excluded_columns)

        groups, missing_training, missing_synthetic, shared_group_count = shared_groups(
            datasets.training,
            datasets.synthetic,
            group_column,
            cfg.max_groups,
            _GROUP_SELECTION_SEED,
        )
        if not groups:
            return SpectralSimilarity._unavailable("No shared groups to evaluate.", excluded_columns)
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
                result, reason = SpectralSimilarity._pair_score(training_group[column], synthetic_group[column], cfg)
                if result is None:
                    skipped.append({"group": group_label, "column": column, "reason": reason})
                else:
                    profiles.append({"group": group_label, "column": column, **result})

        counts = {
            "groups": len(groups),
            "shared_groups": shared_group_count,
            "omitted_shared_groups": omitted_groups,
            "columns": len(columns),
            "evaluated_profiles": len(profiles),
            "skipped": len(skipped),
        }
        if not profiles:
            unavailable_note = "No usable group/column spectral profiles."
            if omitted_groups:
                unavailable_note += (
                    f" Evaluated {len(groups)} of {shared_group_count} shared groups using a reproducible sample."
                )
            return SpectralSimilarity._unavailable(
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
            "max_cycle_length": cfg.max_cycle_length,
            "min_points": cfg.min_points,
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
        return SpectralSimilarity(score=score, details=details)

    @staticmethod
    def _unavailable(
        note: str,
        excluded_columns: list[str],
        details: dict[str, Any] | None = None,
    ) -> SpectralSimilarity:
        """Return an unscored component whose notes and details still name left-out columns."""
        excluded_note = excluded_columns_note(excluded_columns)
        return SpectralSimilarity(
            score=EvaluationScore(notes=note if excluded_note is None else f"{note} {excluded_note}"),
            details={**(details or {}), "excluded_columns": excluded_columns},
        )

    @staticmethod
    def _pair_score(
        training: pd.Series,
        synthetic: pd.Series,
        cfg: SpectralSimilarityParameters,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Compare how one training and synthetic sequence spread their variation across cycle lengths.

        Each sequence's spectrum is computed from all of its observations, and
        its share of variation in each cycle-length band is compared on the
        same bands, so sequences of different lengths can be compared. The
        difference between the two band profiles is scored by
        ``_noise_aware_error``, so differences explained by sampling noise do
        not lower the score on average. Level, scale, phase, and a straight-line
        trend do not affect the score.

        Training sequences that are too short, constant, or trend-only are
        skipped, because they have no spectrum to reproduce. Synthetic
        sequences with the same problems, or with fewer than ``min_points``
        finite values, score 0.

        Args:
            training: Ordered training values for one group and column.
            synthetic: Ordered synthetic values for the same group and column.
            cfg: Parameters controlling minimum length and the longest cycle.

        Returns:
            Profile details and ``None`` on success, or ``None`` and a
            human-readable skip reason when the training sequence is unusable.
        """
        training_values = SpectralSimilarity._prepare_values(training)
        synthetic_values = SpectralSimilarity._prepare_values(synthetic)
        training_reason = SpectralSimilarity._unusable_reason(training_values, "training", cfg.min_points)
        if training_reason is not None:
            return None, training_reason
        synthetic_reason = SpectralSimilarity._unusable_reason(synthetic_values, "synthetic", cfg.min_points)

        # A synthetic sequence with too few values may not cover even the
        # two-step band, so its pair takes the bands of the training sequence alone.
        synthetic_length = len(synthetic_values)
        if int(np.sum(np.isfinite(synthetic_values))) < cfg.min_points:
            synthetic_length = len(training_values)
        centers = SpectralSimilarity._band_centers(min(len(training_values), synthetic_length), cfg.max_cycle_length)
        training_weights, synthetic_weights, cycle_lengths = SpectralSimilarity._shared_band_weights(
            len(training_values), synthetic_length, centers
        )
        training_profile = SpectralSimilarity._band_profile(training_values, training_weights)
        if synthetic_reason is not None:
            return {
                "band_cycle_lengths": cycle_lengths.tolist(),
                "error": 1.0,
                "similarity": 0.0,
                "reason": synthetic_reason,
                "training_bands": training_profile.shares.tolist(),
                "synthetic_bands": [None] * len(cycle_lengths),
                "band_noise": training_profile.noise.tolist(),
            }, None

        synthetic_profile = SpectralSimilarity._band_profile(synthetic_values, synthetic_weights)
        error = SpectralSimilarity._noise_aware_error(training_profile, synthetic_profile)
        return {
            "band_cycle_lengths": cycle_lengths.tolist(),
            "error": error,
            "similarity": 1.0 - error,
            "training_bands": training_profile.shares.tolist(),
            "synthetic_bands": synthetic_profile.shares.tolist(),
            "band_noise": (training_profile.noise + synthetic_profile.noise).tolist(),
        }, None

    @staticmethod
    def _unusable_reason(values: NDArray[np.float64], dataset: str, min_points: int) -> str | None:
        """Return why a training or synthetic sequence has no spectrum to compare, or ``None`` if it has one."""
        if int(np.sum(np.isfinite(values))) < min_points:
            if dataset == "training":
                return f"The training sequence needs at least {min_points} finite observations."
            return f"The synthetic sequence has fewer than {min_points} finite observations."
        if SpectralSimilarity._is_effectively_constant(values):
            return f"The {dataset} sequence is constant or indistinguishable from constant at its numeric scale."
        if SpectralSimilarity._is_trend_only(values):
            return f"The {dataset} sequence has no variation beyond a straight-line trend."
        return None

    @staticmethod
    def _band_centers(length: int, max_cycle_length: int) -> NDArray[np.float64]:
        """Return band centers, in octaves of cycle length, from 2 observations up to the longest distinguished cycle.

        The longest cycle is the shorter sequence's length or
        ``max_cycle_length``, whichever is smaller. Centers are half an octave
        apart, and the longest cycle is always the last center, so it can sit
        less than half an octave above the one before it.
        """
        longest = np.log2(min(length, max_cycle_length))
        centers = np.arange(_SHORTEST_BAND_OCTAVE, longest + _CENTER_TOLERANCE, _BAND_SPACING)
        if longest - centers[-1] > _CENTER_TOLERANCE:
            centers = np.append(centers, longest)
        return centers

    @staticmethod
    def _shared_band_weights(
        training_length: int,
        synthetic_length: int,
        centers: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return each sequence's band weights over its own frequencies, on the bands both sequences cover.

        A sequence of ``n`` observations measures cycles of ``n / j``
        observations for ``j = 1 .. n // 2``. Each frequency's weight
        falls linearly to zero at the neighboring band centers, and cycles
        longer than the last center belong to the last band. Bands that
        contain no frequency in either sequence are dropped, which happens for
        the longest cycles, where ``n / j`` steps are more than half an octave
        apart. Every frequency's weights sum to 1, so band shares sum to 1.

        Args:
            training_length: Number of training positions, including gaps.
            synthetic_length: Number of synthetic positions, including gaps.
            centers: Band centers from ``_band_centers``.

        Returns:
            Training weights and synthetic weights, each with one row per band
            and one column per frequency, and the cycle length at each band
            center.
        """
        training_weights = SpectralSimilarity._band_weights(training_length, centers)
        synthetic_weights = SpectralSimilarity._band_weights(synthetic_length, centers)
        used = (np.sum(training_weights, axis=1) > 0) & (np.sum(synthetic_weights, axis=1) > 0)
        return (
            SpectralSimilarity._normalize_columns(training_weights[used]),
            SpectralSimilarity._normalize_columns(synthetic_weights[used]),
            2.0 ** centers[used],
        )

    @staticmethod
    def _band_weights(length: int, centers: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return unnormalized band weights that fall linearly from each center to its neighbors."""
        frequencies = np.arange(1, length // 2 + 1)
        position = np.log2(length / frequencies)
        return np.stack([np.interp(position, centers, band) for band in np.eye(len(centers))])

    @staticmethod
    def _normalize_columns(weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scale each frequency's weights to sum to 1, leaving frequencies outside every band at zero."""
        totals = np.sum(weights, axis=0)
        return np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)

    @staticmethod
    def _band_profile(values: NDArray[np.float64], weights: NDArray[np.float64]) -> _BandProfile:
        """Return the share of variation in each band and the sampling variance of each share.

        The spectrum is the periodogram of the whole sequence after removing a
        straight-line trend fitted at the original positions. Gaps stay in
        place: they contribute zero after centering, and power is divided by
        the number of finite values.

        Each frequency below ``n / 2`` stands for a positive and a negative
        frequency, so it counts twice. At even ``n`` the two-step cycle
        ``n / 2`` has no partner and counts once.

        Periodogram noise is exponential with mean equal to the local noise
        level, estimated as the running median over neighboring frequencies
        divided by ``ln 2``, so a single peak doesn't raise it. The level is
        capped at the frequency's own power. A steady cycle of power ``C`` on
        noise of level ``N`` varies by ``N^2 + 2 N C``, far less than its own
        power, so a cycle the synthetic data misses is not forgiven as noise.
        The two-step cycle at even ``n`` follows a chi-square distribution with
        one degree of freedom and varies by ``2 N^2 + 4 N C``. Band variances
        follow from these by the delta method, including the division by total
        power, and are scaled up for missing values.

        Args:
            values: Values in temporal order, with gaps as ``NaN``.
            weights: Band weights over this sequence's frequencies.

        Returns:
            Band shares and their sampling variances.
        """
        detrended = SpectralSimilarity._detrend(values)
        finite = np.isfinite(detrended)
        centered = np.where(finite, detrended - np.mean(detrended[finite]), 0.0)
        frequencies = weights.shape[1]
        power = np.abs(np.fft.rfft(centered))[1 : frequencies + 1] ** 2 / float(np.sum(finite))
        padded = np.pad(power, _NOISE_HALF_WIDTH, mode="edge")
        neighborhoods = sliding_window_view(padded, 2 * _NOISE_HALF_WIDTH + 1)
        level = np.minimum(np.median(neighborhoods, axis=1) / np.log(2), power)
        excess = np.maximum(power - level, 0.0)
        variance = level**2 + 2 * level * excess
        multiplicity = np.full(frequencies, 2.0)
        if len(values) % 2 == 0:
            multiplicity[-1] = 1.0
            variance[-1] = 2 * level[-1] ** 2 + 4 * level[-1] * excess[-1]
        power = multiplicity * power
        variance = multiplicity**2 * variance
        bands = np.einsum("bf,f->b", weights, power)
        total = float(np.sum(bands))
        shares = bands / total
        gradient = (weights - shares[:, None] * np.sum(weights, axis=0)[None, :]) / total
        noise = np.einsum("bf,f->b", gradient**2, variance) / float(np.mean(finite))
        return _BandProfile(shares=shares, noise=noise)

    @staticmethod
    def _noise_aware_error(training: _BandProfile, synthetic: _BandProfile) -> float:
        """Return the share of the band profile that differs beyond sampling noise.

        Sampling noise adds its variance to every squared band difference, so
        the expected noise variance is subtracted from each squared difference
        and the results are pooled across bands before taking the square root,
        as in Autocorrelation Similarity. The remaining difference is divided
        by the larger of the two band shares, floored at the noise, so losing
        variation in a band and inventing it are penalized alike, and bands
        holding more variation count more.

        Args:
            training: Training band shares and noise.
            synthetic: Synthetic band shares and noise over the same bands.

        Returns:
            Error in ``[0, 1]``.
        """
        noise = training.noise + synthetic.noise
        excess = float(np.sum((training.shares - synthetic.shares) ** 2 - noise))
        magnitude = np.maximum(np.maximum(training.shares, synthetic.shares), np.sqrt(noise))
        return float(np.clip(np.sqrt(max(excess, 0.0) / float(np.sum(magnitude**2))), 0.0, 1.0))

    @staticmethod
    def _detrend(values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Remove a least-squares straight line from the finite values, keeping gaps as ``NaN``."""
        finite = np.isfinite(values)
        positions = np.arange(len(values), dtype=float)
        positions -= float(np.mean(positions[finite]))
        center = float(np.mean(values[finite]))
        slope, intercept = np.polyfit(positions[finite], values[finite] - center, 1)
        return np.where(finite, values - center - (slope * positions + intercept), np.nan)

    @staticmethod
    def _is_trend_only(values: NDArray[np.float64]) -> bool:
        """Return whether variation around a straight-line trend is negligible relative to value scale."""
        finite = np.isfinite(values)
        residual = SpectralSimilarity._detrend(values)[finite]
        scale = float(np.max(np.abs(values[finite])))
        tolerance = _CONSTANT_TOLERANCE_FACTOR * np.finfo(float).eps
        return bool(float(np.std(residual)) <= tolerance * scale)

    @staticmethod
    def _prepare_values(values: pd.Series) -> NDArray[np.float64]:
        """Convert a series to floats while preserving non-finite positions as gaps."""
        numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(numeric_values), numeric_values, np.nan)

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
