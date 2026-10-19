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

Classes:
    SpectralSimilarity: Component that computes and summarizes the spectral
        fidelity score.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field

from ...artifacts.analyzers.field_features import FieldType
from ...config.evaluate import SpectralSimilarityParameters
from ...config.parameters import SafeSynthesizerParameters
from ...defaults import PSEUDO_GROUP_COLUMN
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...observability import get_logger
from .component import Component

logger = get_logger(__name__)

_GROUP_SELECTION_SEED = 2112
_CONSTANT_TOLERANCE_FACTOR = 32.0
_MIN_SEGMENT_LENGTH = 8
_MIN_SEGMENTS = 2
# Band centers are spaced this many octaves apart in cycle length, starting at
# the shortest observable cycle of two observations.
_BAND_SPACING = 0.5
_SHORTEST_BAND_OCTAVE = 1.0


@dataclass(frozen=True)
class _BandProfile:
    """Share of variation in each cycle-length band and the sampling variance of each share."""

    shares: NDArray[np.float64]
    noise: NDArray[np.float64]
    segments: int


class SpectralSimilarity(Component):
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
        """Return score and diagnostic details for report rendering."""
        context = super().jinja_context
        context["details"] = self.details
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
            return SpectralSimilarity(
                score=EvaluationScore(notes=f"Configured group column {group_column!r} is missing from a dataset.")
            )

        columns, column_error = SpectralSimilarity._numeric_columns(datasets, cfg, timestamp_column, group_column)
        if column_error is not None:
            return SpectralSimilarity(score=EvaluationScore(notes=column_error))
        if not columns:
            return SpectralSimilarity(score=EvaluationScore(notes="No shared numeric value columns."))

        groups, missing_training, missing_synthetic, shared_group_count = SpectralSimilarity._shared_groups(
            datasets.training,
            datasets.synthetic,
            group_column,
            cfg.max_groups,
        )
        if not groups:
            return SpectralSimilarity(score=EvaluationScore(notes="No shared groups to evaluate."))
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
            training_group = SpectralSimilarity._group_frame(
                datasets.training, group_column, group_value, timestamp_column
            )
            synthetic_group = SpectralSimilarity._group_frame(
                datasets.synthetic, group_column, group_value, timestamp_column
            )
            group_label = None if group_column is None else str(group_value)
            for column in columns:
                result, reason = SpectralSimilarity._pair_score(training_group[column], synthetic_group[column], cfg)
                SpectralSimilarity._record_pair_result(profiles, skipped, group_label, column, result, reason)

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
            return SpectralSimilarity(
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
            "max_cycle_length": cfg.max_cycle_length,
            "min_points": cfg.min_points,
            "counts": counts,
            "group_selection": group_selection,
            "per_group": SpectralSimilarity._summaries(profiles, "group"),
            "per_column": SpectralSimilarity._summaries(profiles, "column"),
            "profiles": profiles,
            "skipped": skipped,
            "groups_only_in_training": missing_training,
            "groups_only_in_synthetic": missing_synthetic,
        }
        return SpectralSimilarity(score=score, details=details)

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
        cfg: SpectralSimilarityParameters,
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
            invalid_columns = SpectralSimilarity._invalid_value_columns(
                datasets,
                cfg.value_columns,
                timestamp_column,
                group_column,
            )
            if invalid_columns:
                return [], "Invalid spectral value columns: " + "; ".join(invalid_columns)
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
        ordered_shared = sorted(shared_groups, key=SpectralSimilarity._group_sort_key)
        shared = ordered_shared
        if len(ordered_shared) > max_groups:
            shared = random.Random(_GROUP_SELECTION_SEED).sample(ordered_shared, k=max_groups)
            shared.sort(key=SpectralSimilarity._group_sort_key)
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
        cfg: SpectralSimilarityParameters,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Compare how one training and synthetic sequence spread their variation across cycle lengths.

        Both sequences are cut into the same overlapping segments, and the
        share of variation in each cycle-length band is averaged over
        segments. The difference between the two band profiles is scored by
        ``_noise_aware_error``, so differences explained by sampling noise do
        not lower the score on average. Level, scale, phase, and a straight-line
        trend do not affect the score.

        Args:
            training: Ordered training values for one group and column.
            synthetic: Ordered synthetic values for the same group and column.
            cfg: Parameters controlling minimum length and the longest cycle.

        Returns:
            Profile details and ``None`` on success, or ``None`` and a
            human-readable skip reason when the comparison is unusable.
        """
        training_values = SpectralSimilarity._prepare_values(training)
        synthetic_values = SpectralSimilarity._prepare_values(synthetic)
        n = min(int(np.sum(np.isfinite(training_values))), int(np.sum(np.isfinite(synthetic_values))))
        if n < cfg.min_points:
            return None, f"Each sequence needs at least {cfg.min_points} finite observations."
        if SpectralSimilarity._is_effectively_constant(training_values):
            return None, "The training sequence is constant or indistinguishable from constant at its numeric scale."
        if SpectralSimilarity._is_trend_only(training_values):
            return None, "The training sequence has no variation beyond a straight-line trend."

        segment_length = SpectralSimilarity._segment_length(
            min(len(training_values), len(synthetic_values)), cfg.max_cycle_length
        )
        weights, cycle_lengths = SpectralSimilarity._band_weights(segment_length)
        training_profile = SpectralSimilarity._band_profile(training_values, segment_length, weights)
        if training_profile is None:
            return None, "Fewer than two training segments have at least half of their observations finite."
        base = {
            "segment_length": segment_length,
            "band_cycle_lengths": cycle_lengths.tolist(),
            "training_segments": training_profile.segments,
        }
        synthetic_failure = None
        if SpectralSimilarity._is_effectively_constant(synthetic_values):
            synthetic_failure = (
                "The synthetic sequence is constant or indistinguishable from constant at its numeric scale."
            )
        elif SpectralSimilarity._is_trend_only(synthetic_values):
            synthetic_failure = "The synthetic sequence has no variation beyond a straight-line trend."
        if synthetic_failure is not None:
            return {
                **base,
                "synthetic_segments": 0,
                "error": 1.0,
                "similarity": 0.0,
                "reason": synthetic_failure,
                "training_bands": training_profile.shares.tolist(),
                "synthetic_bands": [None] * len(cycle_lengths),
                "band_noise": training_profile.noise.tolist(),
            }, None

        synthetic_profile = SpectralSimilarity._band_profile(synthetic_values, segment_length, weights)
        if synthetic_profile is None:
            return None, "Fewer than two synthetic segments have at least half of their observations finite."
        error = SpectralSimilarity._noise_aware_error(training_profile, synthetic_profile)
        return {
            **base,
            "synthetic_segments": synthetic_profile.segments,
            "error": error,
            "similarity": 1.0 - error,
            "training_bands": training_profile.shares.tolist(),
            "synthetic_bands": synthetic_profile.shares.tolist(),
            "band_noise": (training_profile.noise + synthetic_profile.noise).tolist(),
        }, None

    @staticmethod
    def _segment_length(length: int, max_cycle_length: int) -> int:
        """Return an even segment length of at most half the shorter sequence and ``max_cycle_length``."""
        return max(_MIN_SEGMENT_LENGTH, min(max_cycle_length, length // 2) // 2 * 2)

    @staticmethod
    def _band_weights(segment_length: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return overlapping triangular band weights over a segment's frequencies and each band's cycle length.

        A segment of ``L`` observations measures cycles of ``L / j`` observations
        for ``j = 1 .. L / 2 - 1``. Band centers are spaced half an octave apart
        in cycle length, from 2 observations up to about ``L``, and each
        frequency's weight falls linearly to zero at the neighboring centers.
        Every frequency's weights sum to 1, so band shares sum to 1. Bands that
        contain no frequency are dropped, which happens for the longest cycles,
        where ``L / j`` steps are more than half an octave apart.

        Args:
            segment_length: Even number of observations per segment.

        Returns:
            Weights with one row per band and one column per frequency, and the
            cycle length at each band center.
        """
        frequencies = np.arange(1, segment_length // 2)
        position = np.log2(segment_length / frequencies)
        centers = np.arange(_SHORTEST_BAND_OCTAVE, np.log2(segment_length) + _BAND_SPACING, _BAND_SPACING)
        weights = np.maximum(0.0, 1.0 - np.abs(position[None, :] - centers[:, None]) / _BAND_SPACING)
        used = np.sum(weights, axis=1) > 0
        weights = weights[used] / np.sum(weights[used], axis=0)
        return weights, 2.0 ** centers[used]

    @staticmethod
    def _band_profile(
        values: NDArray[np.float64],
        segment_length: int,
        weights: NDArray[np.float64],
    ) -> _BandProfile | None:
        """Average band shares over segments and measure their noise from the spread between segments.

        Each segment's band power is linearized around the overall share,
        ``(band_s - share * total_s) / total``, so the variance across
        segments accounts for neighboring frequencies sharing leakage and for
        dividing by the total. A steady cycle, whose power is the same in
        every segment, has little noise.

        Args:
            values: Values in temporal order, with gaps as ``NaN``.
            segment_length: Observations per segment.
            weights: Band weights from ``_band_weights``.

        Returns:
            Band shares and their sampling variances, or ``None`` when fewer
            than two segments are usable or the sequence has no power.
        """
        powers, inflation = SpectralSimilarity._segment_powers(values, segment_length)
        if len(powers) < _MIN_SEGMENTS:
            return None
        bands = np.einsum("bf,sf->sb", weights, powers)
        totals = np.sum(powers, axis=1)
        total = float(np.mean(totals))
        if total <= 0:
            return None
        shares = np.mean(bands, axis=0) / total
        residuals = (bands - shares[None, :] * totals[:, None]) / total
        noise = np.var(residuals, axis=0, ddof=1) * inflation / len(powers)
        return _BandProfile(shares=shares, noise=noise, segments=len(powers))

    @staticmethod
    def _segment_powers(values: NDArray[np.float64], segment_length: int) -> tuple[NDArray[np.float64], float]:
        """Return one windowed periodogram per usable segment and the variance inflation of their average.

        Values are detrended by a straight line fitted at their original
        positions, then cut into Hann-windowed segments that overlap by half
        (Welch's method). Gaps stay in place: each segment is centered on its
        finite values, gaps contribute zero, and power is divided by the
        window energy over finite positions. Segments with fewer than half
        their values finite are dropped. The inflation corrects the variance
        of an average of ``K`` segments for their overlap and missing values.

        Args:
            values: Values in temporal order, with gaps as ``NaN``.
            segment_length: Observations per segment.

        Returns:
            Periodograms at frequencies ``1 .. L / 2 - 1``, one row per usable
            segment, and the variance inflation factor.
        """
        detrended = SpectralSimilarity._detrend(values)
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(segment_length) / segment_length)
        step = segment_length // 2
        powers: list[NDArray[np.float64]] = []
        coverage: list[float] = []
        for start in range(0, len(detrended) - segment_length + 1, step):
            segment = detrended[start : start + segment_length]
            finite = np.isfinite(segment)
            if np.sum(finite) < segment_length / 2:
                continue
            centered = np.where(finite, segment - np.mean(segment[finite]), 0.0)
            power = np.abs(np.fft.rfft(window * centered)) ** 2 / np.sum((window * finite) ** 2)
            powers.append(power[1:step])
            coverage.append(float(np.mean(finite)))
        if not powers:
            return np.empty((0, step - 1)), 1.0
        inflation = SpectralSimilarity._overlap_inflation(len(powers), window, step) / float(np.mean(coverage))
        return np.array(powers), inflation

    @staticmethod
    def _overlap_inflation(segments: int, window: NDArray[np.float64], step: int) -> float:
        """Return how much overlap inflates the variance of an average of windowed periodograms.

        For Gaussian noise, periodograms of segments ``k`` steps apart are
        correlated by the squared normalized window overlap, which is 1/36
        for Hann windows that overlap by half.
        """
        energy = float(np.sum(window**2))
        inflation = 1.0
        for lag in range(1, segments):
            shift = lag * step
            if shift >= len(window):
                break
            overlap = float(np.sum(window[:-shift] * window[shift:])) / energy
            inflation += 2 * (1 - lag / segments) * overlap**2
        return inflation

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
