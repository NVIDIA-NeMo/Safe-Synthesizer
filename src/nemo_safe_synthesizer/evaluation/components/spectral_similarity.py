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
from numpy.lib.stride_tricks import sliding_window_view
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

        Each sequence's spectrum is computed from all of its observations, and
        its share of variation in each cycle-length band is compared on the
        same bands, so sequences of different lengths can be compared. The
        difference between the two band profiles is scored by
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

        centers = SpectralSimilarity._band_centers(
            min(len(training_values), len(synthetic_values)), cfg.max_cycle_length
        )
        training_weights, synthetic_weights, cycle_lengths = SpectralSimilarity._shared_band_weights(
            len(training_values), len(synthetic_values), centers
        )
        training_profile = SpectralSimilarity._band_profile(training_values, training_weights)
        base = {"band_cycle_lengths": cycle_lengths.tolist()}
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
                "error": 1.0,
                "similarity": 0.0,
                "reason": synthetic_failure,
                "training_bands": training_profile.shares.tolist(),
                "synthetic_bands": [None] * len(cycle_lengths),
                "band_noise": training_profile.noise.tolist(),
            }, None

        synthetic_profile = SpectralSimilarity._band_profile(synthetic_values, synthetic_weights)
        error = SpectralSimilarity._noise_aware_error(training_profile, synthetic_profile)
        return {
            **base,
            "error": error,
            "similarity": 1.0 - error,
            "training_bands": training_profile.shares.tolist(),
            "synthetic_bands": synthetic_profile.shares.tolist(),
            "band_noise": (training_profile.noise + synthetic_profile.noise).tolist(),
        }, None

    @staticmethod
    def _band_centers(length: int, max_cycle_length: int) -> NDArray[np.float64]:
        """Return band centers, in octaves of cycle length, from 2 observations up to the longest distinguished cycle.

        The longest cycle is the shorter sequence's length or
        ``max_cycle_length``, whichever is smaller. Centers are half an octave
        apart.
        """
        longest = np.log2(min(length, max_cycle_length))
        return np.arange(_SHORTEST_BAND_OCTAVE, longest + _CENTER_TOLERANCE, _BAND_SPACING)

    @staticmethod
    def _shared_band_weights(
        training_length: int,
        synthetic_length: int,
        centers: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return each sequence's band weights over its own frequencies, on the bands both sequences cover.

        A sequence of ``n`` observations measures cycles of ``n / j``
        observations for ``j = 1 .. (n - 1) // 2``. Each frequency's weight
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
        """Return unnormalized triangular band weights over a sequence's frequencies."""
        frequencies = np.arange(1, (length - 1) // 2 + 1)
        position = np.clip(np.log2(length / frequencies), centers[0], centers[-1])
        return np.maximum(0.0, 1.0 - np.abs(position[None, :] - centers[:, None]) / _BAND_SPACING)

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

        Periodogram noise is exponential with mean equal to the local noise
        level, estimated as the running median over neighboring frequencies
        divided by ``ln 2``, so a single peak doesn't raise it. The level is
        capped at the frequency's own power. A steady cycle of power ``C`` on
        noise of level ``N`` varies by ``N^2 + 2 N C``, far less than its own
        power, so a cycle the synthetic data misses is not forgiven as noise.
        Band variances follow from these by the delta method, including the
        division by total power, and are scaled up for missing values.

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
        variance = level**2 + 2 * level * np.maximum(power - level, 0.0)
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
