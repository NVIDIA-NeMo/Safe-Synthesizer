# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autocorrelation-based fidelity evaluation for numeric time-series channels.

The metric compares lagged self-correlation in training and synthetic
sequences. It evaluates each shared group and numeric value column separately,
skips profiles with unusable training data, treats constant synthetic output
as a fidelity failure, and averages the remaining similarities into a 0--10
score.

Classes:
    AutocorrelationSimilarity: Component that computes and summarizes the
        autocorrelation fidelity score.
"""

from __future__ import annotations

import random
from collections import defaultdict
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field

from ...artifacts.analyzers.field_features import FieldType
from ...config.evaluate import AutocorrelationSimilarityParameters
from ...config.parameters import SafeSynthesizerParameters
from ...defaults import PSEUDO_GROUP_COLUMN
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...observability import get_logger
from .component import Component

logger = get_logger(__name__)

_MIN_VALID_PAIRS = 3
_GROUP_SELECTION_SEED = 2112
_CONSTANT_TOLERANCE_FACTOR = 32.0


class AutocorrelationSimilarity(Component):
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
        """Return score and diagnostic details for report rendering."""
        context = super().jinja_context
        context["details"] = self.details
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
        # Optional metrics must fail independently so one diagnostic cannot
        # prevent the rest of the evaluation report from being produced.
        try:
            return AutocorrelationSimilarity._evaluate(evaluation_datasets, cfg, config)
        except Exception as exc:
            logger.exception("Failed to compute Autocorrelation Similarity.")
            return AutocorrelationSimilarity(score=EvaluationScore(notes=str(exc)))

    @staticmethod
    def _resolve_config(config: SafeSynthesizerParameters | None) -> AutocorrelationSimilarityParameters:
        """Return configured metric parameters or an isolated default model."""
        if config is None:
            return AutocorrelationSimilarityParameters()
        return config.evaluation.time_series.autocorrelation

    @staticmethod
    def _evaluate(
        datasets: EvaluationDatasets,
        cfg: AutocorrelationSimilarityParameters,
        config: SafeSynthesizerParameters | None,
    ) -> AutocorrelationSimilarity:
        """Compute per-group/column profiles and aggregate the component score.

        Args:
            datasets: Training and synthetic datasets to compare.
            cfg: Resolved autocorrelation metric parameters.
            config: Optional top-level configuration used for inherited columns.

        Returns:
            A scored component, or an unavailable component with explanatory
            notes when no usable comparison remains.
        """
        timestamp_column = config.time_series.timestamp_column if config is not None else None
        group_column = config.data.group_training_examples_by if config is not None else None
        if group_column == PSEUDO_GROUP_COLUMN:
            # Training injects this reserved column to reuse grouped sequence
            # infrastructure, but evaluation receives frames with it removed.
            group_column = None
        elif group_column is not None and (
            group_column not in datasets.training or group_column not in datasets.synthetic
        ):
            return AutocorrelationSimilarity(
                score=EvaluationScore(notes=f"Configured group column {group_column!r} is missing from a dataset.")
            )

        columns, column_error = AutocorrelationSimilarity._numeric_columns(
            datasets, cfg, timestamp_column, group_column
        )
        if column_error is not None:
            return AutocorrelationSimilarity(score=EvaluationScore(notes=column_error))
        if not columns:
            return AutocorrelationSimilarity(score=EvaluationScore(notes="No shared numeric value columns."))

        groups, missing_training, missing_synthetic, shared_group_count = AutocorrelationSimilarity._shared_groups(
            datasets.training,
            datasets.synthetic,
            group_column,
            cfg.max_groups,
        )
        if not groups:
            return AutocorrelationSimilarity(score=EvaluationScore(notes="No shared groups to evaluate."))
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
            training_group = AutocorrelationSimilarity._group_frame(
                datasets.training, group_column, group_value, timestamp_column
            )
            synthetic_group = AutocorrelationSimilarity._group_frame(
                datasets.synthetic, group_column, group_value, timestamp_column
            )
            for column in columns:
                result, reason = AutocorrelationSimilarity._profile_score(
                    training_group[column], synthetic_group[column], cfg
                )
                group_label = None if group_column is None else str(group_value)
                AutocorrelationSimilarity._record_profile_result(
                    profiles,
                    skipped,
                    group_label,
                    column,
                    result,
                    reason,
                )

        if not profiles:
            unavailable_note = "No usable group/column autocorrelation profiles."
            if omitted_groups:
                unavailable_note += (
                    f" Evaluated {len(groups)} of {shared_group_count} shared groups using a reproducible sample."
                )
            return AutocorrelationSimilarity(
                score=EvaluationScore(notes=unavailable_note),
                details={
                    "counts": {
                        "groups": len(groups),
                        "shared_groups": shared_group_count,
                        "omitted_shared_groups": omitted_groups,
                        "columns": len(columns),
                        "evaluated_profiles": 0,
                        "skipped": len(skipped),
                    },
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
            "max_lag": cfg.max_lag,
            "min_points": cfg.min_points,
            "counts": {
                "groups": len(groups),
                "shared_groups": shared_group_count,
                "omitted_shared_groups": omitted_groups,
                "columns": len(columns),
                "evaluated_profiles": len(profiles),
                "skipped": len(skipped),
            },
            "group_selection": group_selection,
            "per_group": AutocorrelationSimilarity._summaries(profiles, "group"),
            "per_column": AutocorrelationSimilarity._summaries(profiles, "column"),
            "profiles": profiles,
            "skipped": skipped,
            "groups_only_in_training": missing_training,
            "groups_only_in_synthetic": missing_synthetic,
        }
        return AutocorrelationSimilarity(score=score, details=details)

    @staticmethod
    def _record_profile_result(
        profiles: list[dict[str, Any]],
        skipped: list[dict[str, Any]],
        group_label: str | None,
        column: str,
        result: dict[str, Any] | None,
        reason: str | None,
    ) -> None:
        """Record one successful or skipped group-and-column profile."""
        if result is None:
            skipped.append({"group": group_label, "column": column, "reason": reason})
            return
        profiles.append({"group": group_label, "column": column, **result})

    @staticmethod
    def _numeric_columns(
        datasets: EvaluationDatasets,
        cfg: AutocorrelationSimilarityParameters,
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
            invalid_columns = AutocorrelationSimilarity._invalid_value_columns(
                datasets,
                cfg.value_columns,
                timestamp_column,
                group_column,
            )
            if invalid_columns:
                return [], "Invalid autocorrelation value columns: " + "; ".join(invalid_columns)
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
        ordered_shared = sorted(shared_groups, key=AutocorrelationSimilarity._group_sort_key)
        shared = ordered_shared
        if len(ordered_shared) > max_groups:
            shared = random.Random(_GROUP_SELECTION_SEED).sample(ordered_shared, k=max_groups)
            shared.sort(key=AutocorrelationSimilarity._group_sort_key)
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
    def _profile_score(
        training: pd.Series,
        synthetic: pd.Series,
        cfg: AutocorrelationSimilarityParameters,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Compare one training and synthetic autocorrelation profile.

        For each lag, Pearson correlation is computed over positions whose two
        endpoints are finite. Lags are capped at half the shorter finite
        sequence. The mean profile difference is divided by two because
        correlation lies in ``[-1, 1]``.

        Args:
            training: Ordered training values for one group and column.
            synthetic: Ordered synthetic values for the same group and column.
            cfg: Parameters controlling minimum length and maximum lag.

        Returns:
            Profile score details and ``None`` on success, or ``None`` and a
            human-readable skip reason when the comparison is unusable.
        """
        training_values = AutocorrelationSimilarity._prepare_values(training)
        synthetic_values = AutocorrelationSimilarity._prepare_values(synthetic)
        training_count = int(np.sum(np.isfinite(training_values)))
        synthetic_count = int(np.sum(np.isfinite(synthetic_values)))
        n = min(training_count, synthetic_count)
        if n < cfg.min_points:
            return None, f"Each sequence needs at least {cfg.min_points} finite observations."
        if AutocorrelationSimilarity._is_effectively_constant(training_values):
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
        if AutocorrelationSimilarity._is_effectively_constant(synthetic_values):
            return {
                "lags": lags,
                "effective_max_lag": effective_max_lag,
                "evaluated_lags": 0,
                "error": 1.0,
                "similarity": 0.0,
                "reason": "The synthetic sequence is constant or indistinguishable from constant at its numeric scale.",
                "training_acf": AutocorrelationSimilarity._profile_details(training_acf),
                "synthetic_acf": [None] * effective_max_lag,
                "training_pair_support": training_support.tolist(),
                "synthetic_pair_support": synthetic_support.tolist(),
            }, None

        shared_valid_lags = np.isfinite(training_acf) & np.isfinite(synthetic_acf)
        if not np.any(shared_valid_lags):
            return None, f"No lags have at least {_MIN_VALID_PAIRS} usable endpoint pairs in both sequences."
        error = float(np.mean(np.abs(training_acf[shared_valid_lags] - synthetic_acf[shared_valid_lags])) / 2.0)
        error = float(np.clip(error, 0.0, 1.0))
        return {
            "lags": lags,
            "effective_max_lag": effective_max_lag,
            "evaluated_lags": int(np.sum(shared_valid_lags)),
            "error": error,
            "similarity": 1.0 - error,
            "training_acf": AutocorrelationSimilarity._profile_details(training_acf),
            "synthetic_acf": AutocorrelationSimilarity._profile_details(synthetic_acf),
            "training_pair_support": training_support.tolist(),
            "synthetic_pair_support": synthetic_support.tolist(),
        }, None

    @staticmethod
    def _prepare_values(values: pd.Series) -> NDArray[np.float64]:
        """Convert a series to floats while preserving non-finite positions as gaps."""
        numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(numeric_values), numeric_values, np.nan)

    @staticmethod
    def _profile_details(values: NDArray[np.float64]) -> list[float | None]:
        """Convert an ACF vector into JSON-safe full-precision details."""
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
            if AutocorrelationSimilarity._is_effectively_constant(earlier):
                continue
            if AutocorrelationSimilarity._is_effectively_constant(later):
                continue
            correlation = float(np.corrcoef(earlier, later)[0, 1])
            acf[lag - 1] = np.clip(correlation, -1.0, 1.0)
        return acf, support

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
