# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autocorrelation-based fidelity evaluation for numeric time-series channels.

The metric compares lagged self-correlation in training and synthetic
sequences. It evaluates each shared group and numeric value column separately,
skips profiles that are too short or effectively constant, and averages the
remaining similarities into a 0--10 score.

Classes:
    AutocorrelationSimilarity: Component that computes and summarizes the
        autocorrelation fidelity score.
"""

from __future__ import annotations

from collections import defaultdict
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


class AutocorrelationSimilarity(Component):
    """Measure fidelity of lagged self-dependence in numeric time-series data.

    The component compares training and synthetic autocorrelation profiles
    independently for each shared group and numeric value column. Every usable
    comparison contributes equally to the final score. Skipped comparisons and
    drill-down summaries remain available in ``details`` so callers can locate
    columns or groups responsible for a mismatch.

    Attributes:
        name: Display name used in serialized evaluation results.
        details: Atomic profiles, skipped comparisons, and grouped summaries.
    """

    name: str = Field(
        default="Autocorrelation Similarity",
        description="Display name used in serialized evaluation results.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Atomic autocorrelation profiles, skipped comparisons, and summary statistics.",
    )

    @staticmethod
    def from_evaluation_datasets(
        evaluation_datasets: EvaluationDatasets,
        config: SafeSynthesizerParameters | None = None,
    ) -> AutocorrelationSimilarity:
        """Evaluate autocorrelation fidelity for paired time-series datasets.

        Explicit metric configuration takes precedence over automatic
        time-series enablement. Evaluation failures are isolated to this
        component and returned in ``score.notes`` instead of aborting the full
        evaluation pipeline.

        Args:
            evaluation_datasets: Training and synthetic datasets to compare.
            config: Optional pipeline and metric configuration.

        Returns:
            A component containing the score, diagnostic details, and notes.
        """
        cfg = AutocorrelationSimilarity._resolve_config(config)
        if not AutocorrelationSimilarity._is_enabled(cfg, config):
            return AutocorrelationSimilarity(score=EvaluationScore(notes="Autocorrelation Similarity is disabled."))

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
    def _is_enabled(
        cfg: AutocorrelationSimilarityParameters,
        config: SafeSynthesizerParameters | None,
    ) -> bool:
        """Resolve an explicit enable flag before automatic time-series enablement."""
        if cfg.enabled is not None:
            return cfg.enabled
        return bool(config and config.time_series.is_timeseries)

    @staticmethod
    def _evaluate(
        datasets: EvaluationDatasets,
        cfg: AutocorrelationSimilarityParameters,
        config: SafeSynthesizerParameters | None,
    ) -> AutocorrelationSimilarity:
        """Compute atomic profiles and aggregate them into a component result.

        Args:
            datasets: Training and synthetic datasets to compare.
            cfg: Resolved autocorrelation metric parameters.
            config: Optional top-level configuration used for inherited columns.

        Returns:
            A scored component, or an unavailable component with explanatory
            notes when no usable comparison remains.
        """
        timestamp_column = config.time_series.timestamp_column if config is not None else None
        inherited_group_column = config.data.group_training_examples_by if config is not None else None
        group_column = cfg.group_column or inherited_group_column
        if cfg.group_column is None and inherited_group_column == PSEUDO_GROUP_COLUMN:
            # Training injects this reserved column to reuse grouped sequence
            # infrastructure, but evaluation receives frames with it removed.
            group_column = None
        elif group_column is not None and (
            group_column not in datasets.training or group_column not in datasets.synthetic
        ):
            return AutocorrelationSimilarity(
                score=EvaluationScore(notes=f"Configured group column {group_column!r} is missing from a dataset.")
            )

        columns = AutocorrelationSimilarity._numeric_columns(datasets, cfg, timestamp_column, group_column)
        if not columns:
            return AutocorrelationSimilarity(score=EvaluationScore(notes="No shared numeric value columns."))

        groups, missing_training, missing_synthetic = AutocorrelationSimilarity._shared_groups(
            datasets.training,
            datasets.synthetic,
            group_column,
            cfg.max_groups,
        )
        if not groups:
            return AutocorrelationSimilarity(score=EvaluationScore(notes="No shared groups to evaluate."))

        atomics: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for group_value in groups:
            training_group = AutocorrelationSimilarity._group_frame(
                datasets.training, group_column, group_value, timestamp_column
            )
            synthetic_group = AutocorrelationSimilarity._group_frame(
                datasets.synthetic, group_column, group_value, timestamp_column
            )
            for column in columns:
                result, reason = AutocorrelationSimilarity._atomic_score(
                    training_group[column], synthetic_group[column], cfg
                )
                group_label = None if group_column is None else str(group_value)
                if result is None:
                    skipped.append({"group": group_label, "column": column, "reason": reason})
                    continue
                atomics.append({"group": group_label, "column": column, **result})

        if not atomics:
            return AutocorrelationSimilarity(
                score=EvaluationScore(notes="No usable group/column autocorrelation profiles."),
                details={"skipped": skipped},
            )

        similarity = float(np.mean([item["similarity"] for item in atomics]))
        score = EvaluationScore.finalize_grade(raw_score=similarity, score=10.0 * similarity)
        if skipped:
            score.notes = f"Skipped {len(skipped)} unusable group/column comparisons."

        details = {
            "evaluation_mode": "per_group" if group_column else "global",
            "timestamp_column": timestamp_column,
            "group_column": group_column,
            "max_lag": cfg.max_lag,
            "min_points": cfg.min_points,
            "counts": {
                "groups": len(groups),
                "columns": len(columns),
                "atomic_scores": len(atomics),
                "skipped": len(skipped),
            },
            "per_group": AutocorrelationSimilarity._summaries(atomics, "group"),
            "per_column": AutocorrelationSimilarity._summaries(atomics, "column"),
            "atomics": atomics,
            "skipped": skipped,
            "groups_only_in_reference": missing_training,
            "groups_only_in_synthetic": missing_synthetic,
        }
        return AutocorrelationSimilarity(score=score, details=details)

    @staticmethod
    def _numeric_columns(
        datasets: EvaluationDatasets,
        cfg: AutocorrelationSimilarityParameters,
        timestamp_column: str | None,
        group_column: str | None,
    ) -> list[str]:
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
            Shared numeric value column names to evaluate.
        """
        if cfg.value_columns is not None:
            return [
                column
                for column in cfg.value_columns
                if column in datasets.training
                and column in datasets.synthetic
                and pd.api.types.is_numeric_dtype(datasets.training[column])
                and pd.api.types.is_numeric_dtype(datasets.synthetic[column])
                and column not in {timestamp_column, group_column, PSEUDO_GROUP_COLUMN}
            ]
        numeric = set(datasets.get_columns_of_type({FieldType.NUMERIC}, based_on="both"))
        numeric.difference_update(filter(None, [timestamp_column, group_column, PSEUDO_GROUP_COLUMN]))
        return sorted(numeric)

    @staticmethod
    def _shared_groups(
        training: pd.DataFrame,
        synthetic: pd.DataFrame,
        group_column: str | None,
        max_groups: int,
    ) -> tuple[list[Any], list[str], list[str]]:
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
            Shared group keys, training-only labels, and synthetic-only labels.
        """
        if group_column is None:
            return [None], [], []
        if group_column not in training or group_column not in synthetic:
            return [], [], []
        training_groups = set(training[group_column].dropna().unique())
        synthetic_groups = set(synthetic[group_column].dropna().unique())
        shared = sorted(training_groups & synthetic_groups, key=str)[:max_groups]
        only_training = [str(value) for value in sorted(training_groups - synthetic_groups, key=str)]
        only_synthetic = [str(value) for value in sorted(synthetic_groups - training_groups, key=str)]
        return shared, only_training, only_synthetic

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
    def _atomic_score(
        training: pd.Series,
        synthetic: pd.Series,
        cfg: AutocorrelationSimilarityParameters,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Compare one training and synthetic autocorrelation profile.

        The usable length is the shorter finite sequence. Lags are capped at
        half that length so each correlation retains substantial overlap. The
        mean profile difference is divided by two because autocorrelation lies
        in ``[-1, 1]`` and the largest possible lag-level difference is two.

        Args:
            training: Ordered training values for one group and column.
            synthetic: Ordered synthetic values for the same group and column.
            cfg: Parameters controlling minimum length and maximum lag.

        Returns:
            Atomic score details and ``None`` on success, or ``None`` and a
            human-readable skip reason when the profile is unusable.
        """
        training_values = training.dropna().to_numpy(dtype=float)
        synthetic_values = synthetic.dropna().to_numpy(dtype=float)
        training_values = training_values[np.isfinite(training_values)]
        synthetic_values = synthetic_values[np.isfinite(synthetic_values)]
        n = min(len(training_values), len(synthetic_values))
        if n < cfg.min_points:
            return None, f"fewer than {cfg.min_points} points"
        if np.std(training_values) <= 1e-12 or np.std(synthetic_values) <= 1e-12:
            return None, "constant or near-constant series"

        # Retaining at least half of the shorter sequence at every lag avoids
        # presenting correlations based on only a small tail of observations.
        effective_max_lag = min(cfg.max_lag, (n - 1) // 2)
        if effective_max_lag < 1:
            return None, "no stable lags"
        reference_acf = AutocorrelationSimilarity._acf_vector(training_values, effective_max_lag)
        synthetic_acf = AutocorrelationSimilarity._acf_vector(synthetic_values, effective_max_lag)
        # ACF values are bounded by -1 and 1. Dividing their absolute
        # difference by two maps the theoretical maximum error to one.
        error = float(np.mean(np.abs(reference_acf - synthetic_acf)) / 2.0)
        error = float(np.clip(error, 0.0, 1.0))
        return {
            "effective_max_lag": effective_max_lag,
            "error": round(error, 6),
            "similarity": round(1.0 - error, 6),
            "reference_acf": np.round(reference_acf, 6).tolist(),
            "synthetic_acf": np.round(synthetic_acf, 6).tolist(),
        }, None

    @staticmethod
    def _acf_vector(values: NDArray[np.float64], max_lag: int) -> NDArray[np.float64]:
        """Compute a consistently normalized autocorrelation vector.

        The estimator centers the full sequence once and uses
        ``n * population_variance`` as the denominator at every lag. Keeping a
        common denominator makes lag profiles directly comparable and yields
        the bounded, gradually tapered form used by the metric.

        Args:
            values: Finite, nonconstant values in temporal order.
            max_lag: Largest positive lag to include.

        Returns:
            Autocorrelation values for lags 1 through ``max_lag``.
        """
        centered = values - np.mean(values)
        variance = float(np.var(centered))
        return np.array(
            [np.dot(centered[:-lag], centered[lag:]) / (len(centered) * variance) for lag in range(1, max_lag + 1)]
        )

    @staticmethod
    def _summaries(atomics: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        """Average atomic similarities by a diagnostic key.

        Args:
            atomics: Successful group-and-column comparison details.
            key: Detail key to group by, such as ``group`` or ``column``.

        Returns:
            Deterministically ordered summaries with similarity and count.
        """
        scores: defaultdict[Any, list[float]] = defaultdict(list)
        for item in atomics:
            scores[item[key]].append(item["similarity"])
        return [
            {key: value, "similarity": round(float(np.mean(values)), 6), "count": len(values)}
            for value, values in sorted(scores.items(), key=lambda item: str(item[0]))
        ]
