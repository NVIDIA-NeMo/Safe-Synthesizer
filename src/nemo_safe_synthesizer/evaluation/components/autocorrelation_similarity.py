# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...observability import get_logger
from .component import Component

logger = get_logger(__name__)


class AutocorrelationSimilarity(Component):
    """Compare bounded autocorrelation profiles for numeric time-series channels."""

    name: str = Field(default="Autocorrelation Similarity")
    details: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def from_evaluation_datasets(
        evaluation_datasets: EvaluationDatasets,
        config: SafeSynthesizerParameters | None = None,
    ) -> AutocorrelationSimilarity:
        cfg = AutocorrelationSimilarity._resolve_config(config)
        if not AutocorrelationSimilarity._is_enabled(cfg, config):
            return AutocorrelationSimilarity(score=EvaluationScore(notes="Autocorrelation Similarity is disabled."))

        try:
            return AutocorrelationSimilarity._evaluate(evaluation_datasets, cfg, config)
        except Exception as exc:
            logger.exception("Failed to compute Autocorrelation Similarity.")
            return AutocorrelationSimilarity(score=EvaluationScore(notes=str(exc)))

    @staticmethod
    def _resolve_config(config: SafeSynthesizerParameters | None) -> AutocorrelationSimilarityParameters:
        if config is None:
            return AutocorrelationSimilarityParameters()
        return config.evaluation.time_series.autocorrelation

    @staticmethod
    def _is_enabled(
        cfg: AutocorrelationSimilarityParameters,
        config: SafeSynthesizerParameters | None,
    ) -> bool:
        if cfg.enabled is not None:
            return cfg.enabled
        return bool(config and config.time_series.is_timeseries)

    @staticmethod
    def _evaluate(
        datasets: EvaluationDatasets,
        cfg: AutocorrelationSimilarityParameters,
        config: SafeSynthesizerParameters | None,
    ) -> AutocorrelationSimilarity:
        timestamp_column = cfg.timestamp_column or (config.time_series.timestamp_column if config is not None else None)
        group_column = cfg.group_column or (config.data.group_training_examples_by if config is not None else None)
        columns = AutocorrelationSimilarity._numeric_columns(datasets, cfg, timestamp_column, group_column)
        if not columns:
            return AutocorrelationSimilarity(score=EvaluationScore(notes="No shared numeric value columns."))

        groups, missing_reference, missing_synthetic = AutocorrelationSimilarity._shared_groups(
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
            reference_group = AutocorrelationSimilarity._group_frame(
                datasets.training, group_column, group_value, timestamp_column
            )
            synthetic_group = AutocorrelationSimilarity._group_frame(
                datasets.synthetic, group_column, group_value, timestamp_column
            )
            for column in columns:
                result, reason = AutocorrelationSimilarity._atomic_score(
                    reference_group[column], synthetic_group[column], cfg
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
            "groups_only_in_reference": missing_reference,
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
        if cfg.value_columns is not None:
            return [
                column
                for column in cfg.value_columns
                if column in datasets.training
                and column in datasets.synthetic
                and pd.api.types.is_numeric_dtype(datasets.training[column])
                and pd.api.types.is_numeric_dtype(datasets.synthetic[column])
                and column not in {timestamp_column, group_column}
            ]
        numeric = set(datasets.get_columns_of_type({FieldType.NUMERIC}, based_on="both"))
        numeric.difference_update(filter(None, [timestamp_column, group_column]))
        return sorted(numeric)

    @staticmethod
    def _shared_groups(
        reference: pd.DataFrame,
        synthetic: pd.DataFrame,
        group_column: str | None,
        max_groups: int,
    ) -> tuple[list[Any], list[str], list[str]]:
        if group_column is None:
            return [None], [], []
        if group_column not in reference or group_column not in synthetic:
            return [], [], []
        reference_groups = set(reference[group_column].dropna().unique())
        synthetic_groups = set(synthetic[group_column].dropna().unique())
        shared = sorted(reference_groups & synthetic_groups, key=str)[:max_groups]
        only_reference = [str(value) for value in sorted(reference_groups - synthetic_groups, key=str)]
        only_synthetic = [str(value) for value in sorted(synthetic_groups - reference_groups, key=str)]
        return shared, only_reference, only_synthetic

    @staticmethod
    def _group_frame(
        df: pd.DataFrame,
        group_column: str | None,
        group_value: Any,
        timestamp_column: str | None,
    ) -> pd.DataFrame:
        frame = df if group_column is None else df[df[group_column] == group_value]
        if timestamp_column and timestamp_column in frame:
            frame = frame.sort_values(timestamp_column, kind="mergesort")
        return frame

    @staticmethod
    def _atomic_score(
        reference: pd.Series,
        synthetic: pd.Series,
        cfg: AutocorrelationSimilarityParameters,
    ) -> tuple[dict[str, Any] | None, str | None]:
        reference_values = reference.dropna().to_numpy(dtype=float)
        synthetic_values = synthetic.dropna().to_numpy(dtype=float)
        n = min(len(reference_values), len(synthetic_values))
        if n < cfg.min_points:
            return None, f"fewer than {cfg.min_points} points"
        if np.std(reference_values) <= 1e-12 or np.std(synthetic_values) <= 1e-12:
            return None, "constant or near-constant series"

        effective_max_lag = min(cfg.max_lag, (n - 1) // 2)
        if effective_max_lag < 1:
            return None, "no stable lags"
        reference_acf = AutocorrelationSimilarity._acf_vector(reference_values, effective_max_lag)
        synthetic_acf = AutocorrelationSimilarity._acf_vector(synthetic_values, effective_max_lag)
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
        centered = values - np.mean(values)
        variance = float(np.var(centered))
        return np.array(
            [np.dot(centered[:-lag], centered[lag:]) / (len(centered) * variance) for lag in range(1, max_lag + 1)]
        )

    @staticmethod
    def _summaries(atomics: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        scores: defaultdict[Any, list[float]] = defaultdict(list)
        for item in atomics:
            scores[item[key]].append(item["similarity"])
        return [
            {key: value, "similarity": round(float(np.mean(values)), 6), "count": len(values)}
            for value, values in sorted(scores.items(), key=lambda item: str(item[0]))
        ]
