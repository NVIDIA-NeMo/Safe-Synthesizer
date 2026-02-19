from __future__ import annotations

from functools import cached_property
from typing import Iterable

import numpy as np
import pandas as pd
from nemo_safe_synthesizer.config.evaluate import AutocorrelationSimilarityParameters, TimeSeriesEvaluationParameters
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.evaluation.components.component import Component
from nemo_safe_synthesizer.evaluation.data_model.evaluation_dataset import EvaluationDataset
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import EvaluationScore
from nemo_safe_synthesizer.observability import get_logger
from pydantic import ConfigDict, Field

logger = get_logger(__name__)


def get_time_series_config(config: SafeSynthesizerParameters | None) -> TimeSeriesEvaluationParameters | None:
    if config is None or getattr(config, "evaluation", None) is None:
        return None
    ts_config = getattr(config.evaluation, "time_series", None)
    if ts_config is None or not ts_config.enabled:
        return None
    return ts_config


def sort_time_series(
    df: pd.DataFrame, timestamp_column: str | None = None, group_column: str | None = None
) -> pd.DataFrame:
    if timestamp_column and timestamp_column in df.columns:
        sort_cols: list[str] = [timestamp_column]
        if group_column and group_column in df.columns:
            sort_cols = [group_column, timestamp_column]
        return df.sort_values(by=sort_cols, kind="mergesort", ignore_index=True)
    return df.reset_index(drop=True)


def resolve_columns(
    df: pd.DataFrame, candidate_columns: Iterable[str] | None, exclude: set[str] | None = None
) -> list[str]:
    exclude = exclude or set()
    if candidate_columns:
        return [col for col in candidate_columns if col in df.columns and col not in exclude]
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if numeric_cols:
        return [col for col in numeric_cols if col not in exclude]
    # Fallback to all columns except excluded ones.
    return [col for col in df.columns if col not in exclude]


class AutocorrelationSimilarity(Component):
    name: str = Field(default="Autocorrelation Similarity")
    details: dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def jinja_context(self):
        ctx = super().jinja_context
        ctx["anchor_link"] = "#acf-similarity"
        ctx["details"] = self.details
        return ctx

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None = None
    ) -> AutocorrelationSimilarity:
        ts_eval_config = get_time_series_config(config)
        acf_cfg: AutocorrelationSimilarityParameters | None = (
            getattr(ts_eval_config, "autocorrelation", None) if ts_eval_config else None
        )

        # Check if explicitly enabled via evaluation config
        explicitly_enabled = acf_cfg and acf_cfg.enabled

        # Check if auto-enabled via is_timeseries flag
        is_timeseries = config.time_series.is_timeseries if config and getattr(config, "time_series", None) else False

        if not explicitly_enabled and not is_timeseries:
            return AutocorrelationSimilarity(score=EvaluationScore())

        # If auto-enabled via is_timeseries, create config from main time_series settings
        if not explicitly_enabled and is_timeseries:
            ts_main = config.time_series
            acf_cfg = AutocorrelationSimilarityParameters(
                enabled=True,
                timestamp_column=ts_main.timestamp_column,
            )

        try:
            details, average_similarity = AutocorrelationSimilarity._evaluate(evaluation_dataset, acf_cfg)
            evaluation_score = EvaluationScore.finalize_grade(
                raw_score=average_similarity, score=average_similarity * 10
            )
            return AutocorrelationSimilarity(score=evaluation_score, details=details)
        except Exception as exc:
            logger.exception("Failed to compute autocorrelation similarity metric.")
            return AutocorrelationSimilarity(score=EvaluationScore(notes=str(exc)))

    @staticmethod
    def _evaluate(
        evaluation_dataset: EvaluationDataset, cfg: AutocorrelationSimilarityParameters
    ) -> tuple[dict, float]:
        reference = evaluation_dataset.reference
        synthetic = evaluation_dataset.output

        timestamp_col = cfg.timestamp_column
        group_col = cfg.group_column
        max_lag = cfg.max_lag

        # Debug logging
        logger.info(f"[ACF] Reference shape: {reference.shape}, columns: {list(reference.columns)}")
        logger.info(f"[ACF] Synthetic shape: {synthetic.shape}, columns: {list(synthetic.columns)}")
        logger.info(f"[ACF] Config - group_col: {group_col}, timestamp_col: {timestamp_col}")
        if group_col:
            logger.info(f"[ACF] group_col '{group_col}' in reference: {group_col in reference.columns}")
            logger.info(f"[ACF] group_col '{group_col}' in synthetic: {group_col in synthetic.columns}")

        columns = resolve_columns(reference, cfg.value_columns, exclude=set(filter(None, [timestamp_col, group_col])))
        if not columns:
            raise ValueError("No numeric columns available for autocorrelation comparison.")

        # Check if we should do per-group evaluation
        if group_col and group_col in reference.columns and group_col in synthetic.columns:
            logger.info(f"[ACF] Running PER-GROUP evaluation with group_col='{group_col}'")
            return AutocorrelationSimilarity._evaluate_per_group(
                reference, synthetic, columns, timestamp_col, group_col, max_lag, cfg.distance_metric
            )
        else:
            # Original behavior: global evaluation without grouping
            logger.info("[ACF] Running GLOBAL evaluation (no grouping)")
            return AutocorrelationSimilarity._evaluate_global(
                reference, synthetic, columns, timestamp_col, None, max_lag, cfg.distance_metric
            )

    @staticmethod
    def _evaluate_per_group(
        reference: pd.DataFrame,
        synthetic: pd.DataFrame,
        columns: list[str],
        timestamp_col: str | None,
        group_col: str,
        max_lag: int,
        distance_metric: str,
    ) -> tuple[dict, float]:
        """Evaluate autocorrelation similarity per group and compute average score."""

        # Get unique groups from both datasets
        ref_groups = set(reference[group_col].dropna().unique())
        syn_groups = set(synthetic[group_col].dropna().unique())

        # Find common groups, groups only in reference, and groups only in synthetic
        common_groups = ref_groups & syn_groups
        groups_only_in_reference = ref_groups - syn_groups
        groups_only_in_synthetic = syn_groups - ref_groups

        if not common_groups:
            raise ValueError(
                f"No common groups found between reference and synthetic datasets for column '{group_col}'."
            )

        # Compute similarity for each common group
        per_group_results = []
        group_scores = []

        for group_value in sorted(common_groups, key=str):
            ref_group_df = reference[reference[group_col] == group_value]
            syn_group_df = synthetic[synthetic[group_col] == group_value]

            try:
                group_details, group_similarity = AutocorrelationSimilarity._evaluate_single_group(
                    ref_group_df, syn_group_df, columns, timestamp_col, max_lag, distance_metric, group_value
                )
                group_scores.append(group_similarity)
                per_group_results.append(
                    {
                        "group": str(group_value),
                        "score": round(group_similarity, 4),
                        "ref_rows": len(ref_group_df),
                        "syn_rows": len(syn_group_df),
                        "columns": group_details.get("columns", []),
                    }
                )

                logger.debug(
                    f"Group '{group_value}': similarity={group_similarity:.4f} (ref_rows={len(ref_group_df)}, syn_rows={len(syn_group_df)})"
                )

            except Exception as e:
                logger.warning(f"Failed to compute ACF similarity for group '{group_value}': {e}")
                per_group_results.append(
                    {
                        "group": str(group_value),
                        "score": None,
                        "ref_rows": len(ref_group_df),
                        "syn_rows": len(syn_group_df),
                        "error": str(e),
                    }
                )

        if not group_scores:
            raise ValueError("Unable to compute autocorrelation similarity for any group.")

        # Compute final score as average of group scores
        average_similarity = float(np.mean(group_scores))

        details = {
            "max_lag": max_lag,
            "distance_metric": distance_metric,
            "group_column": group_col,
            "evaluation_mode": "per_group",
            "summary": {
                "total_groups_reference": len(ref_groups),
                "total_groups_synthetic": len(syn_groups),
                "common_groups": len(common_groups),
                "evaluated_groups": len(group_scores),
                "average_score": round(average_similarity, 4),
                "min_score": round(min(group_scores), 4),
                "max_score": round(max(group_scores), 4),
                "std_score": round(float(np.std(group_scores)), 4),
            },
            "per_group": per_group_results,
            "groups_only_in_reference": [str(g) for g in sorted(groups_only_in_reference, key=str)],
            "groups_only_in_synthetic": [str(g) for g in sorted(groups_only_in_synthetic, key=str)],
        }

        return details, average_similarity

    @staticmethod
    def _evaluate_single_group(
        ref_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        columns: list[str],
        timestamp_col: str | None,
        max_lag: int,
        distance_metric: str,
        group_value: str,
    ) -> tuple[dict, float]:
        """Evaluate autocorrelation similarity for a single group."""

        per_column = []
        normalized_diffs = []

        for column in columns:
            if column not in ref_df.columns or column not in syn_df.columns:
                continue

            ref_acf = AutocorrelationSimilarity._autocorrelation_single(ref_df, column, timestamp_col, max_lag)
            syn_acf = AutocorrelationSimilarity._autocorrelation_single(syn_df, column, timestamp_col, max_lag)

            if ref_acf.size == 0 or syn_acf.size == 0:
                continue

            diff = AutocorrelationSimilarity._distance(ref_acf, syn_acf, distance_metric)
            normalized_diff = AutocorrelationSimilarity._normalize_distance(diff, max_lag, distance_metric)
            normalized_diffs.append(normalized_diff)

            per_column.append(
                {
                    "column": column,
                    "distance": round(diff, 4),
                    "lag_values": list(range(1, max_lag + 1)),
                    "reference_acf": [round(v, 4) for v in ref_acf],
                    "synthetic_acf": [round(v, 4) for v in syn_acf],
                }
            )

        if not normalized_diffs:
            raise ValueError(f"Unable to compute autocorrelation for group '{group_value}'.")

        similarity = AutocorrelationSimilarity._compute_rms_similarity(normalized_diffs)

        details = {
            "columns": per_column,
        }
        return details, similarity

    @staticmethod
    def _evaluate_global(
        reference: pd.DataFrame,
        synthetic: pd.DataFrame,
        columns: list[str],
        timestamp_col: str | None,
        group_col: str | None,
        max_lag: int,
        distance_metric: str,
    ) -> tuple[dict, float]:
        """Original global evaluation (without per-group breakdown)."""

        per_column = []
        normalized_diffs = []

        for column in columns:
            if column not in reference.columns or column not in synthetic.columns:
                logger.warning("Skipping column %s; not present in both datasets.", column)
                continue
            ref_acf = AutocorrelationSimilarity._autocorrelation(reference, column, timestamp_col, group_col, max_lag)
            syn_acf = AutocorrelationSimilarity._autocorrelation(synthetic, column, timestamp_col, group_col, max_lag)
            if ref_acf.size == 0 or syn_acf.size == 0:
                continue

            diff = AutocorrelationSimilarity._distance(ref_acf, syn_acf, distance_metric)
            normalized_diff = AutocorrelationSimilarity._normalize_distance(diff, max_lag, distance_metric)
            normalized_diffs.append(normalized_diff)

            per_column.append(
                {
                    "column": column,
                    "distance": round(diff, 4),
                    "lag_values": list(range(1, max_lag + 1)),
                    "reference_acf": [round(v, 4) for v in ref_acf],
                    "synthetic_acf": [round(v, 4) for v in syn_acf],
                }
            )

        if not normalized_diffs:
            raise ValueError("Unable to compute autocorrelation similarity for provided columns.")

        similarity = AutocorrelationSimilarity._compute_rms_similarity(normalized_diffs)

        details = {
            "max_lag": max_lag,
            "distance_metric": distance_metric,
            "evaluation_mode": "global",
            "columns": per_column,
        }
        return details, similarity

    @staticmethod
    def _autocorrelation_single(
        df: pd.DataFrame, column: str, timestamp_column: str | None, max_lag: int
    ) -> np.ndarray:
        """Compute autocorrelation for a single series (no group averaging)."""
        subset_cols = [column]
        if timestamp_column and timestamp_column in df.columns:
            subset_cols.append(timestamp_column)
        ordered = sort_time_series(df[subset_cols], timestamp_column, None)
        series = ordered[column].dropna().astype(float)
        if series.empty:
            return np.zeros(max_lag)
        return AutocorrelationSimilarity._acf_vector(series.to_numpy(), max_lag)

    @staticmethod
    def _autocorrelation(
        df: pd.DataFrame, column: str, timestamp_column: str | None, group_column: str | None, max_lag: int
    ) -> np.ndarray:
        subset_cols = [column]
        if timestamp_column and timestamp_column in df.columns:
            subset_cols.append(timestamp_column)
        if group_column and group_column in df.columns:
            subset_cols.append(group_column)
        ordered = sort_time_series(df[subset_cols], timestamp_column, group_column)
        series = ordered[column].dropna().astype(float)
        if series.empty:
            return np.zeros(max_lag)

        if group_column and group_column in ordered.columns:
            grouped = ordered[[column, group_column]].dropna().groupby(group_column, sort=False)
            acfs = []
            for _, grp in grouped:
                grp_vals = grp[column].astype(float).to_numpy()
                if grp_vals.size > 1:
                    acfs.append(AutocorrelationSimilarity._acf_vector(grp_vals, max_lag))
            if acfs:
                return np.nanmean(acfs, axis=0)
        return AutocorrelationSimilarity._acf_vector(series.to_numpy(), max_lag)

    @staticmethod
    def _acf_vector(values: np.ndarray, max_lag: int) -> np.ndarray:
        # Mean-centering the data
        values = values - np.mean(values)

        # Calculate the variance (autocovariance at lag 0), which uses a 1/N divisor by default for np.var
        var = np.var(values)

        if var == 0 or values.size <= 1:
            return np.zeros(max_lag)

        N = values.size  # Length of the time series
        acf = []
        for lag in range(1, max_lag + 1):
            if lag >= N:
                acf.append(0.0)
                continue

            # Numerator: Sum of products of lagged values
            numerator = np.dot(values[:-lag], values[lag:])

            # The standard biased ACF estimator divides the numerator by N and the variance by N,
            # which simplifies to dividing the numerator by N * var.
            acf.append(float(numerator / (N * var)))

        return np.clip(np.array(acf), -1.0, 1.0)

    @staticmethod
    def _distance(ref: np.ndarray, syn: np.ndarray, metric: str) -> float:
        diff = ref - syn
        if metric == "mae":
            return float(np.mean(np.abs(diff)))
        return float(np.linalg.norm(diff))

    @staticmethod
    def _normalize_distance(diff: float, max_lag: int, distance_metric: str) -> float:
        """Normalize distance to [0, 1] range based on metric type.

        Args:
            diff: Raw distance value.
            max_lag: Maximum lag used for ACF computation.
            distance_metric: Either "mae" or "euclidean".

        Returns:
            Normalized distance clamped to [0, 1].
        """
        if distance_metric == "mae":
            max_diff = 1.0  # realistic max for MAE between ACF vectors
        else:
            max_diff = np.sqrt(max_lag)  # realistic max for Euclidean
        return min(1.0, diff / max_diff)

    @staticmethod
    def _compute_rms_similarity(normalized_diffs: list[float]) -> float:
        """Compute similarity score from normalized differences using RMS.

        Uses RMS instead of mean so large differences are penalized more heavily.

        Args:
            normalized_diffs: List of normalized distance values.

        Returns:
            Similarity score in [0, 1] range.
        """
        rms_diff = float(np.sqrt(np.mean(np.square(normalized_diffs))))
        return max(0.0, 1.0 - rms_diff)
