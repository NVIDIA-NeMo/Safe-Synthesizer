from __future__ import annotations

from functools import cached_property

import numpy as np
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.evaluation.components.autocorrelation_similarity import (
    get_time_series_config,
    resolve_columns,
    sort_time_series,
)
from nemo_safe_synthesizer.evaluation.components.component import Component
from nemo_safe_synthesizer.evaluation.data_model.evaluation_dataset import EvaluationDataset
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import EvaluationScore
from nemo_safe_synthesizer.observability import get_logger
from numpy.typing import NDArray
from pydantic import ConfigDict, Field

logger = get_logger(__name__)

DEFAULT_ROLLING_WINDOW = 20


class RollingStatsSimilarity(Component):
    """Measures how similar the local mean and variance evolution are between two series.

    Uses RMSE between rolling statistics (mean and standard deviation) of the
    original and synthetic series, normalised to a [0, 1] error range.
    """

    name: str = Field(default="Rolling Stats Similarity")
    details: dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def jinja_context(self):
        ctx = super().jinja_context
        ctx["anchor_link"] = "#rolling-stats-similarity"
        ctx["details"] = self.details
        return ctx

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset,
        config: SafeSynthesizerParameters | None = None,
        rolling_window: int = DEFAULT_ROLLING_WINDOW,
    ) -> RollingStatsSimilarity:
        ts_eval_config = get_time_series_config(config)
        is_timeseries = config.time_series.is_timeseries if config and getattr(config, "time_series", None) else False

        if not ts_eval_config and not is_timeseries:
            return RollingStatsSimilarity(score=EvaluationScore())

        try:
            reference = evaluation_dataset.reference
            synthetic = evaluation_dataset.output

            timestamp_col = None
            group_col = None
            if ts_eval_config:
                acf_cfg = getattr(ts_eval_config, "autocorrelation", None)
                if acf_cfg:
                    timestamp_col = acf_cfg.timestamp_column
                    group_col = acf_cfg.group_column

            columns = resolve_columns(reference, None, exclude=set(filter(None, [timestamp_col, group_col])))
            if not columns:
                return RollingStatsSimilarity(score=EvaluationScore(notes="No numeric columns available."))

            per_column = []
            mean_scores = []
            var_scores = []
            for col in columns:
                if col not in reference.columns or col not in synthetic.columns:
                    continue
                ref_sorted = sort_time_series(
                    reference[[col] + [c for c in [timestamp_col] if c and c in reference.columns]], timestamp_col, None
                )
                syn_sorted = sort_time_series(
                    synthetic[[col] + [c for c in [timestamp_col] if c and c in synthetic.columns]], timestamp_col, None
                )
                ref_values = ref_sorted[col].dropna().astype(float).to_numpy()
                syn_values = syn_sorted[col].dropna().astype(float).to_numpy()

                rm_rmse = RollingStatsSimilarity._rolling_mean_rmse(ref_values, syn_values, rolling_window)
                rv_rmse = RollingStatsSimilarity._rolling_var_rmse(ref_values, syn_values, rolling_window)

                # Normalise using the same scale as the DTW scorecard
                scale_mean = max(np.std(ref_values), np.std(syn_values))
                if scale_mean < 1e-8:
                    scale_mean = abs(np.mean(ref_values) - np.mean(syn_values)) + 1e-8
                scale_var = max(np.std(ref_values), np.std(syn_values), 1e-8)

                rm_score = min(1.0, rm_rmse / scale_mean) if not np.isnan(rm_rmse) else 0.0
                rv_score = min(1.0, rv_rmse / scale_var) if not np.isnan(rv_rmse) else 0.0
                mean_scores.append(rm_score)
                var_scores.append(rv_score)
                per_column.append(
                    {
                        "column": col,
                        "rolling_mean_rmse": round(rm_rmse, 6) if not np.isnan(rm_rmse) else None,
                        "rolling_var_rmse": round(rv_rmse, 6) if not np.isnan(rv_rmse) else None,
                        "rolling_mean_score": round(rm_score, 4),
                        "rolling_var_score": round(rv_score, 4),
                    }
                )

            if not mean_scores:
                return RollingStatsSimilarity(score=EvaluationScore(notes="No valid columns for rolling stats."))

            avg_mean_score = float(np.mean(mean_scores))
            avg_var_score = float(np.mean(var_scores))
            avg_error = (avg_mean_score + avg_var_score) / 2.0
            similarity = max(0.0, 1.0 - avg_error)
            evaluation_score = EvaluationScore.finalize_grade(raw_score=similarity, score=similarity * 10)
            details = {
                "rolling_window": rolling_window,
                "columns": per_column,
                "average_mean_score": round(avg_mean_score, 4),
                "average_var_score": round(avg_var_score, 4),
            }
            return RollingStatsSimilarity(score=evaluation_score, details=details)

        except Exception as exc:
            logger.exception("Failed to compute rolling stats similarity metric.")
            return RollingStatsSimilarity(score=EvaluationScore(notes=str(exc)))

    @staticmethod
    def from_arrays(
        original: NDArray,
        synthetic: NDArray,
        rolling_window: int = DEFAULT_ROLLING_WINDOW,
    ) -> RollingStatsSimilarity:
        """Convenience entry point for raw NumPy arrays (used by notebooks and tests)."""
        original = np.asarray(original, dtype=float)
        synthetic = np.asarray(synthetic, dtype=float)

        rm_rmse = RollingStatsSimilarity._rolling_mean_rmse(original, synthetic, rolling_window)
        rv_rmse = RollingStatsSimilarity._rolling_var_rmse(original, synthetic, rolling_window)

        # Normalise scores to [0, 1] using the same logic as the scorecard
        scale_mean = max(np.std(original), np.std(synthetic))
        if scale_mean < 1e-8:
            scale_mean = abs(np.mean(original) - np.mean(synthetic)) + 1e-8
        scale_var = max(np.std(original), np.std(synthetic), 1e-8)

        rm_score = min(1.0, rm_rmse / scale_mean) if not np.isnan(rm_rmse) else 0.0
        rv_score = min(1.0, rv_rmse / scale_var) if not np.isnan(rv_rmse) else 0.0

        avg_error = (rm_score + rv_score) / 2.0
        similarity = max(0.0, 1.0 - avg_error)

        evaluation_score = EvaluationScore.finalize_grade(raw_score=similarity, score=similarity * 10)
        details = {
            "rolling_window": rolling_window,
            "rolling_mean_rmse": round(rm_rmse, 6) if not np.isnan(rm_rmse) else None,
            "rolling_var_rmse": round(rv_rmse, 6) if not np.isnan(rv_rmse) else None,
            "rolling_mean_score": round(rm_score, 4),
            "rolling_var_score": round(rv_score, 4),
        }
        return RollingStatsSimilarity(score=evaluation_score, details=details)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_stat(series: NDArray, window: int, stat: str = "mean") -> NDArray:
        """Compute rolling mean or std using a simple sliding window (no pandas needed)."""
        if series.size < window:
            return np.array([])
        # Use cumsum trick for rolling mean
        if stat == "mean":
            cs = np.cumsum(series)
            cs = np.insert(cs, 0, 0.0)
            return (cs[window:] - cs[:-window]) / window
        # Rolling std
        out = []
        for i in range(len(series) - window + 1):
            out.append(np.std(series[i : i + window], ddof=1))
        return np.array(out)

    @staticmethod
    def _rolling_mean_rmse(original: NDArray, synthetic: NDArray, window: int = DEFAULT_ROLLING_WINDOW) -> float:
        """RMSE between the rolling means of two series."""
        rm_orig = RollingStatsSimilarity._rolling_stat(original, window, "mean")
        rm_syn = RollingStatsSimilarity._rolling_stat(synthetic, window, "mean")
        min_len = min(len(rm_orig), len(rm_syn))
        if min_len == 0:
            return float("nan")
        diff = rm_orig[:min_len] - rm_syn[:min_len]
        return float(np.sqrt(np.mean(diff**2)))

    @staticmethod
    def _rolling_var_rmse(original: NDArray, synthetic: NDArray, window: int = DEFAULT_ROLLING_WINDOW) -> float:
        """RMSE between the rolling standard deviations of two series."""
        rv_orig = RollingStatsSimilarity._rolling_stat(original, window, "std")
        rv_syn = RollingStatsSimilarity._rolling_stat(synthetic, window, "std")
        min_len = min(len(rv_orig), len(rv_syn))
        if min_len == 0:
            return float("nan")
        diff = rv_orig[:min_len] - rv_syn[:min_len]
        return float(np.sqrt(np.mean(diff**2)))
