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


class DriftSimilarity(Component):
    """Measures trend direction similarity via the mean of first differences.

    The drift mean ``E[y_t - y_{t-1}]`` captures whether a series trends upward,
    downward, or stays flat.  The error between two series is the absolute
    difference of their drift means, normalised by the larger magnitude.
    """

    name: str = Field(default="Drift Similarity")
    details: dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def jinja_context(self):
        ctx = super().jinja_context
        ctx["anchor_link"] = "#drift-similarity"
        ctx["details"] = self.details
        return ctx

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None = None
    ) -> DriftSimilarity:
        ts_eval_config = get_time_series_config(config)
        is_timeseries = config.time_series.is_timeseries if config and getattr(config, "time_series", None) else False

        if not ts_eval_config and not is_timeseries:
            return DriftSimilarity(score=EvaluationScore())

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
                return DriftSimilarity(score=EvaluationScore(notes="No numeric columns available."))

            per_column = []
            errors = []
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

                d_orig = DriftSimilarity._drift_mean(ref_values)
                d_syn = DriftSimilarity._drift_mean(syn_values)
                error = DriftSimilarity._drift_mean_error(ref_values, syn_values)
                errors.append(min(1.0, error))
                per_column.append(
                    {
                        "column": col,
                        "drift_original": round(d_orig, 6),
                        "drift_synthetic": round(d_syn, 6),
                        "error": round(error, 4),
                    }
                )

            if not errors:
                return DriftSimilarity(score=EvaluationScore(notes="No valid columns for drift comparison."))

            avg_error = float(np.mean(errors))
            similarity = max(0.0, 1.0 - avg_error)
            evaluation_score = EvaluationScore.finalize_grade(raw_score=similarity, score=similarity * 10)
            details = {"columns": per_column, "average_error": round(avg_error, 4)}
            return DriftSimilarity(score=evaluation_score, details=details)

        except Exception as exc:
            logger.exception("Failed to compute drift similarity metric.")
            return DriftSimilarity(score=EvaluationScore(notes=str(exc)))

    @staticmethod
    def from_arrays(original: NDArray, synthetic: NDArray) -> DriftSimilarity:
        """Convenience entry point for raw NumPy arrays (used by notebooks and tests)."""
        original = np.asarray(original, dtype=float)
        synthetic = np.asarray(synthetic, dtype=float)

        d_orig = DriftSimilarity._drift_mean(original)
        d_syn = DriftSimilarity._drift_mean(synthetic)
        error = DriftSimilarity._drift_mean_error(original, synthetic)
        error_clamped = min(1.0, error)
        similarity = max(0.0, 1.0 - error_clamped)

        evaluation_score = EvaluationScore.finalize_grade(raw_score=similarity, score=similarity * 10)
        details = {
            "drift_original": round(d_orig, 6),
            "drift_synthetic": round(d_syn, 6),
            "error": round(error_clamped, 4),
        }
        return DriftSimilarity(score=evaluation_score, details=details)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    @staticmethod
    def _drift_mean(series: NDArray) -> float:
        """Mean of first differences (average step change)."""
        if series.size < 2:
            return 0.0
        return float(np.mean(np.diff(series)))

    @staticmethod
    def _drift_mean_error(original: NDArray, synthetic: NDArray) -> float:
        """Absolute difference between drift means, normalized by the larger drift magnitude.

        Returns a value >= 0. Lower is better.
        Uses max(|d_orig|, |d_syn|) as the scale with a reasonable floor (1e-2)
        to prevent near-zero denominators from inflating the error for
        series with little or no net drift (e.g., sinusoids).
        """
        d_orig = DriftSimilarity._drift_mean(original)
        d_syn = DriftSimilarity._drift_mean(synthetic)
        abs_err = abs(d_orig - d_syn)
        # Normalize by the larger drift magnitude (with floor to avoid /0)
        scale = max(abs(d_orig), abs(d_syn), 1e-2)
        return abs_err / scale
