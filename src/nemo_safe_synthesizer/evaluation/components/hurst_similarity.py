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
from scipy import stats as scipy_stats

logger = get_logger(__name__)


class HurstSimilarity(Component):
    """Measures trend persistence / memory similarity via the Hurst exponent.

    The Hurst exponent estimated through the Rescaled Range (R/S) method captures
    long-term memory:

    * H > 0.5 -- persistent (trending)
    * H ~ 0.5 -- random walk
    * H < 0.5 -- mean-reverting (anti-persistent)

    The similarity score is ``1 - |H_orig - H_syn|``.
    """

    name: str = Field(default="Hurst Similarity")
    details: dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def jinja_context(self):
        ctx = super().jinja_context
        ctx["anchor_link"] = "#hurst-similarity"
        ctx["details"] = self.details
        return ctx

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None = None
    ) -> HurstSimilarity:
        ts_eval_config = get_time_series_config(config)
        is_timeseries = config.time_series.is_timeseries if config and getattr(config, "time_series", None) else False

        if not ts_eval_config and not is_timeseries:
            return HurstSimilarity(score=EvaluationScore())

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
                return HurstSimilarity(score=EvaluationScore(notes="No numeric columns available."))

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

                h_orig = HurstSimilarity._hurst_exponent(ref_values)
                h_syn = HurstSimilarity._hurst_exponent(syn_values)
                error = abs(h_orig - h_syn)
                errors.append(error)
                per_column.append(
                    {
                        "column": col,
                        "hurst_original": round(h_orig, 4),
                        "hurst_synthetic": round(h_syn, 4),
                        "error": round(error, 4),
                    }
                )

            if not errors:
                return HurstSimilarity(score=EvaluationScore(notes="No valid columns for Hurst comparison."))

            avg_error = float(np.mean(errors))
            similarity = max(0.0, 1.0 - avg_error)
            evaluation_score = EvaluationScore.finalize_grade(raw_score=similarity, score=similarity * 10)
            details = {"columns": per_column, "average_error": round(avg_error, 4)}
            return HurstSimilarity(score=evaluation_score, details=details)

        except Exception as exc:
            logger.exception("Failed to compute Hurst similarity metric.")
            return HurstSimilarity(score=EvaluationScore(notes=str(exc)))

    @staticmethod
    def from_arrays(original: NDArray, synthetic: NDArray) -> HurstSimilarity:
        """Convenience entry point for raw NumPy arrays (used by notebooks and tests)."""
        original = np.asarray(original, dtype=float)
        synthetic = np.asarray(synthetic, dtype=float)

        h_orig = HurstSimilarity._hurst_exponent(original)
        h_syn = HurstSimilarity._hurst_exponent(synthetic)
        error = abs(h_orig - h_syn)
        similarity = max(0.0, 1.0 - error)

        evaluation_score = EvaluationScore.finalize_grade(raw_score=similarity, score=similarity * 10)
        details = {
            "hurst_original": round(h_orig, 4),
            "hurst_synthetic": round(h_syn, 4),
            "error": round(error, 4),
        }
        return HurstSimilarity(score=evaluation_score, details=details)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    @staticmethod
    def _hurst_exponent(series: NDArray, min_window: int = 8) -> float:
        """Estimate the Hurst exponent via the Rescaled Range (R/S) method.

        The R/S method is designed for stationary data, so we first-difference
        the series to remove non-stationarity (trends, unit roots).  The Hurst
        exponent of the *increments* tells us about the long-range memory of
        the underlying process:

        * H > 0.5 -- persistent increments (trending process)
        * H ~ 0.5 -- independent increments (random walk)
        * H < 0.5 -- anti-persistent increments (mean-reverting process)

        Uses a range of window sizes and fits log(R/S) ~ H * log(n).
        """
        if len(series) < min_window * 2 + 1:
            return 0.5  # not enough data

        # Work on first differences so the input to R/S is stationary.
        # Without this, integrated series (random walks, trends) inflate H
        # toward 1.0 because cumulative_dev scales as n^{3/2} instead of n^{1/2}.
        series = np.diff(series)
        N = len(series)

        # Generate window sizes: powers of 2 up to N//2
        max_k = int(np.log2(N // 2))
        min_k = int(np.log2(min_window))
        if max_k <= min_k:
            return 0.5

        ns = [2**k for k in range(min_k, max_k + 1)]
        rs_values = []

        for n in ns:
            num_chunks = N // n
            if num_chunks == 0:
                continue
            rs_list = []
            for i in range(num_chunks):
                chunk = series[i * n : (i + 1) * n]
                mean_chunk = np.mean(chunk)
                cumulative_dev = np.cumsum(chunk - mean_chunk)
                R = np.max(cumulative_dev) - np.min(cumulative_dev)
                S = np.std(chunk, ddof=1)
                if S > 0:
                    rs_list.append(R / S)
            if rs_list:
                rs_values.append((np.log(n), np.log(np.mean(rs_list))))

        if len(rs_values) < 2:
            return 0.5

        log_n, log_rs = zip(*rs_values)
        slope, _, _, _, _ = scipy_stats.linregress(log_n, log_rs)
        return float(np.clip(slope, 0.0, 1.0))

    @staticmethod
    def _hurst_error(original: NDArray, synthetic: NDArray) -> float:
        """Absolute difference in Hurst exponents. Range: [0, 1]."""
        return abs(HurstSimilarity._hurst_exponent(original) - HurstSimilarity._hurst_exponent(synthetic))
