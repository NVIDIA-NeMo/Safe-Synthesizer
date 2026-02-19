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


class ADFStationarity(Component):
    """Augmented Dickey-Fuller stationarity test for time series pairs.

    Tests whether each series has a unit root (non-stationary):

    * p-value < 0.05 -- stationary (can reject unit root)
    * p-value >= 0.05 -- non-stationary (has trend / unit root)

    The score reflects whether the original and synthetic series agree on
    stationarity status.  This is an *informational* metric and is typically
    not included in the weighted similarity score.
    """

    name: str = Field(default="ADF Stationarity")
    details: dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def jinja_context(self):
        ctx = super().jinja_context
        ctx["anchor_link"] = "#adf-stationarity"
        ctx["details"] = self.details
        return ctx

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None = None
    ) -> ADFStationarity:
        ts_eval_config = get_time_series_config(config)
        is_timeseries = config.time_series.is_timeseries if config and getattr(config, "time_series", None) else False

        if not ts_eval_config and not is_timeseries:
            return ADFStationarity(score=EvaluationScore())

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
                return ADFStationarity(score=EvaluationScore(notes="No numeric columns available."))

            per_column = []
            matches = []
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

                result = ADFStationarity._stationarity_match(ref_values, syn_values)
                matches.append(result["stationarity_match"])
                per_column.append(
                    {
                        "column": col,
                        "original": result["original"],
                        "synthetic": result["synthetic"],
                        "stationarity_match": result["stationarity_match"],
                    }
                )

            if not matches:
                return ADFStationarity(score=EvaluationScore(notes="No valid columns for ADF test."))

            match_rate = sum(1 for m in matches if m) / len(matches)
            evaluation_score = EvaluationScore.finalize_grade(raw_score=match_rate, score=match_rate * 10)
            details = {
                "columns": per_column,
                "match_rate": round(match_rate, 4),
            }
            return ADFStationarity(score=evaluation_score, details=details)

        except Exception as exc:
            logger.exception("Failed to compute ADF stationarity metric.")
            return ADFStationarity(score=EvaluationScore(notes=str(exc)))

    @staticmethod
    def from_arrays(original: NDArray, synthetic: NDArray) -> ADFStationarity:
        """Convenience entry point for raw NumPy arrays (used by notebooks and tests)."""
        original = np.asarray(original, dtype=float)
        synthetic = np.asarray(synthetic, dtype=float)

        result = ADFStationarity._stationarity_match(original, synthetic)
        match = result["stationarity_match"]
        score_val = 1.0 if match else 0.0

        evaluation_score = EvaluationScore.finalize_grade(raw_score=score_val, score=score_val * 10)
        details = {
            "original": result["original"],
            "synthetic": result["synthetic"],
            "stationarity_match": match,
        }
        return ADFStationarity(score=evaluation_score, details=details)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    @staticmethod
    def _adf_test(series: NDArray, max_lags: int | None = None) -> dict:
        """Run an Augmented Dickey-Fuller test.

        Uses OLS regression: delta_y_t = alpha + beta*y_{t-1} + sum gamma_i*delta_y_{t-i} + eps_t
        Tests H0: beta = 0 (unit root exists).

        Returns dict with test_statistic, p_value_approx, is_stationary.
        """
        y = np.asarray(series, dtype=float)
        if y.size < 10:
            return {"test_statistic": np.nan, "p_value_approx": np.nan, "is_stationary": None}

        dy = np.diff(y)
        if max_lags is None:
            max_lags = min(int(np.floor(12 * (len(y) / 100) ** 0.25)), len(dy) // 3)
        max_lags = max(max_lags, 1)

        n = len(dy)
        # Build regression matrix: [y_{t-1}, delta_y_{t-1}, ..., delta_y_{t-p}, 1]
        dy_dep = dy[max_lags:]  # delta_y_t, length = n - max_lags
        y_lag = y[max_lags : max_lags + len(dy_dep)]  # y_{t-1}, same length

        X_parts = [y_lag[:, None]]  # y_{t-1}
        for lag in range(1, max_lags + 1):
            X_parts.append(dy[max_lags - lag : n - lag, None])  # delta_y_{t-lag}
        X_parts.append(np.ones((len(dy_dep), 1)))  # intercept
        X = np.hstack(X_parts)

        # OLS
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X, dy_dep, rcond=None)
        except np.linalg.LinAlgError:
            return {"test_statistic": np.nan, "p_value_approx": np.nan, "is_stationary": None}

        resid = dy_dep - X @ beta
        sigma2 = np.sum(resid**2) / (len(resid) - X.shape[1])

        # Standard error of the beta coefficient for y_{t-1}
        try:
            cov_matrix = sigma2 * np.linalg.inv(X.T @ X)
            se_beta = np.sqrt(cov_matrix[0, 0])
        except np.linalg.LinAlgError:
            return {"test_statistic": np.nan, "p_value_approx": np.nan, "is_stationary": None}

        t_stat = beta[0] / se_beta

        # Approximate p-value using MacKinnon critical values for 'c' (constant) model
        # Critical values for N=inf: 1% = -3.43, 5% = -2.86, 10% = -2.57
        if t_stat < -3.43:
            p_approx = 0.01
        elif t_stat < -2.86:
            p_approx = 0.05
        elif t_stat < -2.57:
            p_approx = 0.10
        else:
            p_approx = 0.50  # cannot reject unit root

        return {
            "test_statistic": float(t_stat),
            "p_value_approx": p_approx,
            "is_stationary": p_approx < 0.05,
        }

    @staticmethod
    def _stationarity_match(original: NDArray, synthetic: NDArray) -> dict:
        """Compare ADF stationarity results between two series."""
        adf_orig = ADFStationarity._adf_test(original)
        adf_syn = ADFStationarity._adf_test(synthetic)
        match = adf_orig["is_stationary"] == adf_syn["is_stationary"]
        return {
            "original": adf_orig,
            "synthetic": adf_syn,
            "stationarity_match": match,
        }
