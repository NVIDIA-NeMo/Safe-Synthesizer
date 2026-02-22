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


class DTWSimilarity(Component):
    """Measures overall shape similarity via Dynamic Time Warping.

    DTW finds the optimal alignment between two series by allowing "stretching"
    in time.  The normalised DTW cost is scaled by the series' standard deviation
    to produce a [0, 1] error, and the similarity is ``1 - error``.
    """

    name: str = Field(default="DTW Similarity")
    details: dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def jinja_context(self):
        ctx = super().jinja_context
        ctx["anchor_link"] = "#dtw-similarity"
        ctx["details"] = self.details
        return ctx

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None = None
    ) -> DTWSimilarity:
        ts_eval_config = get_time_series_config(config)
        is_timeseries = config.time_series.is_timeseries if config and getattr(config, "time_series", None) else False

        if not ts_eval_config and not is_timeseries:
            return DTWSimilarity(score=EvaluationScore())

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

            # Fall back to top-level config when ts_eval_config doesn't supply them
            if group_col is None and config and getattr(config, "data", None):
                group_col = getattr(config.data, "group_training_examples_by", None)
            if timestamp_col is None and config and getattr(config, "time_series", None):
                timestamp_col = getattr(config.time_series, "timestamp_column", None)

            columns = resolve_columns(reference, None, exclude=set(filter(None, [timestamp_col, group_col])))
            if not columns:
                return DTWSimilarity(score=EvaluationScore(notes="No numeric columns available."))

            # When group and timestamp columns are known, compare mean series across groups
            # (one representative 47-point series per channel) rather than the full
            # flattened array, which produces meaningless cross-group discontinuities.
            use_mean_series = (
                group_col is not None
                and timestamp_col is not None
                and group_col in reference.columns
                and timestamp_col in reference.columns
                and group_col in synthetic.columns
                and timestamp_col in synthetic.columns
            )

            per_column = []
            scores = []
            for col in columns:
                if col not in reference.columns or col not in synthetic.columns:
                    continue
                if use_mean_series:
                    ref_values = (
                        reference.groupby(timestamp_col)[col].mean().sort_index().dropna().astype(float).to_numpy()
                    )
                    syn_values = (
                        synthetic.groupby(timestamp_col)[col].mean().sort_index().dropna().astype(float).to_numpy()
                    )
                else:
                    ref_sorted = sort_time_series(
                        reference[[col] + [c for c in [timestamp_col] if c and c in reference.columns]],
                        timestamp_col,
                        None,
                    )
                    syn_sorted = sort_time_series(
                        synthetic[[col] + [c for c in [timestamp_col] if c and c in synthetic.columns]],
                        timestamp_col,
                        None,
                    )
                    ref_values = ref_sorted[col].dropna().astype(float).to_numpy()
                    syn_values = syn_sorted[col].dropna().astype(float).to_numpy()

                dtw_raw = DTWSimilarity._dtw_distance(ref_values, syn_values)
                dtw_norm = dtw_raw / np.sqrt(max(len(ref_values), len(syn_values)))
                dtw_score = DTWSimilarity._normalize_dtw_score(ref_values, syn_values, dtw_norm)
                scores.append(dtw_score)
                per_column.append(
                    {
                        "column": col,
                        "dtw_raw": round(dtw_raw, 4),
                        "dtw_normalized": round(dtw_norm, 6),
                        "dtw_score": round(dtw_score, 4),
                    }
                )

            if not scores:
                return DTWSimilarity(score=EvaluationScore(notes="No valid columns for DTW comparison."))

            avg_score = float(np.mean(scores))
            similarity = max(0.0, 1.0 - avg_score)
            evaluation_score = EvaluationScore.finalize_grade(raw_score=similarity, score=similarity * 10)
            details = {"columns": per_column, "average_dtw_score": round(avg_score, 4)}
            return DTWSimilarity(score=evaluation_score, details=details)

        except Exception as exc:
            logger.exception("Failed to compute DTW similarity metric.")
            return DTWSimilarity(score=EvaluationScore(notes=str(exc)))

    @staticmethod
    def from_arrays(original: NDArray, synthetic: NDArray, window: int | None = None) -> DTWSimilarity:
        """Convenience entry point for raw NumPy arrays (used by notebooks and tests)."""
        original = np.asarray(original, dtype=float)
        synthetic = np.asarray(synthetic, dtype=float)

        dtw_raw = DTWSimilarity._dtw_distance(original, synthetic, window=window)
        dtw_norm = dtw_raw / np.sqrt(max(len(original), len(synthetic)))
        dtw_score = DTWSimilarity._normalize_dtw_score(original, synthetic, dtw_norm)
        similarity = max(0.0, 1.0 - dtw_score)

        evaluation_score = EvaluationScore.finalize_grade(raw_score=similarity, score=similarity * 10)
        details = {
            "dtw_raw": round(dtw_raw, 4),
            "dtw_normalized": round(dtw_norm, 6),
            "dtw_score": round(dtw_score, 4),
        }
        return DTWSimilarity(score=evaluation_score, details=details)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    @staticmethod
    def _dtw_distance(original: NDArray, synthetic: NDArray, window: int | None = None) -> float:
        """Compute DTW distance between two 1-D series via aeon's compiled implementation.

        Uses the Sakoe-Chiba band constraint when *window* is set. *window* is expressed
        as an integer number of steps; it is converted to the float fraction expected by
        aeon's API (fraction = window / max(N, M)).
        Returns the raw DTW cost.
        """
        from aeon.distances import dtw_distance as aeon_dtw_distance

        window_frac: float | None = None
        if window is not None:
            window_frac = window / max(len(original), len(synthetic))
        # aeon returns the accumulated squared cost; take sqrt to match the
        # original implementation's scale (sqrt(sum_sq_along_path)).
        return float(np.sqrt(aeon_dtw_distance(original, synthetic, window=window_frac)))

    @staticmethod
    def _dtw_normalized(original: NDArray, synthetic: NDArray, window: int | None = None) -> float:
        """DTW distance normalized by series length so it's comparable across different-length series.

        Uses RMSE-like normalization: ``raw / sqrt(max(N, M))``.  The raw DTW
        cost is ``sqrt(sum_of_squared_diffs)`` which scales as ``d * sqrt(N)``
        for a path of length ~N with average per-step difference *d*.  Dividing
        by ``sqrt(N)`` yields a length-independent quantity proportional to *d*.
        """
        raw = DTWSimilarity._dtw_distance(original, synthetic, window)
        return raw / np.sqrt(max(len(original), len(synthetic)))

    @staticmethod
    def _normalize_dtw_score(original: NDArray, synthetic: NDArray, dtw_norm: float) -> float:
        """Normalise the DTW distance to a [0, 1] error score using the series scale."""
        scale = max(np.std(original), np.std(synthetic))
        if scale < 1e-8:
            scale = abs(np.mean(original) - np.mean(synthetic)) + 1e-8
        return min(1.0, dtw_norm / scale)
