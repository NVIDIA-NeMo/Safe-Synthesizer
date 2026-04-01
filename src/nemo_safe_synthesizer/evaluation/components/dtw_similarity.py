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

LABEL_COLUMN = "label"


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

            exclude_cols = set(filter(None, [timestamp_col, group_col]))
            if LABEL_COLUMN in reference.columns:
                exclude_cols.add(LABEL_COLUMN)
            columns = resolve_columns(reference, None, exclude=exclude_cols)
            if not columns:
                return DTWSimilarity(score=EvaluationScore(notes="No numeric columns available."))

            # When group and timestamp columns are known, compare class-level
            # mean+std envelopes (distributional DTW) rather than the full
            # flattened array, which produces meaningless cross-group discontinuities.
            use_mean_series = (
                group_col is not None
                and timestamp_col is not None
                and group_col in reference.columns
                and timestamp_col in reference.columns
                and group_col in synthetic.columns
                and timestamp_col in synthetic.columns
            )

            # Class-envelope DTW: compare class-mean and class-std series
            # between reference and synthetic with a fixed absolute window.
            max_window = 5

            per_column = []
            scores = []

            if use_mean_series:
                label_col = LABEL_COLUMN if (
                    LABEL_COLUMN in reference.columns and LABEL_COLUMN in synthetic.columns
                ) else None

                per_column, scores = DTWSimilarity._compute_class_envelope_dtw(
                    reference=reference,
                    synthetic=synthetic,
                    columns=columns,
                    timestamp_col=timestamp_col,
                    label_col=label_col,
                    max_window=max_window,
                )
            else:
                for col in columns:
                    if col not in reference.columns or col not in synthetic.columns:
                        continue
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

                    dtw_raw = DTWSimilarity._dtw_distance(ref_values, syn_values, window=max_window)
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
    # Class-envelope distributional DTW
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_class_envelope_dtw(
        reference,
        synthetic,
        columns: list[str],
        timestamp_col: str,
        label_col: str | None,
        max_window: int,
    ) -> tuple[list[dict], list[float]]:
        """Compare class-level mean and std envelopes via DTW.

        Instead of matching individual group_ids (instance-level), this computes
        the mean and standard-deviation series for each class (or globally if no
        label column), then DTWs the envelopes. This is consistent with how all
        other similarity components compare distributional properties.

        Returns ``(per_column_details, scores)`` ready for the existing
        aggregation logic.
        """
        import pandas as pd

        # Determine classes to stratify by
        if label_col is not None:
            ref_classes = set(reference[label_col].unique())
            syn_classes = set(synthetic[label_col].unique())
            common_classes = sorted(ref_classes & syn_classes, key=str)
            if not common_classes:
                # No overlapping classes — fall back to global aggregation
                common_classes = [None]
                label_col = None
        else:
            common_classes = [None]

        per_column: list[dict] = []
        scores: list[float] = []

        for col in columns:
            if col not in reference.columns or col not in synthetic.columns:
                continue

            class_combined_scores: list[float] = []
            class_mean_scores: list[float] = []
            class_std_scores: list[float] = []

            for cls in common_classes:
                # Filter by class
                if cls is not None and label_col is not None:
                    ref_subset = reference.loc[reference[label_col] == cls]
                    syn_subset = synthetic.loc[synthetic[label_col] == cls]
                else:
                    ref_subset = reference
                    syn_subset = synthetic

                # Compute mean and std series
                ref_mean = ref_subset.groupby(timestamp_col)[col].mean().sort_index()
                syn_mean = syn_subset.groupby(timestamp_col)[col].mean().sort_index()
                ref_std = ref_subset.groupby(timestamp_col)[col].std().sort_index().fillna(0.0)
                syn_std = syn_subset.groupby(timestamp_col)[col].std().sort_index().fillna(0.0)

                # Truncate to common length
                min_len = min(len(ref_mean), len(syn_mean))
                if min_len < 2:
                    continue

                ref_mean_arr = ref_mean.values[:min_len].astype(float)
                syn_mean_arr = syn_mean.values[:min_len].astype(float)
                ref_std_arr = ref_std.values[:min_len].astype(float)
                syn_std_arr = syn_std.values[:min_len].astype(float)

                # DTW on mean series (shape similarity)
                mean_dtw_raw = DTWSimilarity._dtw_distance(ref_mean_arr, syn_mean_arr, window=max_window)
                mean_dtw_norm = mean_dtw_raw / np.sqrt(min_len)
                mean_score = DTWSimilarity._normalize_dtw_score(ref_mean_arr, syn_mean_arr, mean_dtw_norm)

                # DTW on std series (spread similarity)
                if np.all(ref_std_arr == 0.0) and np.all(syn_std_arr == 0.0):
                    std_score = 0.0
                else:
                    std_dtw_raw = DTWSimilarity._dtw_distance(ref_std_arr, syn_std_arr, window=max_window)
                    std_dtw_norm = std_dtw_raw / np.sqrt(min_len)
                    std_score = DTWSimilarity._normalize_dtw_score(ref_std_arr, syn_std_arr, std_dtw_norm)

                combined = 0.7 * mean_score + 0.3 * std_score
                class_combined_scores.append(combined)
                class_mean_scores.append(mean_score)
                class_std_scores.append(std_score)

            if class_combined_scores:
                avg_col_score = float(np.mean(class_combined_scores))
                scores.append(avg_col_score)
                per_column.append(
                    {
                        "column": col,
                        "dtw_score": round(avg_col_score, 4),
                        "mean_dtw_score": round(float(np.mean(class_mean_scores)), 4),
                        "std_dtw_score": round(float(np.mean(class_std_scores)), 4),
                        "n_classes": len(class_combined_scores),
                        "label_stratified": label_col is not None,
                    }
                )

        return per_column, scores

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
