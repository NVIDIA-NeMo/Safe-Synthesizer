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


class SpectralSimilarity(Component):
    """Compares the frequency content of two series via their power spectra.

    Computes the normalised L2 distance between magnitude spectra after
    detrending.  Catches periodicity mismatches, missing seasonality, and
    structural form differences.

    Returns a distance in [0, 1] where 0 = identical spectra.
    """

    name: str = Field(default="Spectral Similarity")
    details: dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def jinja_context(self):
        ctx = super().jinja_context
        ctx["anchor_link"] = "#spectral-similarity"
        ctx["details"] = self.details
        return ctx

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None = None
    ) -> SpectralSimilarity:
        ts_eval_config = get_time_series_config(config)
        is_timeseries = config.time_series.is_timeseries if config and getattr(config, "time_series", None) else False

        if not ts_eval_config and not is_timeseries:
            return SpectralSimilarity(score=EvaluationScore())

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
                return SpectralSimilarity(score=EvaluationScore(notes="No numeric columns available."))

            per_column = []
            distances = []
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

                dist = SpectralSimilarity._spectral_distance(ref_values, syn_values)
                distances.append(dist)
                per_column.append(
                    {
                        "column": col,
                        "spectral_distance": round(dist, 4),
                    }
                )

            if not distances:
                return SpectralSimilarity(score=EvaluationScore(notes="No valid columns for spectral comparison."))

            avg_dist = float(np.mean(distances))
            similarity = max(0.0, 1.0 - avg_dist)
            evaluation_score = EvaluationScore.finalize_grade(raw_score=similarity, score=similarity * 10)
            details = {"columns": per_column, "average_distance": round(avg_dist, 4)}
            return SpectralSimilarity(score=evaluation_score, details=details)

        except Exception as exc:
            logger.exception("Failed to compute spectral similarity metric.")
            return SpectralSimilarity(score=EvaluationScore(notes=str(exc)))

    @staticmethod
    def from_arrays(original: NDArray, synthetic: NDArray) -> SpectralSimilarity:
        """Convenience entry point for raw NumPy arrays (used by notebooks and tests)."""
        original = np.asarray(original, dtype=float)
        synthetic = np.asarray(synthetic, dtype=float)

        dist = SpectralSimilarity._spectral_distance(original, synthetic)
        similarity = max(0.0, 1.0 - dist)

        evaluation_score = EvaluationScore.finalize_grade(raw_score=similarity, score=similarity * 10)
        details = {
            "spectral_distance": round(dist, 4),
        }
        return SpectralSimilarity(score=evaluation_score, details=details)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    @staticmethod
    def _spectral_distance(original: NDArray, synthetic: NDArray) -> float:
        """Distance between the power spectra of two series.

        Computes the normalized L2 distance between magnitude spectra.
        Returns a value in [0, 1] where 0 = identical spectra.
        """
        # Detrend (remove linear trend) so trend energy doesn't drown out structural features
        n = min(len(original), len(synthetic))
        t = np.arange(n, dtype=float)
        orig_detrend = original[:n] - np.polyval(np.polyfit(t, original[:n], 1), t)
        syn_detrend = synthetic[:n] - np.polyval(np.polyfit(t, synthetic[:n], 1), t)

        # Compute magnitude spectra (positive frequencies only)
        orig_fft = np.abs(np.fft.rfft(orig_detrend))
        syn_fft = np.abs(np.fft.rfft(syn_detrend))

        # Normalize each spectrum to unit energy so we compare shape, not scale
        orig_norm = np.linalg.norm(orig_fft)
        syn_norm = np.linalg.norm(syn_fft)
        if orig_norm < 1e-12 or syn_norm < 1e-12:
            # One or both series are constant -- spectra are trivial
            if orig_norm < 1e-12 and syn_norm < 1e-12:
                return 0.0  # both constant = same spectrum
            return 1.0  # one has content, the other doesn't
        orig_fft = orig_fft / orig_norm
        syn_fft = syn_fft / syn_norm

        # L2 distance between normalized spectra, scaled to [0, 1]
        # Max possible L2 between two unit vectors is sqrt(2) (orthogonal)
        dist = float(np.linalg.norm(orig_fft - syn_fft))
        return min(1.0, dist / np.sqrt(2))
