from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.evaluation.components.adf_stationarity import ADFStationarity
from nemo_safe_synthesizer.evaluation.components.autocorrelation_similarity import AutocorrelationSimilarity
from nemo_safe_synthesizer.evaluation.components.component import Component
from nemo_safe_synthesizer.evaluation.components.composite_score import CompositeScore
from nemo_safe_synthesizer.evaluation.components.drift_similarity import DriftSimilarity
from nemo_safe_synthesizer.evaluation.components.dtw_similarity import DTWSimilarity
from nemo_safe_synthesizer.evaluation.components.hurst_similarity import HurstSimilarity
from nemo_safe_synthesizer.evaluation.components.rolling_stats_similarity import RollingStatsSimilarity
from nemo_safe_synthesizer.evaluation.components.spectral_similarity import SpectralSimilarity
from nemo_safe_synthesizer.evaluation.data_model.evaluation_dataset import EvaluationDataset
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import EvaluationScore
from nemo_safe_synthesizer.observability import get_logger
from numpy.typing import NDArray
from pydantic import Field

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Scorecard data structures (preserved from the notebook for compatibility)
# ---------------------------------------------------------------------------


@dataclass
class ScorecardWeights:
    """Weights for the similarity scorecard.

    method:
        "additive"       -- score = max(0, 1 - sum(w_i * e_i))
                            Weights are unconstrained (no sum-to-1 requirement).
        "multiplicative" -- score = product((1 - e_i) ^ w_i)
                            Geometric mean; a single bad metric tanks the score.
    """

    acf: float = 0.0
    drift: float = 0.1583
    hurst: float = 0.1821
    dtw: float = 0.2472
    rolling_mean: float = 0.1801
    rolling_var: float = 0.0
    spectral: float = 0.2871
    method: str = "additive"


@dataclass
class ScorecardResult:
    """Full result from the similarity scorecard."""

    # Per-metric raw values
    acf_distance: float = 0.0
    drift_original: float = 0.0
    drift_synthetic: float = 0.0
    drift_error: float = 0.0
    hurst_original: float = 0.0
    hurst_synthetic: float = 0.0
    hurst_error: float = 0.0
    dtw_raw: float = 0.0
    dtw_normalized: float = 0.0
    dtw_score: float = 0.0  # normalized to [0, 1]

    # Rolling stats (raw RMSE and [0,1]-normalized scores)
    rolling_mean_rmse: float = 0.0
    rolling_var_rmse: float = 0.0
    rolling_mean_score: float = 0.0  # normalized to [0, 1]
    rolling_var_score: float = 0.0  # normalized to [0, 1]

    # Spectral
    spectral_distance: float = 0.0  # already in [0, 1]

    # ADF stationarity
    adf_original: dict = field(default_factory=dict)
    adf_synthetic: dict = field(default_factory=dict)
    stationarity_match: bool = True

    # Combined score
    weighted_score: float = 0.0
    weights: ScorecardWeights = field(default_factory=ScorecardWeights)

    def summary_table(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "=" * 65,
            f"  SIMILARITY SCORECARD  (method: {self.weights.method})",
            "=" * 65,
            f"  ACF Distance (mae)         : {self.acf_distance:.4f}  (weight: {self.weights.acf:.4f})",
            f"  Drift Mean  orig / syn     : {self.drift_original:+.4f} / {self.drift_synthetic:+.4f}",
            f"  Drift Mean Error (norm)    : {self.drift_error:.4f}  (weight: {self.weights.drift:.4f})",
            f"  Hurst Exp   orig / syn     : {self.hurst_original:.4f} / {self.hurst_synthetic:.4f}",
            f"  Hurst Error                : {self.hurst_error:.4f}  (weight: {self.weights.hurst:.4f})",
            f"  DTW (score)                : {self.dtw_score:.4f}  (weight: {self.weights.dtw:.4f})",
            f"  Rolling Mean (score)       : {self.rolling_mean_score:.4f}  (weight: {self.weights.rolling_mean:.4f})",
            f"  Rolling Var  (score)       : {self.rolling_var_score:.4f}  (weight: {self.weights.rolling_var:.4f})",
            f"  Spectral Distance          : {self.spectral_distance:.4f}  (weight: {self.weights.spectral:.4f})",
            "-" * 65,
            f"  ADF orig stationary?       : {self.adf_original.get('is_stationary')}",
            f"  ADF syn  stationary?       : {self.adf_synthetic.get('is_stationary')}",
            f"  Stationarity match?        : {self.stationarity_match}",
            "=" * 65,
            f"  WEIGHTED SIMILARITY SCORE  : {self.weighted_score:.4f}  (1.0 = perfect)",
            "=" * 65,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Composite Component
# ---------------------------------------------------------------------------


class TimeSeriesSimilarityScore(CompositeScore):
    """Weighted aggregate of time-series similarity metrics.

    Analogous to :class:`SQSScore` for tabular quality, this composite
    component combines ACF, drift, Hurst, DTW, rolling stats, and spectral
    similarity into a single score.
    """

    name: str = Field(default="Time Series Similarity Score")

    _WEIGHT_MAP: ClassVar[dict[type, str]] = {
        AutocorrelationSimilarity: "acf",
        DriftSimilarity: "drift",
        HurstSimilarity: "hurst",
        DTWSimilarity: "dtw",
        RollingStatsSimilarity: "rolling_mean",
        SpectralSimilarity: "spectral",
    }

    @staticmethod
    def from_components(
        components: list[Component] | Component,
        weights: ScorecardWeights | None = None,
    ) -> TimeSeriesSimilarityScore:
        """Build from a list of pre-computed child components using weighted additive scoring.

        Each component is mapped to a weight via ``_WEIGHT_MAP``.  Components
        whose type is not in the map (e.g. ``ADFStationarity``,
        ``TimeSeriesLineChart``) are silently skipped.

        Additive formula:  ``score = max(0, 1 - sum(w_i * (1 - s_i)))``
        where *s_i* is the child similarity score (0-1 scale, derived from
        ``score.score / 10`` since ``EvaluationScore.score`` is on a 0-10 scale).
        """
        if isinstance(components, Component):
            components = [components]
        if components is None or len(components) == 0:
            return TimeSeriesSimilarityScore(score=EvaluationScore())

        if weights is None:
            weights = ScorecardWeights()

        weighted_error = 0.0
        total_weight = 0.0
        for c in components:
            if c.score is None or c.score.score is None:
                continue
            weight_attr = TimeSeriesSimilarityScore._WEIGHT_MAP.get(type(c))
            if weight_attr is None:
                continue
            w = getattr(weights, weight_attr, 0.0)
            if w <= 0:
                continue
            similarity = c.score.score / 10.0
            weighted_error += w * (1.0 - similarity)
            total_weight += w

        if total_weight <= 0:
            return TimeSeriesSimilarityScore(score=EvaluationScore())

        score = max(0.0, min(1.0, 1.0 - weighted_error))
        return TimeSeriesSimilarityScore(
            score=EvaluationScore.finalize_grade(raw_score=score, score=score * 10)
        )

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None = None
    ) -> TimeSeriesSimilarityScore:
        """Compute all child metrics from an EvaluationDataset and aggregate."""
        children = [
            AutocorrelationSimilarity.from_evaluation_dataset(evaluation_dataset, config),
            DriftSimilarity.from_evaluation_dataset(evaluation_dataset, config),
            HurstSimilarity.from_evaluation_dataset(evaluation_dataset, config),
            DTWSimilarity.from_evaluation_dataset(evaluation_dataset, config),
            RollingStatsSimilarity.from_evaluation_dataset(evaluation_dataset, config),
            SpectralSimilarity.from_evaluation_dataset(evaluation_dataset, config),
            # ADF is informational -- not included in the aggregate score
        ]
        return TimeSeriesSimilarityScore.from_components(children)

    # ------------------------------------------------------------------
    # Notebook-compatible scorecard
    # ------------------------------------------------------------------

    @staticmethod
    def similarity_scorecard(
        original: NDArray,
        synthetic: NDArray,
        weights: ScorecardWeights | None = None,
        acf_max_lag: int = 40,
        rolling_window: int = 20,
        dtw_window: int | None = None,
    ) -> ScorecardResult:
        """Compute the full similarity scorecard between two time series.

        This is the direct equivalent of the notebook's ``similarity_scorecard``
        function, preserved for backward compatibility.
        """
        if weights is None:
            weights = ScorecardWeights()

        original = np.asarray(original, dtype=float)
        synthetic = np.asarray(synthetic, dtype=float)

        # --- Individual metrics (delegating to Component static methods) ---
        ref_acf = AutocorrelationSimilarity._acf_vector(original, acf_max_lag)
        syn_acf = AutocorrelationSimilarity._acf_vector(synthetic, acf_max_lag)
        raw_acf_dist = AutocorrelationSimilarity._distance(ref_acf, syn_acf, "mae")
        acf_dist = AutocorrelationSimilarity._normalize_distance(raw_acf_dist, acf_max_lag, "mae")

        d_orig = DriftSimilarity._drift_mean(original)
        d_syn = DriftSimilarity._drift_mean(synthetic)
        d_err = DriftSimilarity._drift_mean_error(original, synthetic)
        d_err_clamped = min(1.0, d_err)

        h_orig = HurstSimilarity._hurst_exponent(original)
        h_syn = HurstSimilarity._hurst_exponent(synthetic)
        h_err = abs(h_orig - h_syn)

        dtw_raw = DTWSimilarity._dtw_distance(original, synthetic, window=dtw_window)
        dtw_norm = dtw_raw / np.sqrt(max(len(original), len(synthetic)))
        dtw_score_val = DTWSimilarity._normalize_dtw_score(original, synthetic, dtw_norm)

        rm_rmse = RollingStatsSimilarity._rolling_mean_rmse(original, synthetic, window=rolling_window)
        rv_rmse = RollingStatsSimilarity._rolling_var_rmse(original, synthetic, window=rolling_window)
        # Normalise rolling stats scores using the same scale as DTW
        scale_dtw = max(np.std(original), np.std(synthetic))
        if scale_dtw < 1e-8:
            scale_dtw = abs(np.mean(original) - np.mean(synthetic)) + 1e-8
        rm_score = min(1.0, rm_rmse / scale_dtw)
        rv_score = min(1.0, rv_rmse / max(np.std(original), np.std(synthetic), 1e-8))

        spec_dist = SpectralSimilarity._spectral_distance(original, synthetic)

        adf_result = ADFStationarity._stationarity_match(original, synthetic)

        # --- Weighted score ---
        errors = [acf_dist, d_err_clamped, h_err, dtw_score_val, rm_score, rv_score, spec_dist]
        ws = [
            weights.acf,
            weights.drift,
            weights.hurst,
            weights.dtw,
            weights.rolling_mean,
            weights.rolling_var,
            weights.spectral,
        ]

        if weights.method == "multiplicative":
            weighted_similarity = 1.0
            for e, w in zip(errors, ws):
                if w > 0:
                    weighted_similarity *= max(1e-12, 1.0 - e) ** w
        else:
            weighted_error = sum(w * e for w, e in zip(ws, errors))
            weighted_similarity = max(0.0, min(1.0, 1.0 - weighted_error))

        return ScorecardResult(
            acf_distance=acf_dist,
            drift_original=d_orig,
            drift_synthetic=d_syn,
            drift_error=d_err_clamped,
            hurst_original=h_orig,
            hurst_synthetic=h_syn,
            hurst_error=h_err,
            dtw_raw=dtw_raw,
            dtw_normalized=dtw_norm,
            dtw_score=dtw_score_val,
            rolling_mean_rmse=rm_rmse,
            rolling_var_rmse=rv_rmse,
            rolling_mean_score=rm_score,
            rolling_var_score=rv_score,
            spectral_distance=spec_dist,
            adf_original=adf_result["original"],
            adf_synthetic=adf_result["synthetic"],
            stationarity_match=adf_result["stationarity_match"],
            weighted_score=weighted_similarity,
            weights=weights,
        )
