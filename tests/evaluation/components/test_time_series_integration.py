# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from nemo_safe_synthesizer.config import DataParameters
from nemo_safe_synthesizer.config.evaluate import TimeSeriesEvaluationParameters
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters
from nemo_safe_synthesizer.evaluation.data_model.evaluation_dataset import EvaluationDataset
from nemo_safe_synthesizer.evaluation.components.adf_stationarity import ADFStationarity
from nemo_safe_synthesizer.evaluation.components.autocorrelation_similarity import AutocorrelationSimilarity
from nemo_safe_synthesizer.evaluation.components.drift_similarity import DriftSimilarity
from nemo_safe_synthesizer.evaluation.components.dtw_similarity import DTWSimilarity
from nemo_safe_synthesizer.evaluation.components.hurst_similarity import HurstSimilarity
from nemo_safe_synthesizer.evaluation.components.rolling_stats_similarity import RollingStatsSimilarity
from nemo_safe_synthesizer.evaluation.components.spectral_similarity import SpectralSimilarity
from nemo_safe_synthesizer.evaluation.components.time_series_similarity_score import (
    ScorecardWeights,
    TimeSeriesSimilarityScore,
)
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import EvaluationScore
from nemo_safe_synthesizer.evaluation.reports.multimodal.multimodal_report import MultimodalReport


@pytest.fixture
def ts_config():
    return SafeSynthesizerParameters(
        enable_synthesis=False,
        time_series=TimeSeriesParameters(is_timeseries=True, timestamp_column="ts"),
    )


@pytest.fixture
def non_ts_config():
    return SafeSynthesizerParameters(enable_synthesis=False)


@pytest.fixture
def ts_dataframes():
    rng = np.random.default_rng(42)
    n = 100
    t = np.arange(n, dtype=float)
    ref = pd.DataFrame({"ts": t, "value": 0.5 * t + rng.normal(0, 1, n)})
    syn = pd.DataFrame({"ts": t, "value": 0.4 * t + rng.normal(0, 1.2, n)})
    return ref, syn


class TestTimeSeriesEvaluationParametersEnabled:
    def test_returns_bool_not_list(self):
        params = TimeSeriesEvaluationParameters()
        result = params.enabled
        assert isinstance(result, bool)

    def test_disabled_by_default(self):
        params = TimeSeriesEvaluationParameters()
        assert params.enabled is False


class TestScorecardWeightsDefaults:
    def test_optimized_weight_values(self):
        w = ScorecardWeights()
        assert w.acf == 0.0
        assert w.drift == 0.1583
        assert w.hurst == 0.1821
        assert w.dtw == 0.2472
        assert w.rolling_mean == 0.1801
        assert w.rolling_var == 0.0
        assert w.spectral == 0.2871
        assert w.method == "additive"

    def test_nonzero_weights_sum_near_one(self):
        w = ScorecardWeights()
        total = w.drift + w.hurst + w.dtw + w.rolling_mean + w.spectral
        assert abs(total - 1.0) < 0.06


class TestTimeSeriesSimilarityScoreWeighted:
    def _make_component(self, cls, score_value: float):
        return cls(score=EvaluationScore.finalize_grade(raw_score=score_value, score=score_value * 10))

    def test_perfect_scores_give_perfect_composite(self):
        components = [
            self._make_component(DriftSimilarity, 1.0),
            self._make_component(HurstSimilarity, 1.0),
            self._make_component(DTWSimilarity, 1.0),
            self._make_component(RollingStatsSimilarity, 1.0),
            self._make_component(SpectralSimilarity, 1.0),
        ]
        result = TimeSeriesSimilarityScore.from_components(components)
        assert result.score.score is not None
        assert result.score.score == pytest.approx(10.0, abs=0.01)

    def test_zero_scores_give_low_composite(self):
        components = [
            self._make_component(DriftSimilarity, 0.0),
            self._make_component(HurstSimilarity, 0.0),
            self._make_component(DTWSimilarity, 0.0),
            self._make_component(RollingStatsSimilarity, 0.0),
            self._make_component(SpectralSimilarity, 0.0),
        ]
        result = TimeSeriesSimilarityScore.from_components(components)
        assert result.score.score is not None
        assert result.score.score < 2.0

    def test_adf_excluded_from_weighted_score(self):
        scored = [self._make_component(DriftSimilarity, 0.8)]
        result_without_adf = TimeSeriesSimilarityScore.from_components(scored)

        scored_with_adf = scored + [self._make_component(ADFStationarity, 0.0)]
        result_with_adf = TimeSeriesSimilarityScore.from_components(scored_with_adf)

        assert result_without_adf.score.score == result_with_adf.score.score

    def test_acf_zero_weight_excluded(self):
        """ACF weight is 0.0 so it should not affect the composite."""
        scored = [self._make_component(DriftSimilarity, 0.8)]
        result_without_acf = TimeSeriesSimilarityScore.from_components(scored)

        scored_with_acf = scored + [self._make_component(AutocorrelationSimilarity, 0.0)]
        result_with_acf = TimeSeriesSimilarityScore.from_components(scored_with_acf)

        assert result_without_acf.score.score == result_with_acf.score.score

    def test_empty_components_return_empty_score(self):
        result = TimeSeriesSimilarityScore.from_components([])
        assert result.score.score is None


class TestMultimodalReportTimeSeries:
    def test_ts_components_included_when_is_timeseries(self, ts_config, ts_dataframes):
        ref, syn = ts_dataframes
        report = MultimodalReport.from_dataframes(reference=ref, output=syn, config=ts_config)

        component_names = [c.name for c in report.components]
        assert "Drift Similarity" in component_names
        assert "Hurst Similarity" in component_names
        assert "DTW Similarity" in component_names
        assert "Rolling Stats Similarity" in component_names
        assert "Spectral Similarity" in component_names
        assert "ADF Stationarity" in component_names
        assert "Time Series Similarity Score" in component_names

    def test_ts_components_excluded_when_not_timeseries(self, non_ts_config, ts_dataframes):
        ref, syn = ts_dataframes
        report = MultimodalReport.from_dataframes(reference=ref, output=syn, config=non_ts_config)

        component_names = [c.name for c in report.components]
        assert "Drift Similarity" not in component_names
        assert "DTW Similarity" not in component_names
        assert "Time Series Similarity Score" not in component_names

    def test_ts_similarity_score_is_computed(self, ts_config, ts_dataframes):
        ref, syn = ts_dataframes
        report = MultimodalReport.from_dataframes(reference=ref, output=syn, config=ts_config)

        ts_score = None
        for c in report.components:
            if c.name == "Time Series Similarity Score":
                ts_score = c
                break

        assert ts_score is not None
        assert ts_score.score.score is not None
        assert 0 <= ts_score.score.score <= 10


def _make_grouped_ts_data(
    n_groups: int,
    n_timesteps: int,
    classes: list | None = None,
    seed: int = 42,
    noise_scale: float = 0.1,
):
    """Build reference and synthetic DataFrames with group_id, ts, label, and value columns."""
    rng = np.random.default_rng(seed)
    rows = []
    for gid in range(n_groups):
        cls = classes[gid % len(classes)] if classes else None
        for t in range(n_timesteps):
            rows.append(
                {
                    "group_id": gid,
                    "ts": t,
                    "label": cls,
                    "value": np.sin(2 * np.pi * t / n_timesteps) + rng.normal(0, noise_scale),
                }
            )
    ref = pd.DataFrame(rows)

    # Synthetic: same structure, slightly perturbed
    syn_rows = []
    for gid in range(n_groups):
        cls = classes[gid % len(classes)] if classes else None
        for t in range(n_timesteps):
            syn_rows.append(
                {
                    "group_id": gid + n_groups,  # new group_ids (no instance overlap)
                    "ts": t,
                    "label": cls,
                    "value": np.sin(2 * np.pi * t / n_timesteps) + rng.normal(0, noise_scale * 1.5),
                }
            )
    syn = pd.DataFrame(syn_rows)
    return ref, syn


def _dtw_config_with_groups():
    return SafeSynthesizerParameters(
        enable_synthesis=False,
        data=DataParameters(group_training_examples_by="group_id"),
        time_series=TimeSeriesParameters(is_timeseries=True, timestamp_column="ts"),
    )


class TestDTWClassEnvelopeDTW:
    """Tests for the class-mean + class-std envelope distributional DTW."""

    def test_dtw_class_envelope_with_labels(self):
        """2 classes, 3 groups each, shared timestamp range. Distributional match should score > 0.5."""
        ref, syn = _make_grouped_ts_data(
            n_groups=6, n_timesteps=50, classes=["A", "B"], seed=42, noise_scale=0.1,
        )
        config = _dtw_config_with_groups()
        dataset = EvaluationDataset(reference=ref, output=syn)

        result = DTWSimilarity.from_evaluation_dataset(dataset, config)

        assert result.score.score is not None
        similarity = result.score.raw_score
        assert similarity > 0.5, f"Expected similarity > 0.5, got {similarity}"
        assert "columns" in result.details
        col_detail = result.details["columns"][0]
        assert col_detail["label_stratified"] is True
        assert col_detail["n_classes"] == 2
        assert "mean_dtw_score" in col_detail
        assert "std_dtw_score" in col_detail

    def test_dtw_class_envelope_without_labels(self):
        """No label column — falls back to global aggregation. Still produces valid score."""
        ref, syn = _make_grouped_ts_data(
            n_groups=6, n_timesteps=50, classes=["A", "B"], seed=42, noise_scale=0.1,
        )
        # Drop the label column
        ref = ref.drop(columns=["label"])
        syn = syn.drop(columns=["label"])
        config = _dtw_config_with_groups()
        dataset = EvaluationDataset(reference=ref, output=syn)

        result = DTWSimilarity.from_evaluation_dataset(dataset, config)

        assert result.score.score is not None
        similarity = result.score.raw_score
        assert similarity > 0.0
        col_detail = result.details["columns"][0]
        assert col_detail["label_stratified"] is False
        assert col_detail["n_classes"] == 1

    def test_dtw_single_group_per_class(self):
        """1 group per class — std is all zeros. Should not crash; score driven by mean DTW only."""
        ref, syn = _make_grouped_ts_data(
            n_groups=2, n_timesteps=50, classes=["A", "B"], seed=42, noise_scale=0.1,
        )
        config = _dtw_config_with_groups()
        dataset = EvaluationDataset(reference=ref, output=syn)

        result = DTWSimilarity.from_evaluation_dataset(dataset, config)

        assert result.score.score is not None
        similarity = result.score.raw_score
        assert similarity > 0.0
        col_detail = result.details["columns"][0]
        # With 1 group per class, std series should be all zeros → std_dtw_score = 0.0
        assert col_detail["std_dtw_score"] == 0.0

    def test_dtw_missing_class_in_synthetic(self):
        """Ref has classes A, B, C; syn has A, B. Score should be based on A+B only."""
        ref, syn = _make_grouped_ts_data(
            n_groups=9, n_timesteps=50, classes=["A", "B", "C"], seed=42, noise_scale=0.1,
        )
        # Remove class C from synthetic
        syn = syn[syn["label"] != "C"].reset_index(drop=True)
        config = _dtw_config_with_groups()
        dataset = EvaluationDataset(reference=ref, output=syn)

        result = DTWSimilarity.from_evaluation_dataset(dataset, config)

        assert result.score.score is not None
        col_detail = result.details["columns"][0]
        assert col_detail["n_classes"] == 2  # Only A and B
        assert col_detail["label_stratified"] is True

    def test_dtw_from_arrays_unchanged(self):
        """Regression test: from_arrays() should still work as before."""
        rng = np.random.default_rng(42)
        original = np.sin(np.linspace(0, 4 * np.pi, 100)) + rng.normal(0, 0.05, 100)
        synthetic = np.sin(np.linspace(0, 4 * np.pi, 100)) + rng.normal(0, 0.05, 100)

        result = DTWSimilarity.from_arrays(original, synthetic, window=5)

        assert result.score.score is not None
        assert result.score.raw_score > 0.5
        assert "dtw_raw" in result.details
        assert "dtw_normalized" in result.details
        assert "dtw_score" in result.details
