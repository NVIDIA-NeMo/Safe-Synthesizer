"""Unit tests for the AutocorrelationSimilarity component."""

import numpy as np
import pandas as pd
import pytest
from nemo_safe_synthesizer.evaluation.components.autocorrelation_similarity import (
    AutocorrelationSimilarity,
)


class TestAcfVector:
    """Tests for the _acf_vector static method."""

    def test_known_periodic_pattern(self):
        """Test ACF on a periodic signal (sine wave) produces expected pattern."""
        # Sine wave has known autocorrelation properties
        t = np.linspace(0, 4 * np.pi, 100)
        values = np.sin(t)
        max_lag = 10

        acf = AutocorrelationSimilarity._acf_vector(values, max_lag)

        assert len(acf) == max_lag
        # ACF values should be bounded [-1, 1]
        assert np.all(acf >= -1.0) and np.all(acf <= 1.0)
        # First lag of sine should have high positive correlation
        assert acf[0] > 0.5

    def test_constant_series_returns_zeros(self):
        """Test that constant series (zero variance) returns zeros."""
        values = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        max_lag = 3

        acf = AutocorrelationSimilarity._acf_vector(values, max_lag)

        assert len(acf) == max_lag
        assert np.allclose(acf, np.zeros(max_lag))

    def test_single_element_returns_zeros(self):
        """Test that single element series returns zeros."""
        values = np.array([42.0])
        max_lag = 3

        acf = AutocorrelationSimilarity._acf_vector(values, max_lag)

        assert len(acf) == max_lag
        assert np.allclose(acf, np.zeros(max_lag))

    def test_lag_exceeds_length_returns_zeros_for_excess(self):
        """Test that lags >= series length produce zeros."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # length 5
        max_lag = 7

        acf = AutocorrelationSimilarity._acf_vector(values, max_lag)

        assert len(acf) == max_lag
        # Lags 5, 6 (indices 4, 5, 6) should be zero
        assert acf[4] == 0.0
        assert acf[5] == 0.0
        assert acf[6] == 0.0

    def test_clipping_to_valid_range(self):
        """Test that ACF values are clipped to [-1, 1]."""
        # Random values - ACF should always be in valid range
        np.random.seed(42)
        values = np.random.randn(50)
        max_lag = 10

        acf = AutocorrelationSimilarity._acf_vector(values, max_lag)

        assert np.all(acf >= -1.0) and np.all(acf <= 1.0)


class TestDistance:
    """Tests for the _distance static method."""

    def test_mae_calculation(self):
        """Test MAE distance calculation."""
        ref = np.array([0.5, 0.3, 0.2])
        syn = np.array([0.4, 0.4, 0.1])
        # MAE = mean(|0.1|, |0.1|, |0.1|) = 0.1

        result = AutocorrelationSimilarity._distance(ref, syn, "mae")

        assert pytest.approx(result, abs=1e-6) == 0.1

    def test_euclidean_calculation(self):
        """Test Euclidean distance calculation."""
        ref = np.array([3.0, 0.0])
        syn = np.array([0.0, 4.0])
        # Euclidean = sqrt(9 + 16) = 5

        result = AutocorrelationSimilarity._distance(ref, syn, "euclidean")

        assert pytest.approx(result, abs=1e-6) == 5.0

    def test_identical_arrays_return_zero(self):
        """Test that identical arrays return zero distance."""
        ref = np.array([0.5, 0.3, 0.2, 0.1])
        syn = np.array([0.5, 0.3, 0.2, 0.1])

        mae_result = AutocorrelationSimilarity._distance(ref, syn, "mae")
        euc_result = AutocorrelationSimilarity._distance(ref, syn, "euclidean")

        assert mae_result == 0.0
        assert euc_result == 0.0


class TestNormalizeDistance:
    """Tests for the _normalize_distance static method."""

    def test_mae_uses_max_diff_one(self):
        """Test that MAE normalization uses max_diff=1.0."""
        diff = 0.5
        max_lag = 10  # Should not affect MAE

        result = AutocorrelationSimilarity._normalize_distance(diff, max_lag, "mae")

        assert result == 0.5  # 0.5 / 1.0

    def test_euclidean_uses_sqrt_max_lag(self):
        """Test that Euclidean normalization uses max_diff=sqrt(max_lag)."""
        max_lag = 16
        diff = 2.0  # sqrt(16) = 4, so 2/4 = 0.5

        result = AutocorrelationSimilarity._normalize_distance(diff, max_lag, "euclidean")

        assert pytest.approx(result, abs=1e-6) == 0.5

    def test_clamps_to_one(self):
        """Test that normalized distance is clamped to max 1.0."""
        diff = 2.0  # Greater than max_diff for MAE
        max_lag = 10

        result = AutocorrelationSimilarity._normalize_distance(diff, max_lag, "mae")

        assert result == 1.0


class TestComputeRmsSimilarity:
    """Tests for the _compute_rms_similarity static method."""

    def test_zero_diffs_return_one(self):
        """Test that zero differences return similarity of 1.0."""
        normalized_diffs = [0.0, 0.0, 0.0]

        result = AutocorrelationSimilarity._compute_rms_similarity(normalized_diffs)

        assert result == 1.0

    def test_max_diffs_return_zero(self):
        """Test that max differences return similarity of 0.0."""
        normalized_diffs = [1.0, 1.0, 1.0]

        result = AutocorrelationSimilarity._compute_rms_similarity(normalized_diffs)

        assert result == 0.0

    def test_rms_calculation(self):
        """Test RMS-based similarity calculation."""
        # RMS of [0.2, 0.4] = sqrt((0.04 + 0.16) / 2) = sqrt(0.1) ≈ 0.316
        # Similarity = 1 - 0.316 ≈ 0.684
        normalized_diffs = [0.2, 0.4]

        result = AutocorrelationSimilarity._compute_rms_similarity(normalized_diffs)

        expected_rms = np.sqrt(np.mean(np.square([0.2, 0.4])))
        expected_similarity = 1.0 - expected_rms
        assert pytest.approx(result, abs=1e-6) == expected_similarity


class TestAutocorrelationSingle:
    """Tests for the _autocorrelation_single static method."""

    def test_basic_series(self):
        """Test ACF computation for a basic series."""
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            }
        )
        max_lag = 3

        acf = AutocorrelationSimilarity._autocorrelation_single(df, "value", "timestamp", max_lag)

        assert len(acf) == max_lag
        # Linear trend should have high first-lag correlation
        assert acf[0] > 0.5

    def test_empty_series_returns_zeros(self):
        """Test that empty series returns zeros."""
        df = pd.DataFrame({"timestamp": [], "value": []})
        max_lag = 3

        acf = AutocorrelationSimilarity._autocorrelation_single(df, "value", "timestamp", max_lag)

        assert len(acf) == max_lag
        assert np.allclose(acf, np.zeros(max_lag))


class TestAutocorrelation:
    """Tests for the _autocorrelation static method (with optional grouping)."""

    def test_without_groups(self):
        """Test ACF without group column."""
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4, 5],
                "value": [1.0, 2.0, 1.0, 2.0, 1.0],
            }
        )
        max_lag = 2

        acf = AutocorrelationSimilarity._autocorrelation(df, "value", "timestamp", None, max_lag)

        assert len(acf) == max_lag

    def test_with_groups_averages_acf(self):
        """Test that ACF is averaged across groups."""
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
                "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0],  # A increasing, B decreasing
            }
        )
        max_lag = 2

        acf = AutocorrelationSimilarity._autocorrelation(df, "value", "timestamp", "group", max_lag)

        assert len(acf) == max_lag
        # Both groups have strong first-lag correlation (positive for A, negative for B)
        # Average should still be meaningful


class TestEvaluateGlobal:
    """Tests for the _evaluate_global static method."""

    def test_identical_data_returns_high_similarity(self):
        """Test that identical reference and synthetic data return high similarity."""
        reference = pd.DataFrame(
            {
                "timestamp": list(range(20)),
                "value": [float(i % 5) for i in range(20)],
            }
        )
        synthetic = reference.copy()
        columns = ["value"]

        details, similarity = AutocorrelationSimilarity._evaluate_global(
            reference, synthetic, columns, "timestamp", None, max_lag=5, distance_metric="mae"
        )

        assert similarity == 1.0
        assert details["evaluation_mode"] == "global"
        assert len(details["columns"]) == 1

    def test_different_data_returns_lower_similarity(self):
        """Test that different data returns lower similarity."""
        reference = pd.DataFrame(
            {
                "timestamp": list(range(20)),
                "value": [float(i) for i in range(20)],  # Linear trend
            }
        )
        synthetic = pd.DataFrame(
            {
                "timestamp": list(range(20)),
                "value": [float(i % 3) for i in range(20)],  # Periodic
            }
        )
        columns = ["value"]

        details, similarity = AutocorrelationSimilarity._evaluate_global(
            reference, synthetic, columns, "timestamp", None, max_lag=5, distance_metric="mae"
        )

        assert similarity < 1.0
        assert similarity >= 0.0

    def test_no_valid_columns_raises(self):
        """Test that no valid columns raises ValueError."""
        reference = pd.DataFrame({"timestamp": [1, 2, 3]})
        synthetic = pd.DataFrame({"timestamp": [1, 2, 3]})
        columns = ["nonexistent"]

        with pytest.raises(ValueError, match="Unable to compute autocorrelation"):
            AutocorrelationSimilarity._evaluate_global(
                reference, synthetic, columns, "timestamp", None, max_lag=2, distance_metric="mae"
            )
