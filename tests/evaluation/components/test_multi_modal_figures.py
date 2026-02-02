"""Tests for multi_modal_figures.py - Plotly figure generation functions."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from nemo_safe_synthesizer.evaluation.components.multi_modal_figures import (
    bar_chart,
    combine_subplots,
    correlation_heatmap,
    degree_to_radian,
    gauge_chart,
    generate_aia_figure,
    generate_combined_correlation_figure,
    generate_mia_figure,
    generate_text_semantic_similarity_figures,
    generate_text_structure_similarity_figures,
    get_auto_bins,
    histogram,
    histogram_figure,
    pie,
    scatter,
    scatter_plot,
    structure_stability_figure,
)
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import (
    EvaluationScore,
    Grade,
    PrivacyGrade,
)


class TestDegreeToRadian:
    """Tests for degree_to_radian function."""

    def test_zero_degrees(self):
        assert degree_to_radian(0) == 0

    def test_180_degrees(self):
        assert degree_to_radian(180) == pytest.approx(np.pi)

    def test_90_degrees(self):
        assert degree_to_radian(90) == pytest.approx(np.pi / 2)

    def test_360_degrees(self):
        assert degree_to_radian(360) == pytest.approx(2 * np.pi)

    def test_negative_degrees(self):
        assert degree_to_radian(-90) == pytest.approx(-np.pi / 2)


class TestGaugeChart:
    """Tests for gauge_chart function."""

    def test_gauge_chart_with_score(self):
        evaluation_score = EvaluationScore(score=7.5, grade=Grade.GOOD)
        fig = gauge_chart(evaluation_score)

        assert isinstance(fig, go.Figure)
        # Should have at least 2 traces (filled and non-filled arcs) plus text
        assert len(fig.data) >= 2

    def test_gauge_chart_without_score(self):
        evaluation_score = EvaluationScore(score=None, grade=Grade.UNAVAILABLE)
        fig = gauge_chart(evaluation_score)

        assert isinstance(fig, go.Figure)
        # Text trace should show "--"
        text_traces = [t for t in fig.data if hasattr(t, "text") and t.text is not None]
        assert any("--" in str(t.text) for t in text_traces)

    def test_gauge_chart_min_size(self):
        evaluation_score = EvaluationScore(score=5.0, grade=Grade.MODERATE)
        fig = gauge_chart(evaluation_score, min=True)

        assert isinstance(fig, go.Figure)
        # Min gauge should have smaller dimensions
        assert fig.layout.width == 90
        assert fig.layout.height == 75

    def test_gauge_chart_regular_size(self):
        evaluation_score = EvaluationScore(score=5.0, grade=Grade.MODERATE)
        fig = gauge_chart(evaluation_score, min=False)

        assert isinstance(fig, go.Figure)
        assert fig.layout.width == 215
        assert fig.layout.height == 180

    def test_gauge_chart_privacy_grade_uses_dps_color(self):
        evaluation_score = EvaluationScore(score=8.0, grade=PrivacyGrade.VERY_GOOD)
        fig = gauge_chart(evaluation_score)

        assert isinstance(fig, go.Figure)

    def test_gauge_chart_extreme_scores(self):
        # Test score of 0
        evaluation_score = EvaluationScore(score=0.0, grade=Grade.VERY_POOR)
        fig = gauge_chart(evaluation_score)
        assert isinstance(fig, go.Figure)

        # Test score of 10
        evaluation_score = EvaluationScore(score=10.0, grade=Grade.EXCELLENT)
        fig = gauge_chart(evaluation_score)
        assert isinstance(fig, go.Figure)


class TestPie:
    """Tests for pie function."""

    def test_pie_basic(self):
        labels = ["Good", "Moderate", "Poor"]
        values = pd.Series([50, 30, 20])
        fig = pie(labels, values)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Pie)

    def test_pie_with_privacy_grades(self):
        labels = [grade.value for grade in PrivacyGrade if grade.value != "Unavailable"]
        values = pd.Series([20, 20, 20, 20, 20])
        fig = pie(labels, values, sort=False)

        assert isinstance(fig, go.Figure)
        pie_trace = fig.data[0]
        assert list(pie_trace.labels) == labels

    def test_pie_custom_textinfo(self):
        # Labels must be valid keys in INFERENCE_ATTACK_VALUES_FOR_GRAPHS
        labels = ["Good", "Moderate"]
        values = pd.Series([60, 40])
        fig = pie(labels, values, textinfo="percent")

        assert isinstance(fig, go.Figure)


class TestScatter:
    """Tests for scatter function (trace generator)."""

    def test_scatter_basic(self):
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([2, 4, 6, 8, 10])
        trace = scatter(x, y)

        assert isinstance(trace, go.Scatter)
        assert trace.mode == "markers"
        assert trace.name == "Reference"

    def test_scatter_custom_params(self):
        x = pd.Series([1, 2, 3])
        y = pd.Series([3, 2, 1])
        trace = scatter(x, y, mode="lines", color="#FF0000", name="Custom")

        assert isinstance(trace, go.Scatter)
        assert trace.mode == "lines"
        assert trace.name == "Custom"


class TestHistogram:
    """Tests for histogram function (trace generator)."""

    def test_histogram_basic(self):
        x = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 5])
        trace = histogram(x)

        assert isinstance(trace, go.Histogram)
        assert trace.name == "Reference"
        assert trace.showlegend is False

    def test_histogram_custom_params(self):
        x = pd.Series([1, 2, 3, 4, 5])
        trace = histogram(x, color="#00FF00", name="Output", histnorm="percent")

        assert isinstance(trace, go.Histogram)
        assert trace.name == "Output"


class TestGetAutoBins:
    """Tests for get_auto_bins function."""

    def test_get_auto_bins_basic(self):
        x1 = pd.Series([1, 2, 3, 4, 5])
        x2 = pd.Series([2, 3, 4, 5, 6])
        bins = get_auto_bins(x1, x2)

        assert "start" in bins
        assert "end" in bins
        assert "size" in bins
        assert bins["start"] <= 1  # Should include minimum of both
        assert bins["end"] >= 6  # Should include maximum of both

    def test_get_auto_bins_same_data(self):
        x = pd.Series([1, 2, 3, 4, 5])
        bins = get_auto_bins(x, x)

        assert bins["start"] <= 1
        assert bins["end"] >= 5


class TestGenerateMiaFigure:
    """Tests for generate_mia_figure function."""

    def test_generate_mia_figure_basic(self, mia_aia_df):
        df = mia_aia_df
        fig = generate_mia_figure(df)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Pie)

    def test_generate_mia_figure_with_zero_percentages(self, mia_aia_df):
        df = mia_aia_df
        df.loc[df["Attack Percentage"] == 0, "Attack Percentage"] = np.nan
        fig = generate_mia_figure(df)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Pie)

    def test_generate_mia_figure_with_nan_percentages(self, mia_aia_df_with_nan_protection):
        df = mia_aia_df_with_nan_protection
        fig = generate_mia_figure(df)

        assert isinstance(fig, go.Figure)
        # Zero values should be replaced with NaN
        pie_trace = fig.data[0]
        assert pie_trace is not None

    def test_generate_mia_figure_ordering(self, mia_aia_df):
        # Test that the figure sorts by Protection grade
        df = mia_aia_df
        fig = generate_mia_figure(df)

        assert isinstance(fig, go.Figure)


class TestGenerateAiaFigure:
    """Tests for generate_aia_figure function."""

    def test_generate_aia_figure_basic(self, mia_aia_df):
        df = mia_aia_df
        fig = generate_aia_figure(df)

        assert isinstance(fig, go.Figure)
        df = mia_aia_df
        fig = generate_aia_figure(df)

        assert isinstance(fig, go.Figure)


class TestCorrelationHeatmap:
    """Tests for correlation_heatmap function."""

    def test_correlation_heatmap_basic(self, mia_aia_df):
        matrix = mia_aia_df
        fig = correlation_heatmap(matrix)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_correlation_heatmap_truncates_long_names(self, mia_aia_df):
        matrix = mia_aia_df
        fig = correlation_heatmap(matrix)

        assert isinstance(fig, go.Figure)
        assert isinstance(fig, go.Figure)


class TestGenerateCombinedCorrelationFigure:
    """Tests for generate_combined_correlation_figure function."""

    def test_generate_combined_correlation_figure_basic(self):
        ref_corr = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]], columns=["A", "B"], index=["A", "B"])
        out_corr = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]], columns=["A", "B"], index=["A", "B"])
        diff_corr = (ref_corr - out_corr).abs()

        fig = generate_combined_correlation_figure(ref_corr, out_corr, diff_corr)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3
        assert isinstance(fig.data[0], go.Heatmap)
        assert isinstance(fig.data[1], go.Heatmap)
        assert isinstance(fig.data[2], go.Heatmap)


class TestScatterPlot:
    """Tests for scatter_plot function."""

    def test_scatter_plot_basic(self, mia_aia_df):
        x = mia_aia_df["Risk"]
        y = mia_aia_df["Attack Percentage"]
        fig = scatter_plot(x, y)

        assert isinstance(fig, go.Figure)

    def test_scatter_plot_respects_maximum_points(self, mia_aia_df):
        x = mia_aia_df["Risk"]
        y = mia_aia_df["Attack Percentage"]
        fig = scatter_plot(x, y, maximum_points=100)

        assert isinstance(fig, go.Figure)
        # The scatter should have at most 100 points
        scatter_trace = fig.data[0]
        assert len(scatter_trace.x) <= 100

    def test_scatter_plot_no_cap_when_zero(self, mia_aia_df):
        x = mia_aia_df["Risk"]
        y = mia_aia_df["Attack Percentage"]
        fig = scatter_plot(x, y, maximum_points=0)

        assert isinstance(fig, go.Figure)
        # With maximum_points=0, it should use min(len(x), len(y))
        scatter_trace = fig.data[0]
        assert len(scatter_trace.x) == min(len(x), len(y))


class TestStructureStabilityFigure:
    """Tests for structure_stability_figure function."""

    def test_structure_stability_figure_basic(self):
        reference = pd.DataFrame({"pc1": np.random.randn(100), "pc2": np.random.randn(100)})
        output = pd.DataFrame({"pc1": np.random.randn(100), "pc2": np.random.randn(100)})

        fig = structure_stability_figure(reference, output)

        assert isinstance(fig, go.Figure)
        assert fig.layout.height == 420
        assert fig.layout.width == 900


class TestCombineSubplots:
    """Tests for combine_subplots function."""

    def test_combine_subplots_basic(self):
        fig1 = go.Figure(data=[go.Scatter(x=[1, 2], y=[1, 2])])
        fig2 = go.Figure(data=[go.Scatter(x=[3, 4], y=[3, 4])])

        combined = combine_subplots([fig1, fig2], titles=["Plot 1", "Plot 2"])

        assert isinstance(combined, go.Figure)
        # Should have traces from both figures
        assert len(combined.data) == 2

    def test_combine_subplots_single_figure(self):
        fig1 = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 2, 3])])

        combined = combine_subplots([fig1])

        assert isinstance(combined, go.Figure)
        assert len(combined.data) == 1


class TestBarChart:
    """Tests for bar_chart function."""

    def test_bar_chart_basic(self):
        ref_dist = {"A": 0.3, "B": 0.5, "C": 0.2}
        out_dist = {"A": 0.25, "B": 0.55, "C": 0.2}

        fig = bar_chart(ref_dist, out_dist)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Reference and Output bars
        assert all(isinstance(trace, go.Bar) for trace in fig.data)

    def test_bar_chart_missing_keys(self):
        # Output has key that reference doesn't have
        ref_dist = {"A": 0.5, "B": 0.5}
        out_dist = {"A": 0.4, "C": 0.6}

        fig = bar_chart(ref_dist, out_dist)

        assert isinstance(fig, go.Figure)
        # Should include all keys: A, B, C
        bar_trace = fig.data[0]
        assert len(bar_trace.x) == 3

    def test_bar_chart_empty_distributions(self):
        ref_dist = {}
        out_dist = {}

        fig = bar_chart(ref_dist, out_dist)

        assert isinstance(fig, go.Figure)

    def test_bar_chart_many_columns(self):
        # More than 60 columns should use narrower bars
        ref_dist = {f"col_{i}": 1 / 70 for i in range(70)}
        out_dist = {f"col_{i}": 1 / 70 for i in range(70)}

        fig = bar_chart(ref_dist, out_dist)

        assert isinstance(fig, go.Figure)
        # Bars should have explicit width set
        assert len(fig.data[0].width) == 70


class TestHistogramFigure:
    """Tests for histogram_figure function."""

    def test_histogram_figure_basic(self):
        reference = pd.Series(np.random.randn(1000))
        output = pd.Series(np.random.randn(1000))

        fig = histogram_figure(reference, output)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Reference and Output histograms

    def test_histogram_figure_with_nans(self):
        reference = pd.Series([1.0, 2.0, np.nan, 3.0, np.nan])
        output = pd.Series([1.5, np.nan, 2.5, 3.5, 4.0])

        fig = histogram_figure(reference, output)

        assert isinstance(fig, go.Figure)

    def test_histogram_figure_empty_after_dropna(self):
        reference = pd.Series([np.nan, np.nan])
        output = pd.Series([1.0, 2.0])

        fig = histogram_figure(reference, output)

        # Should return empty figure when one series is all NaN
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_histogram_figure_same_values(self):
        # Edge case: all values are the same (no variance)
        reference = pd.Series([5.0, 5.0, 5.0, 5.0])
        output = pd.Series([5.0, 5.0, 5.0, 5.0])

        fig = histogram_figure(reference, output)

        assert isinstance(fig, go.Figure)


class TestGenerateTextStructureSimilarityFigures:
    """Tests for generate_text_structure_similarity_figures function."""

    def test_returns_none_for_empty_statistics(self):
        training_stats = type(
            "Stats",
            (),
            {"per_record_statistics": pd.DataFrame()},
        )()
        synthetic_stats = type(
            "Stats",
            (),
            {"per_record_statistics": pd.DataFrame()},
        )()

        result = generate_text_structure_similarity_figures(training_stats, synthetic_stats, "Test Title")

        assert result is None

    def test_basic_structure_similarity(self):
        data = {
            "sentence_count": np.random.randint(1, 10, 100),
            "average_words_per_sentence": np.random.uniform(5, 20, 100),
            "average_characters_per_word": np.random.uniform(3, 8, 100),
        }
        training_stats = type(
            "Stats",
            (),
            {"per_record_statistics": pd.DataFrame(data)},
        )()
        synthetic_stats = type(
            "Stats",
            (),
            {"per_record_statistics": pd.DataFrame(data)},
        )()

        result = generate_text_structure_similarity_figures(training_stats, synthetic_stats, "Text Structure")

        assert isinstance(result, go.Figure)


class TestGenerateTextSemanticSimilarityFigures:
    """Tests for generate_text_semantic_similarity_figures function."""

    def test_returns_none_for_empty_pca(self):
        training_pca = pd.DataFrame()
        synthetic_pca = pd.DataFrame()

        result = generate_text_semantic_similarity_figures(training_pca, synthetic_pca, "Test Title")

        assert result is None

    def test_basic_semantic_similarity(self):
        training_pca = pd.DataFrame({"pc1": np.random.randn(100), "pc2": np.random.randn(100)})
        synthetic_pca = pd.DataFrame({"pc1": np.random.randn(100), "pc2": np.random.randn(100)})

        result = generate_text_semantic_similarity_figures(training_pca, synthetic_pca, "Semantic Similarity")

        assert isinstance(result, go.Figure)
