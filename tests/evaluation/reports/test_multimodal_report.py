# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.evaluate import EvaluationParameters, TimeSeriesEvaluationParameters
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters
from nemo_safe_synthesizer.evaluation.components.attribute_inference_protection import AttributeInferenceProtection
from nemo_safe_synthesizer.evaluation.components.autocorrelation_similarity import AutocorrelationSimilarity
from nemo_safe_synthesizer.evaluation.components.membership_inference_protection import MembershipInferenceProtection
from nemo_safe_synthesizer.evaluation.components.text_semantic_similarity import TextSemanticSimilarity
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import EvaluationScore, Grade, PrivacyGrade
from nemo_safe_synthesizer.evaluation.render import render_report
from nemo_safe_synthesizer.evaluation.reports.multimodal import multimodal_report as multimodal_report_module
from nemo_safe_synthesizer.evaluation.reports.multimodal.multimodal_report import MultimodalReport


@pytest.fixture(autouse=True)
def stub_text_semantic_similarity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep report assembly tests independent of sentence-transformer inference."""

    class StubTextSemanticSimilarity:
        @staticmethod
        def from_evaluation_datasets(*_args, **_kwargs) -> TextSemanticSimilarity:
            return TextSemanticSimilarity(score=EvaluationScore.finalize_grade(9.0, 9.0))

    monkeypatch.setattr(
        multimodal_report_module,
        "TextSemanticSimilarity",
        StubTextSemanticSimilarity,
    )


def _minimal_multimodal_report() -> MultimodalReport:
    training_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    synthetic_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    datasets = EvaluationDatasets(training=training_df, synthetic=synthetic_df)
    return MultimodalReport(evaluation_datasets=datasets, components=[])


def _time_series_config(
    *,
    report_rows: int = 5000,
    evaluation_enabled: bool = True,
    time_series_enabled: bool | None = None,
) -> SafeSynthesizerParameters:
    return SafeSynthesizerParameters(
        time_series=TimeSeriesParameters(is_timeseries=True, timestamp_column="time"),
        evaluation=EvaluationParameters(
            enabled=evaluation_enabled,
            mia_enabled=False,
            aia_enabled=False,
            pii_replay_enabled=False,
            sqs_report_rows=report_rows,
            time_series=TimeSeriesEvaluationParameters(enabled=time_series_enabled),
        ),
    )


def test_jinja_context_job_id_none_when_nemo_job_id_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEMO_JOB_ID", raising=False)
    report = _minimal_multimodal_report()
    ctx = report.jinja_context
    assert ctx["job_id"] is None


def test_jinja_context_job_id_set_when_nemo_job_id_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_JOB_ID", "cluster-job-abc123")
    report = _minimal_multimodal_report()
    ctx = report.jinja_context
    assert ctx["job_id"] == "cluster-job-abc123"


def test_from_dataframes_applies_sqs_report_config(fixture_training_df, fixture_synthetic_df, fixture_test_df) -> None:
    """``sqs_report_rows`` / ``sqs_report_columns`` from config drive the actual subsampling.

    Regression: previously the multimodal report looked up ``sqs_rows`` /
    ``sqs_columns`` -- keys that do not exist on ``EvaluationParameters`` --
    so user-supplied row/column limits were silently ignored.
    """
    target_rows = 37
    target_cols = 3
    config = SafeSynthesizerParameters(
        evaluation=EvaluationParameters(
            mia_enabled=False,
            aia_enabled=False,
            pii_replay_enabled=False,
            sqs_report_rows=target_rows,
            sqs_report_columns=target_cols,
        ),
    )

    report = MultimodalReport.from_dataframes(
        training=fixture_training_df,
        synthetic=fixture_synthetic_df,
        test=fixture_test_df,
        config=config,
    )

    assert report.evaluation_datasets is not None
    assert report.evaluation_datasets.training_rows == target_rows
    assert report.evaluation_datasets.synthetic_rows == target_rows
    assert report.evaluation_datasets.training_cols == target_cols
    assert report.evaluation_datasets.synthetic_cols == target_cols


def test_from_dataframes_applies_mia_excluded_columns_to_mia_only(
    monkeypatch: pytest.MonkeyPatch, fixture_training_df, fixture_synthetic_df, fixture_test_df
) -> None:
    """``mia_excluded_columns`` reaches MIP and leaves AIA's columns untouched."""
    seen: dict[str, list[str]] = {}

    def fake_mia(training_df, synthetic_df, test_df, column_name=None):
        seen["mia"] = list(training_df.columns)
        return EvaluationScore(score=10.0, grade=PrivacyGrade.EXCELLENT), None, {}, {}

    def fake_aia(training_df, synthetic_df, quasi_identifier_count):
        seen["aia"] = list(training_df.columns)
        return EvaluationScore(score=10.0, grade=PrivacyGrade.EXCELLENT), None

    monkeypatch.setattr(MembershipInferenceProtection, "mia", staticmethod(fake_mia))
    monkeypatch.setattr(AttributeInferenceProtection, "_aia", staticmethod(fake_aia))

    config = SafeSynthesizerParameters(
        evaluation=EvaluationParameters(pii_replay_enabled=False, mia_excluded_columns=["small_cat"]),
    )
    report = MultimodalReport.from_dataframes(
        training=fixture_training_df,
        synthetic=fixture_synthetic_df,
        test=fixture_test_df,
        config=config,
    )

    assert "small_cat" not in seen["mia"]
    assert "small_cat" in seen["aia"]
    mip = next(c for c in report.components if isinstance(c, MembershipInferenceProtection))
    assert mip.excluded_columns == ["small_cat"]


def test_multimodal_report(
    fixture_training_df, fixture_synthetic_df, fixture_test_df, fixture_skip_privacy_metrics_config
):
    report = MultimodalReport.from_dataframes(
        training=fixture_training_df,
        synthetic=fixture_synthetic_df,
        test=fixture_test_df,
        config=fixture_skip_privacy_metrics_config,
    )

    assert len(report.components) == 11
    assert report.components[-1].name == "Synthetic Quality Score"
    assert report.components[-1].score.grade == Grade.EXCELLENT

    report_dict = report.get_dict()
    assert len(report_dict) == 6
    assert report_dict["Text Semantic Similarity"] == {
        "raw_score": 9.0,
        "grade": "Excellent",
        "score": 9.0,
        "notes": None,
    }
    assert report_dict["Synthetic Quality Score"]["grade"] == "Excellent"
    assert report_dict["Synthetic Quality Score"]["score"] > 0

    report_json = json.loads(report.get_json())
    assert report_json["Text Semantic Similarity"] == report_dict["Text Semantic Similarity"]
    assert report_json["Synthetic Quality Score"] == report_dict["Synthetic Quality Score"]


def test_time_series_metric_is_absent_when_feature_gate_is_disabled() -> None:
    frame = pd.DataFrame({"value": [0.0, 1.0, 3.0, 2.0, 4.0, 1.0]})

    report = MultimodalReport.from_dataframes(frame, frame.copy(), config=SafeSynthesizerParameters())

    assert not any(isinstance(component, AutocorrelationSimilarity) for component in report.components)
    assert report.jinja_context["with_time_series"] is False


def test_time_series_metric_can_be_disabled_in_time_series_mode() -> None:
    frame = pd.DataFrame({"time": range(6), "value": [0.0, 1.0, 3.0, 2.0, 4.0, 1.0]})

    report = MultimodalReport.from_dataframes(
        frame,
        frame.copy(),
        config=_time_series_config(time_series_enabled=False),
    )

    assert not any(isinstance(component, AutocorrelationSimilarity) for component in report.components)
    assert report.jinja_context["with_time_series"] is False


def test_global_evaluation_gate_overrides_time_series_evaluation_gate() -> None:
    frame = pd.DataFrame({"time": range(6), "value": [0.0, 1.0, 3.0, 2.0, 4.0, 1.0]})

    report = MultimodalReport.from_dataframes(
        frame,
        frame.copy(),
        config=_time_series_config(evaluation_enabled=False),
    )

    assert not any(isinstance(component, AutocorrelationSimilarity) for component in report.components)


def test_time_series_metric_uses_unsampled_ordered_data_and_renders_profiles() -> None:
    row_count = 5001
    time = np.arange(row_count)
    training = pd.DataFrame({"time": time, "value": np.sin(time / 11)})
    synthetic = training.sample(frac=1.0, random_state=17).reset_index(drop=True)

    report = MultimodalReport.from_dataframes(
        training,
        synthetic,
        config=_time_series_config(report_rows=100),
    )

    component = next(item for item in report.components if isinstance(item, AutocorrelationSimilarity))
    assert component.score.score == 10
    assert report.evaluation_datasets.training_rows == 100
    assert report.jinja_context["autocorrelation_similarity"]["evaluated_profile_count"] == 1
    output = render_report(report)
    assert output is not None
    assert "Time-Series Metrics" in output
    assert "Training median ACF" in output
    assert "Synthetic median ACF" in output


def test_enabled_unavailable_time_series_metric_renders_actionable_reason() -> None:
    frame = pd.DataFrame({"time": range(6), "value": [1.0] * 6})

    report = MultimodalReport.from_dataframes(frame, frame.copy(), config=_time_series_config())

    component = next(item for item in report.components if isinstance(item, AutocorrelationSimilarity))
    assert component.score.score is None
    assert component.score.notes == "No usable group/column autocorrelation profiles."
    output = render_report(report)
    assert output is not None
    assert "No usable group/column autocorrelation profiles." in output


def test_time_series_report_summarizes_all_profiles_in_one_chart_row() -> None:
    report = _minimal_multimodal_report()
    profiles = [
        {
            "group": str(index),
            "column": column,
            "lags": [1, 2],
            "training_acf": [0.5, 0.2],
            "synthetic_acf": [None, None] if index == 0 else [0.4, None],
            "evaluated_lags": 0 if index == 0 else 1,
            "similarity": index / 13,
        }
        for index in range(13)
        for column in ("value", "other")
    ]
    report.components = [
        AutocorrelationSimilarity(
            score=EvaluationScore.finalize_grade(raw_score=0.5, score=5.0),
            details={"profiles": profiles},
        )
    ]

    context = report.jinja_context["autocorrelation_similarity"]

    assert context["evaluated_profile_count"] == 26
    assert context["evaluated_group_count"] == 13
    assert context["evaluated_column_count"] == 2
    assert context["constant_synthetic_profile_count"] == 2
    assert [figure["title"] for figure in context["figures"]] == [
        "Typical autocorrelation",
        "Difference by lag",
        "Pair scores",
    ]
    assert [figure["column_select"] for figure in context["figures"]] == [True, False, False]
    assert [figure["tooltip_id"] for figure in context["figures"]] == [
        "autocorrelation-typical-info",
        "autocorrelation-difference-info",
        "autocorrelation-pair-scores-info",
    ]
    assert all(figure["tooltip"].strip() for figure in context["figures"])
    assert context["summary_columns"] == ["other", "value"]
    assert context["summary_traces_per_column"] == 4
    assert context["pair_score_column_count"] == 2


def test_time_series_components_drive_the_rendered_time_series_panel() -> None:
    frame = pd.DataFrame({"time": range(60), "value": np.sin(np.arange(60) / 5)})

    report = MultimodalReport.from_dataframes(frame, frame.copy(), config=_time_series_config())

    cards = report.jinja_context["time_series_cards"]
    assert cards == [report.jinja_context["autocorrelation_similarity"]]
    assert cards[0]["card_id"] == "autocorrelation-similarity"
    assert cards[0]["template"] == "jinja/components/autocorrelation_similarity.j2"
    assert cards[0]["score_colored_traces"] is True
    assert report.jinja_context["with_time_series"] is True
    output = render_report(report)
    assert output is not None
    assert 'id="autocorrelation-similarity"' in output
    assert "Measures whether numeric columns change over time like the training data" in output
    assert 'class="time-series-charts" data-score-colored-traces' in output


def test_time_series_report_runs_only_listed_components(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multimodal_report_module, "TIME_SERIES_COMPONENTS", ())
    frame = pd.DataFrame({"time": range(60), "value": np.sin(np.arange(60) / 5)})

    report = MultimodalReport.from_dataframes(frame, frame.copy(), config=_time_series_config())

    assert not any(isinstance(component, AutocorrelationSimilarity) for component in report.components)
    assert report.jinja_context["time_series_cards"] == []
    assert report.jinja_context["with_time_series"] is False


_PRIVACY_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
_RARE_CATEGORY = "rare_tier_unique"


def _privacy_frames(seed: int = 11, rows: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and synthetic frames with distinctive 6-decimal values and one rare training category."""
    rng = np.random.default_rng(seed)

    def text(count: int) -> list[str]:
        return [
            " ".join(
                " ".join(rng.choice(_PRIVACY_WORDS, size=rng.integers(3, 9))).capitalize() + "."
                for _ in range(rng.integers(1, 4))
            )
            for _ in range(count)
        ]

    def frame(rare: bool) -> pd.DataFrame:
        segment = rng.choice(["retail", "smb", "corporate"], rows).astype(object)
        if rare:
            segment[7] = _RARE_CATEGORY
        return pd.DataFrame(
            {
                "amount": np.round(rng.uniform(1000, 9000, rows), 6) + 0.123457,
                "balance": np.round(rng.normal(5000, 1500, rows), 6) + 0.654321,
                "visits": rng.integers(0, 40, rows),
                "segment": segment,
                "notes": text(rows),
            }
        )

    return frame(rare=True), frame(rare=False)


def _plotly_payloads(html: str) -> list[tuple[bool, np.ndarray]]:
    """Decode every numeric array in the report's Plotly figure data.

    Returns:
        Pairs of whether the array belongs to a marker scatter trace (the
        principal component dot plots, which plot one point per record by
        design) and the decoded array.
    """
    decoder = json.JSONDecoder()
    arrays: list[tuple[bool, np.ndarray]] = []

    def collect(value: Any, dots: bool) -> None:
        if isinstance(value, dict):
            if "bdata" in value and "dtype" in value:
                decoded = np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"]))
                arrays.append((dots, decoded))
                return
            for item in value.values():
                collect(item, dots)
        elif isinstance(value, list):
            numbers = [item for item in value if isinstance(item, int | float) and not isinstance(item, bool)]
            if numbers:
                arrays.append((dots, np.asarray(numbers, dtype=float)))
            for item in value:
                if isinstance(item, dict | list):
                    collect(item, dots)

    start = 0
    while (index := html.find("Plotly.newPlot(", start)) != -1:
        position = index + len("Plotly.newPlot(")
        while html[position].isspace():
            position += 1
        start = position
        if html[position] != '"':
            continue
        _, position = decoder.raw_decode(html, position)
        while html[position].isspace() or html[position] == ",":
            position += 1
        data, start = decoder.raw_decode(html, position)
        for trace in data:
            collect(trace, trace.get("type") == "scatter" and trace.get("mode") == "markers")
    return arrays


def _assert_no_training_values(html: str, training: pd.DataFrame, columns: list[str]) -> None:
    arrays = _plotly_payloads(html)
    assert arrays, "expected Plotly figure data in the report"
    for column in columns:
        values = np.round(training[column].to_numpy(dtype=float), 6)
        for _, array in arrays:
            assert not np.isin(np.round(array.astype(float), 6), values).any(), column
        assert not any(f"{value:.6f}" in html for value in values), column
    assert all(array.size < len(training) for dots, array in arrays if not dots)


def test_report_charts_carry_no_training_values_or_rare_categories() -> None:
    training, synthetic = _privacy_frames()
    config = SafeSynthesizerParameters(
        evaluation=EvaluationParameters(mia_enabled=False, aia_enabled=False, pii_replay_enabled=False)
    )

    report = MultimodalReport.from_dataframes(training, synthetic, config=config)
    html = render_report(report)

    assert html is not None
    _assert_no_training_values(html, training, ["amount", "balance"])
    assert _RARE_CATEGORY not in html
    for field in report.jinja_context["column_distribution_stability"]["evaluation_fields"]:
        assert "training_distribution" not in field
        assert "synthetic_distribution" not in field
        assert "unique_values_list" not in field["training_field_features"]
        assert "unique_values_list" not in field["synthetic_field_features"]
    scores = {component.name: component.score.score for component in report.components}
    assert scores["Column Distribution Stability"] == 9.0
    assert scores["Column Correlation Stability"] == 8.3
    assert scores["Deep Structure Stability"] == 8.2
    assert scores["Text Structure Similarity"] == 10.0
    assert scores["Synthetic Quality Score"] == 8.7


def test_time_series_report_charts_carry_no_training_values() -> None:
    rng = np.random.default_rng(3)
    training = pd.DataFrame(
        {
            "g": np.repeat(["a", "b"], 150),
            "time": np.tile(np.arange(150), 2),
            "value": np.round(rng.normal(100, 15, 300), 6) + 0.123457,
        }
    )
    synthetic = training.copy()
    synthetic["value"] = np.round(rng.normal(100, 15, 300), 6)
    config = SafeSynthesizerParameters(
        data=DataParameters(group_training_examples_by="g"),
        time_series=TimeSeriesParameters(is_timeseries=True, timestamp_column="time"),
        evaluation=EvaluationParameters(
            mia_enabled=False,
            aia_enabled=False,
            pii_replay_enabled=False,
            time_series=TimeSeriesEvaluationParameters(enabled=True),
        ),
    )

    html = render_report(MultimodalReport.from_dataframes(training, synthetic, config=config))

    assert html is not None
    assert 'id="autocorrelation-similarity"' in html
    _assert_no_training_values(html, training, ["value"])
