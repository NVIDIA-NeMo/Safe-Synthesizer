# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import re
from typing import Any

import numpy as np
import pandas as pd
import pytest
from jinja2 import Environment, FunctionLoader, select_autoescape

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.evaluate import EvaluationParameters, TimeSeriesEvaluationParameters
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters
from nemo_safe_synthesizer.evaluation.components.attribute_inference_protection import AttributeInferenceProtection
from nemo_safe_synthesizer.evaluation.components.autocorrelation_similarity import AutocorrelationSimilarity
from nemo_safe_synthesizer.evaluation.components.data_privacy_score import DataPrivacyScore
from nemo_safe_synthesizer.evaluation.components.membership_inference_protection import MembershipInferenceProtection
from nemo_safe_synthesizer.evaluation.components.pii_replay import PIIReplay, PIIReplayData
from nemo_safe_synthesizer.evaluation.components.text_semantic_similarity import TextSemanticSimilarity
from nemo_safe_synthesizer.evaluation.components.window_group_mia import WindowGroupMIA
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import EvaluationScore, Grade, PrivacyGrade
from nemo_safe_synthesizer.evaluation.render import _get_template, render_report
from nemo_safe_synthesizer.evaluation.report_context import report_context
from nemo_safe_synthesizer.evaluation.report_data import build_report_data
from nemo_safe_synthesizer.evaluation.reports.multimodal import multimodal_report as multimodal_report_module
from nemo_safe_synthesizer.evaluation.reports.multimodal.multimodal_report import MultimodalReport
from nemo_safe_synthesizer.pii_replacer import ColumnStatistics


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


def _context(report: MultimodalReport) -> dict[str, Any]:
    return report_context(build_report_data(report))


def _card(context: dict[str, Any], card_id: str) -> dict[str, Any]:
    cards = context["time_series_cards"] + context["data_privacy_time_series_cards"]
    return next(card for card in cards if card["card_id"] == card_id)


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


def test_report_context_job_id_none_when_nemo_job_id_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEMO_JOB_ID", raising=False)
    report = _minimal_multimodal_report()
    ctx = _context(report)
    assert ctx["job_id"] is None


def test_report_context_job_id_set_when_nemo_job_id_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_JOB_ID", "cluster-job-abc123")
    report = _minimal_multimodal_report()
    ctx = _context(report)
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
    assert _context(report)["with_time_series"] is False


def test_time_series_metric_can_be_disabled_in_time_series_mode() -> None:
    frame = pd.DataFrame({"time": range(6), "value": [0.0, 1.0, 3.0, 2.0, 4.0, 1.0]})

    report = MultimodalReport.from_dataframes(
        frame,
        frame.copy(),
        config=_time_series_config(time_series_enabled=False),
    )

    assert not any(isinstance(component, AutocorrelationSimilarity) for component in report.components)
    assert _context(report)["with_time_series"] is False


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
    assert _card(_context(report), "autocorrelation-similarity")["evaluated_profile_count"] == 1
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
            "effective_max_lag": 2,
            "training_acf": [0.5, 0.2],
            "synthetic_acf": [None, None] if index == 0 else [0.4, None],
            "evaluated_lags": 0 if index == 0 else 1,
            "error": 1 - index / 13,
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

    context = _card(_context(report), "autocorrelation-similarity")

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

    context = _context(report)
    cards = context["time_series_cards"]
    assert cards == [_card(context, "autocorrelation-similarity")]
    assert cards[0]["card_id"] == "autocorrelation-similarity"
    assert cards[0]["template"] == "jinja/components/autocorrelation_similarity.j2"
    assert cards[0]["score_colored_traces"] is True
    assert _context(report)["with_time_series"] is True
    output = render_report(report)
    assert output is not None
    assert 'id="autocorrelation-similarity"' in output
    assert "Measures whether numeric columns change over time like the training data" in output
    assert 'class="time-series-charts" data-score-colored-traces' in output


def test_time_series_cards_are_split_by_report_section() -> None:
    frame = pd.DataFrame({"time": range(60), "value": np.sin(np.arange(60) / 5)})

    report = MultimodalReport.from_dataframes(frame, frame.copy(), config=_time_series_config())

    context = _context(report)
    window_group_mia = _card(context, "window-group-mia")
    assert window_group_mia in context["data_privacy_time_series_cards"]
    assert window_group_mia not in context["time_series_cards"]


def test_time_series_report_runs_only_listed_components(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multimodal_report_module, "TIME_SERIES_COMPONENTS", ())
    frame = pd.DataFrame({"time": range(60), "value": np.sin(np.arange(60) / 5)})

    report = MultimodalReport.from_dataframes(frame, frame.copy(), config=_time_series_config())

    assert not any(isinstance(component, AutocorrelationSimilarity) for component in report.components)
    assert _context(report)["time_series_cards"] == []
    assert _context(report)["data_privacy_time_series_cards"] == []
    assert _context(report)["with_time_series"] is False


def _rendered_numbers(html: str) -> set[float]:
    """Every number in the HTML: Plotly's base64 typed arrays, decoded, and decimal literals."""
    numbers = {float(value) for value in re.findall(r"-?\d+\.\d+(?:e[-+]?\d+)?", html)}
    for dtype, data in re.findall(r'\{"dtype":"(\w+)","bdata":"([^"]+)"', html):
        decoded = base64.b64decode(data.replace("\\u002f", "/"))
        numbers.update(float(value) for value in np.frombuffer(decoded, dtype=np.dtype(dtype)))
    return numbers


def _window_group_mia_frame(groups: int, prefix: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = [
        pd.DataFrame(
            {
                "entity": f"{prefix}{index}",
                "time": np.arange(60),
                "value": rng.normal(size=60).cumsum(),
                "other": np.sin(np.arange(60) / 4 + rng.uniform(0, 6)),
            }
        )
        for index in range(groups)
    ]
    return pd.concat(parts, ignore_index=True)


def test_window_group_mia_metric_renders_in_the_data_privacy_section() -> None:
    training = _window_group_mia_frame(10, "t", seed=1)
    holdout = _window_group_mia_frame(5, "h", seed=2)
    config = _time_series_config()
    config.data.group_training_examples_by = "entity"

    report = MultimodalReport.from_dataframes(training, training.copy(), test=holdout, config=config)
    _replace_data_privacy_score(report, score=5.0)

    component = next(item for item in report.components if isinstance(item, WindowGroupMIA))
    assert component.score.score == 0.0
    context = _card(_context(report), "window-group-mia")
    assert component.details["closest_match"]["exact_copy"] is True
    closest = next(figure for figure in context["figures"] if figure["title"] == "Closest match")
    assert "Training window" in closest["html"]
    output = render_report(report)
    assert output is not None
    assert output.count('id="window-group-mia"') == 1
    assert output.index('id="data-privacy"') < output.index('id="window-group-mia"')
    assert "100.0% of training windows have an exact copy in the synthetic data." in " ".join(output.split())


def test_window_group_mia_card_is_hidden_with_the_protection_metrics_panel() -> None:
    training = _window_group_mia_frame(10, "t", seed=1)
    holdout = _window_group_mia_frame(5, "h", seed=2)
    config = _time_series_config()
    config.data.group_training_examples_by = "entity"

    report = MultimodalReport.from_dataframes(training, training.copy(), test=holdout, config=config)

    output = render_report(report)
    assert output is not None
    assert 'id="autocorrelation-similarity"' in output
    assert 'id="data-privacy"' not in output
    assert 'id="window-group-mia"' not in output


def test_window_group_mia_report_hides_a_closest_training_window_that_is_not_an_exact_copy() -> None:
    training = _window_group_mia_frame(10, "t", seed=4)
    holdout = _window_group_mia_frame(5, "h", seed=5)
    synthetic = _window_group_mia_frame(10, "t", seed=6)
    config = _time_series_config()
    config.data.group_training_examples_by = "entity"

    report = MultimodalReport.from_dataframes(training, synthetic, test=holdout, config=config)
    _replace_data_privacy_score(report, score=5.0)

    component = next(item for item in report.components if isinstance(item, WindowGroupMIA))
    match = component.details["closest_match"]
    assert match["exact_copy"] is False
    assert "training_values" not in match
    context = _card(_context(report), "window-group-mia")
    assert context["summary_traces_per_column"] == 1
    output = render_report(report)
    assert output is not None
    # Only this card: Column Distribution Stability plots training values of its own.
    card = output[output.index('id="window-group-mia"') : output.index('id="aia"')]
    assert "Closest match shows only the synthetic window." in card
    assert '"name":"Training window"' not in card
    rendered_numbers = _rendered_numbers(card)
    assert set(np.ravel(match["synthetic_values"])) <= rendered_numbers
    assert not set(training[component.details["columns"]].to_numpy().ravel()) & rendered_numbers


def _replace_data_privacy_score(report: MultimodalReport, score: float) -> None:
    report.components = [c for c in report.components if not isinstance(c, DataPrivacyScore)]
    report.components.append(
        DataPrivacyScore(score=EvaluationScore.finalize_grade(raw_score=score / 10, score=score, is_privacy=True))
    )


def _replacement_off_report(
    column_statistics: dict[str, ColumnStatistics] | None, replay: PIIReplay
) -> MultimodalReport:
    training = pd.DataFrame({"full_name": ["Ada Lovelace", "Alan Turing"], "age": [36, 41]})
    datasets = EvaluationDatasets(training=training, synthetic=training.copy(), column_statistics=column_statistics)
    return MultimodalReport(
        evaluation_datasets=datasets, components=[replay], config=SafeSynthesizerParameters(replace_pii=None)
    )


def _render_pii_card(report: MultimodalReport) -> str:
    """Render only the PII card, since the full report template needs every other component too."""
    env = Environment(loader=FunctionLoader(_get_template), autoescape=select_autoescape(["html", "xml"]))
    return env.get_template("jinja/components/pii_replacement.j2").render(ctx=_context(report))


def test_pii_card_lists_detected_columns_as_not_replaced_when_replacement_was_off() -> None:
    statistics = {
        "full_name": ColumnStatistics(
            assigned_type="structured",
            assigned_entity="full_name",
            detected_entity_counts={"full_name": 2},
        )
    }
    replay = PIIReplay(
        pii_replay_data=[
            PIIReplayData(
                column_name="full_name",
                column_assigned_type="structured",
                pii_type="full_name",
                total_training_data=2,
                unique_training_data=2,
                total_synthetic_data=1,
                unique_synthetic_data=1,
                unique_synthetic_data_percentage=50.0,
            )
        ]
    )
    report = _replacement_off_report(statistics, replay)

    pii = _context(report)["pii_replacement"]
    html = _render_pii_card(report)

    assert pii["replacement_off"] is True
    assert [group["replaced_with"] for group in pii["groups"]] == [None]
    assert pii["headline"] == "PII in 1 column, not replaced"
    assert _context(report)["with_transform"] is False
    assert 'id="pii-replay-details" hidden' in html
    assert "Not replaced" in html
    assert "PII replacement was off, so the model was trained on the original values" in html


def test_pii_card_says_why_replay_was_not_run() -> None:
    report = _replacement_off_report(None, PIIReplay(skipped_reason="PII could not be detected (no GPU)"))

    pii = _context(report)["pii_replacement"]
    html = _render_pii_card(report)

    assert pii["headline"] == "Replay not run"
    assert "PII replay was not run: PII could not be detected (no GPU)." in html
    assert "pii-replacement-table" not in html
    assert "No PII was found" not in html


def test_pii_card_keeps_the_detection_failure_with_configured_replay_columns(
    fixture_training_df, fixture_synthetic_df
) -> None:
    config = SafeSynthesizerParameters(
        replace_pii=None,
        evaluation=EvaluationParameters(
            mia_enabled=False, aia_enabled=False, pii_replay_columns=[fixture_training_df.columns[0]]
        ),
    )

    report = MultimodalReport.from_dataframes(
        training=fixture_training_df,
        synthetic=fixture_synthetic_df,
        config=config,
        pii_replay_skipped_reason="PII could not be detected (no GPU)",
    )

    pii = _context(report)["pii_replacement"]
    assert pii["groups"]
    assert pii["replay_note"] == (
        "Only the columns in evaluation.pii_replay_columns were checked: PII could not be detected (no GPU)."
    )


def test_pii_card_says_no_pii_was_found_when_detection_found_none() -> None:
    report = _replacement_off_report({}, PIIReplay())

    html = _render_pii_card(report)

    assert _context(report)["pii_replacement"]["headline"] == "No PII found"
    assert "No PII was found in the training data." in html


def test_no_pii_card_without_replacement_detection_or_a_skip_reason() -> None:
    report = _replacement_off_report(None, PIIReplay())

    assert _context(report)["pii_replacement"] is None


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
