# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import itertools
import json
import subprocess
import sys
from decimal import Decimal
from enum import Enum
from typing import TypeVar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from nemo_safe_synthesizer import results as results_module
from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.differential_privacy import DifferentialPrivacyHyperparams
from nemo_safe_synthesizer.config.evaluate import EvaluationParameters, TimeSeriesEvaluationParameters
from nemo_safe_synthesizer.config.evaluation_report_data import (
    REPORT_DATA_SCHEMA_VERSION,
    EvaluationReportData,
    Metrics,
)
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters
from nemo_safe_synthesizer.evaluation.components import multi_modal_figures
from nemo_safe_synthesizer.evaluation.components.column_distribution import ColumnDistribution
from nemo_safe_synthesizer.evaluation.components.component import Component
from nemo_safe_synthesizer.evaluation.components.correlation import Correlation
from nemo_safe_synthesizer.evaluation.components.membership_inference_protection import MembershipInferenceProtection
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import EvaluationScore
from nemo_safe_synthesizer.evaluation.render import render_report
from nemo_safe_synthesizer.evaluation.report_data import build_report_data, figure_to_json, to_json_safe
from nemo_safe_synthesizer.evaluation.reports.multimodal.multimodal_report import MultimodalReport
from nemo_safe_synthesizer.pii_replacer.transform_result import ColumnStatistics
from nemo_safe_synthesizer.results import make_nss_results

PLANTED_PII = "planted-ssn-123-45-6789@example.com"

ComponentT = TypeVar("ComponentT", bound=Component)


def _reject_constant(constant: str) -> None:
    raise ValueError(f"non-standard JSON constant {constant}")


def _frame(seed: int, rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "amount": rng.normal(size=rows),
            "count": rng.integers(0, 50, size=rows),
            "color": rng.choice(["red", "green", "blue"], size=rows),
            "contact": [f"user{i}@example.com" for i in rng.integers(0, 10_000, size=rows)],
        }
    )


def _tabular_report(training: pd.DataFrame | None = None, pii_replay_enabled: bool = False) -> MultimodalReport:
    training = _frame(1) if training is None else training
    statistics = {
        "contact": ColumnStatistics(
            assigned_type="email",
            assigned_entity="email",
            detected_entity_counts={"email": len(training)},
            detected_entity_values={"email": set(training["contact"])},
            is_transformed=True,
            transform_functions={"fake"},
        )
    }
    return MultimodalReport.from_dataframes(
        training,
        _frame(2),
        _frame(3, rows=50),
        column_statistics=statistics,
        config=SafeSynthesizerParameters(
            evaluation=EvaluationParameters(mia_enabled=False, aia_enabled=False, pii_replay_enabled=pii_replay_enabled)
        ),
    )


def _time_series_report() -> MultimodalReport:
    rng = np.random.default_rng(5)

    def frame(groups: range) -> pd.DataFrame:
        return pd.concat(
            pd.DataFrame({"g": f"g{g}", "time": np.arange(60), "value": rng.normal(size=60).cumsum()}) for g in groups
        )

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
    return MultimodalReport.from_dataframes(frame(range(8)), frame(range(2, 10)), frame(range(20, 26)), config=config)


def _component(report: MultimodalReport, component_type: type[ComponentT]) -> ComponentT:
    return next(c for c in report.components if isinstance(c, component_type))


def test_report_figures_are_built_once_for_html_and_export(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    for name in ("generate_combined_correlation_figure", "histogram_figure", "bar_chart"):
        original = getattr(multi_modal_figures, name)

        def counted(*args, _original=original, _name=name, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(multi_modal_figures, name, counted)
    report = _tabular_report()

    html = render_report(report)
    data = build_report_data(report)

    assert html is not None
    distribution = _component(report, ColumnDistribution)
    assert calls.count("generate_combined_correlation_figure") == 1
    assert calls.count("histogram_figure") + calls.count("bar_chart") == len(distribution.report_figures)
    assert data.metrics.column_distribution_stability is not None
    assert [f.title for f in data.metrics.column_distribution_stability.figures] == list(distribution.report_figures)
    for title in distribution.report_figures:
        assert f'id="{title}"' in html


def test_report_figures_are_cached_figure_objects() -> None:
    report = _tabular_report()
    correlation = _component(report, Correlation)

    figures = correlation.report_figures

    assert figures is correlation.report_figures
    assert list(figures) == ["Column Correlation Stability"]
    assert all(isinstance(figure, go.Figure) for figure in figures.values())


@pytest.mark.parametrize(
    "component",
    [
        pytest.param(
            Correlation(
                score=EvaluationScore(),
                training_correlation=pd.DataFrame(),
                synthetic_correlation=pd.DataFrame(),
                correlation_difference=pd.DataFrame(),
            ),
            id="empty-correlation",
        ),
        pytest.param(MembershipInferenceProtection(score=EvaluationScore()), id="no-attack-summary"),
        pytest.param(ColumnDistribution(score=EvaluationScore()), id="distribution-without-datasets"),
    ],
)
def test_report_figures_are_empty_without_data(component: Component) -> None:
    assert component.report_figures == {}


def test_export_leaves_out_training_values_the_report_does_not_show() -> None:
    training = _frame(1)
    training.loc[0, "contact"] = PLANTED_PII
    report = _tabular_report(training, pii_replay_enabled=True)
    distribution = _component(report, ColumnDistribution)
    contact = next(f for f in distribution.evaluation_fields if f.name == "contact")
    assert PLANTED_PII in contact.training_field_features.unique_values_list
    assert contact.column_statistics is not None
    assert PLANTED_PII in contact.column_statistics.detected_entity_values["email"]

    data = build_report_data(report)
    serialized = data.model_dump_json()

    assert PLANTED_PII not in serialized
    assert "unique_values_list" not in serialized
    assert "detected_entity_values" not in serialized
    assert "training_distribution" not in serialized
    assert [c.name for c in data.columns] == list(training.columns)


def test_export_is_strict_json_without_templates_or_gauges() -> None:
    data = build_report_data(_tabular_report())

    payload = json.loads(data.model_dump_json(), parse_constant=_reject_constant)

    assert payload["schema_version"] == REPORT_DATA_SCHEMA_VERSION
    assert payload["library_version"]
    assert "figures" not in payload["scores"]["synthetic_quality"]
    figures = list(itertools.chain.from_iterable(m["figures"] for m in payload["metrics"].values() if m))
    assert figures
    assert all("template" not in figure["figure"]["layout"] for figure in figures)
    assert all("indicator" not in json.dumps(figure["figure"]["data"]) for figure in figures)


def test_export_includes_every_field_with_explicit_nulls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEMO_JOB_ID", raising=False)
    payload = json.loads(build_report_data(_tabular_report()).model_dump_json())

    assert payload["job_id"] is None
    assert payload["metrics"]["text_semantic_similarity"]["score"]["grade"] == "Unavailable"
    assert set(payload) == set(EvaluationReportData.model_fields)
    assert set(payload["metrics"]) == set(Metrics.model_fields)
    schema = EvaluationReportData.model_json_schema()
    assert set(schema["required"]) == set(EvaluationReportData.model_fields)
    assert set(schema["$defs"]["Metrics"]["required"]) == set(Metrics.model_fields)


def test_export_of_a_time_series_report() -> None:
    payload = json.loads(build_report_data(_time_series_report()).model_dump_json(), parse_constant=_reject_constant)

    assert payload["scores"]["synthetic_quality"] is not None
    assert set(payload["metrics"]) == set(Metrics.model_fields)


def test_export_reads_job_id_and_differential_privacy_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_JOB_ID", " job-1 ")
    report = _tabular_report()
    report.config = SafeSynthesizerParameters(
        privacy=DifferentialPrivacyHyperparams(dp_enabled=True, epsilon=4.0, delta=1e-5)
    )

    data = build_report_data(report)

    assert data.job_id == "job-1"
    assert data.differential_privacy.model_dump() == {"enabled": True, "epsilon": 4.0, "delta": 1e-5}


def test_reader_accepts_newer_schema_version_and_unknown_fields() -> None:
    payload = json.loads(build_report_data(_tabular_report()).model_dump_json())
    payload["schema_version"] = REPORT_DATA_SCHEMA_VERSION + 1
    payload["added_later"] = {"anything": [1, 2]}
    payload["dataset"]["added_later"] = True

    data = EvaluationReportData.model_validate(payload)

    assert data.schema_version == REPORT_DATA_SCHEMA_VERSION + 1


class _Color(Enum):
    RED = "red"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), None),
        (float("inf"), None),
        (np.float32(-np.inf), None),
        (np.int64(3), 3),
        (np.float64(0.5), 0.5),
        (np.bool_(True), True),
        (_Color.RED, "red"),
        ({"b", "a"}, ["a", "b"]),
        (np.array([1.0, np.nan]), [1.0, None]),
        ((1, "x"), [1, "x"]),
        ({np.int64(1): np.float64(2.0)}, {"1": 2.0}),
        (datetime.datetime(2026, 1, 2, 3, 4, 5), "2026-01-02T03:04:05"),
        (datetime.date(2026, 1, 2), "2026-01-02"),
        (pd.Timestamp("2026-01-02T03:04:05"), "2026-01-02T03:04:05"),
        (np.datetime64("2026-01-02T03:04:05"), "2026-01-02T03:04:05"),
        (datetime.timedelta(minutes=1, seconds=30), "P0DT0H1M30S"),
        (pd.Timedelta(days=2), "P2DT0H0M0S"),
        (np.timedelta64(90, "s"), "P0DT0H1M30S"),
        (pd.NaT, None),
        (np.datetime64("NaT"), None),
        (np.timedelta64("NaT"), None),
        (None, None),
        (Decimal("1.5"), 1.5),
        (EvaluationScore(), EvaluationScore().model_dump(mode="json")),
    ],
)
def test_to_json_safe_converts_to_strict_json(value: object, expected: object) -> None:
    converted = to_json_safe(value)

    assert converted == expected
    json.dumps(converted, allow_nan=False)


def test_to_json_safe_rejects_unknown_types() -> None:
    with pytest.raises(TypeError, match="object"):
        to_json_safe(object())


def test_figure_to_json_drops_the_theme_template() -> None:
    figure = go.Figure(go.Scatter(x=[1, 2], y=[3, float("nan")]))

    payload = figure_to_json(figure)

    assert "template" not in payload["layout"]
    json.dumps(payload, allow_nan=False)


def test_results_carry_report_data_next_to_html() -> None:
    report = _tabular_report()

    results = make_nss_results(generate_results=_frame(2), report=report)

    assert results.evaluation_report_html
    assert isinstance(results.evaluation_report_data, EvaluationReportData)


def test_results_have_no_report_data_when_evaluation_is_skipped() -> None:
    results = make_nss_results(generate_results=_frame(2), report=None)

    assert results.evaluation_report_html is None
    assert results.evaluation_report_data is None


def test_results_survive_a_failed_report_data_export(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_report: MultimodalReport) -> EvaluationReportData:
        raise TypeError("boom")

    monkeypatch.setattr(results_module, "build_report_data", fail)

    results = make_nss_results(generate_results=_frame(2), report=_tabular_report())

    assert results.evaluation_report_html
    assert results.evaluation_report_data is None


@pytest.mark.parametrize(
    "module",
    [
        "nemo_safe_synthesizer.config.evaluation_report_data",
        "nemo_safe_synthesizer.config.internal_results",
        "nemo_safe_synthesizer.evaluation.report_data",
        "nemo_safe_synthesizer.results",
    ],
)
def test_report_data_modules_import_first_in_a_fresh_interpreter(module: str) -> None:
    completed = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
