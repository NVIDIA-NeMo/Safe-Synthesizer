# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from decimal import Decimal
from enum import Enum
from typing import TypeVar

import numpy as np
import pandas as pd
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
from nemo_safe_synthesizer.evaluation.components.attribute_inference_protection import AttributeInferenceProtection
from nemo_safe_synthesizer.evaluation.components.column_distribution import ColumnDistribution
from nemo_safe_synthesizer.evaluation.components.component import Component
from nemo_safe_synthesizer.evaluation.components.correlation import Correlation
from nemo_safe_synthesizer.evaluation.components.deep_structure import DeepStructure
from nemo_safe_synthesizer.evaluation.components.membership_inference_protection import MembershipInferenceProtection
from nemo_safe_synthesizer.evaluation.components.text_semantic_similarity import TextSemanticSimilarity
from nemo_safe_synthesizer.evaluation.components.text_structure_similarity import TextStructureSimilarity
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import EvaluationScore
from nemo_safe_synthesizer.evaluation.report_data import build_report_data, to_json_safe
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


def _report_with(*components: Component) -> MultimodalReport:
    report = _tabular_report()
    report.components = list(components)
    return report


def _component(report: MultimodalReport, component_type: type[ComponentT]) -> ComponentT:
    return next(c for c in report.components if isinstance(c, component_type))


def test_column_distributions_match_the_charted_values() -> None:
    report = _tabular_report()
    datasets = report.evaluation_datasets
    assert datasets is not None
    fields = {f.name: f for f in _component(report, ColumnDistribution).evaluation_fields}

    data = build_report_data(report)

    charted = {c.name: c.distribution for c in data.columns if c.distribution is not None}
    assert set(charted) == set(datasets.get_tabular_columns())
    for name, distribution in charted.items():
        field, training, synthetic = fields[name], datasets.training[name], datasets.synthetic[name]
        if distribution.bin_edges is not None:
            figure = multi_modal_figures.histogram_figure(training, synthetic)
        else:
            assert field.training_distribution is not None and field.synthetic_distribution is not None
            figure = multi_modal_figures.bar_chart(
                field.training_distribution, field.synthetic_distribution, len(training)
            )
        training_bar, synthetic_bar = figure.data[:2]
        assert list(training_bar.y) == pytest.approx(distribution.training_percent)
        assert list(synthetic_bar.y) == pytest.approx(distribution.synthetic_percent)
    assert charted["amount"].bin_edges is not None
    assert len(charted["amount"].bin_edges) == len(charted["amount"].labels) + 1
    assert charted["color"].bin_edges is None
    assert charted["color"].labels == ["blue", "green", "red"]


def test_categorical_distribution_folds_rare_training_categories() -> None:
    distribution = multi_modal_figures.categorical_distribution(
        {"a": 60.0, "b": 37.0, "rare-1": 2.0, "rare-2": 1.0}, {"a": 50.0, "b": 50.0}, training_count=100
    )

    assert distribution.labels == ["a", "b", "Other (2 categories)"]
    assert distribution.training_percent == [60.0, 37.0, 3.0]
    assert distribution.synthetic_percent == [50.0, 50.0, 0.0]


def test_numeric_distribution_is_none_without_finite_training_values() -> None:
    assert multi_modal_figures.numeric_distribution(pd.Series([np.nan, np.inf]), pd.Series([1.0])) is None


def test_projection_keeps_the_points_the_chart_draws() -> None:
    rng = np.random.default_rng(0)
    training = pd.DataFrame(rng.normal(size=(6000, 2)), columns=["pc1", "pc2"])
    synthetic = pd.DataFrame(rng.normal(size=(300, 2)), columns=["pc1", "pc2"])
    report = _report_with(DeepStructure(score=EvaluationScore(), training_pca=training, synthetic_pca=synthetic))

    deep_structure = build_report_data(report).metrics.deep_structure_stability

    assert deep_structure is not None
    projection = deep_structure.projection

    figure = multi_modal_figures.structure_stability_figure(training, synthetic)
    assert projection is not None
    assert len(projection.training.x) == 5000
    assert projection.training.x == list(figure.data[0].x)
    assert projection.training.y == list(figure.data[0].y)
    assert projection.synthetic.x == synthetic["pc1"].tolist()


def test_export_carries_privacy_attack_results() -> None:
    attack_summary = pd.DataFrame(
        {
            "Protection": ["Unavailable", "Poor", "Moderate", "Good", "Very Good", "Excellent"],
            "Attack Count": [0, 1, 0, 2, 3, 4],
            "Attack Percentage": [0.0, 10.0, 0.0, 20.0, 30.0, 40.0],
        }
    )
    column_risks = pd.DataFrame({"Column": ["a", "b"], "Risk": [9.5, 2.0], "Protection": ["Excellent", "Poor"]})
    report = _report_with(
        MembershipInferenceProtection(score=EvaluationScore(), attack_sum_df=attack_summary),
        AttributeInferenceProtection(score=EvaluationScore(), col_accuracy_df=column_risks),
    )

    metrics = build_report_data(report).metrics

    assert metrics.membership_inference_protection is not None
    assert [(o.grade, o.count) for o in metrics.membership_inference_protection.outcomes] == [
        ("Excellent", 4),
        ("Very Good", 3),
        ("Good", 2),
        ("Moderate", 0),
        ("Poor", 1),
    ]
    assert metrics.attribute_inference_protection is not None
    assert [(c.column, c.risk, c.grade) for c in metrics.attribute_inference_protection.columns] == [
        ("b", 2.0, "Poor"),
        ("a", 9.5, "Excellent"),
    ]


def test_export_has_empty_chart_values_without_data() -> None:
    report = _report_with(
        Correlation(
            score=EvaluationScore(),
            training_correlation=pd.DataFrame(),
            synthetic_correlation=pd.DataFrame(),
            correlation_difference=pd.DataFrame(),
        ),
        DeepStructure(score=EvaluationScore()),
        MembershipInferenceProtection(score=EvaluationScore()),
        AttributeInferenceProtection(score=EvaluationScore()),
        TextSemanticSimilarity(score=EvaluationScore()),
        TextStructureSimilarity(score=EvaluationScore()),
    )

    metrics = build_report_data(report).metrics.model_dump()

    assert metrics["column_correlation_stability"]["correlations"] is None
    assert metrics["deep_structure_stability"]["projection"] is None
    assert metrics["membership_inference_protection"]["outcomes"] == []
    assert metrics["attribute_inference_protection"]["columns"] == []
    assert metrics["text_semantic_similarity"]["projections"] == {}
    assert metrics["text_structure_similarity"]["columns"] == {}


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


def test_export_is_strict_json_with_correlations_and_no_plotly_figures() -> None:
    data = build_report_data(_tabular_report())
    serialized = data.model_dump_json()

    payload = json.loads(serialized, parse_constant=_reject_constant)

    assert payload["schema_version"] == REPORT_DATA_SCHEMA_VERSION
    assert payload["library_version"]
    assert '"layout"' not in serialized
    correlations = payload["metrics"]["column_correlation_stability"]["correlations"]
    size = len(correlations["columns"])
    assert size > 1
    for matrix in ("training", "synthetic", "difference"):
        assert len(correlations[matrix]) == size
        assert all(len(row) == size for row in correlations[matrix])


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


def test_results_carry_report_data_next_to_html() -> None:
    report = _tabular_report()

    results = make_nss_results(generate_results=_frame(2), report=report)

    assert results.evaluation_report_html
    assert isinstance(results.evaluation_report_data, EvaluationReportData)


def test_results_have_no_report_data_when_evaluation_is_skipped() -> None:
    results = make_nss_results(generate_results=_frame(2), report=None)

    assert results.evaluation_report_html is None
    assert results.evaluation_report_data is None


def test_results_keep_the_synthetic_data_when_report_data_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_report: MultimodalReport) -> EvaluationReportData:
        raise TypeError("boom")

    monkeypatch.setattr(results_module, "build_report_data", fail)

    results = make_nss_results(generate_results=_frame(2), report=_tabular_report())

    assert results.synthetic_data is not None
    assert results.evaluation_report_html is None
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
