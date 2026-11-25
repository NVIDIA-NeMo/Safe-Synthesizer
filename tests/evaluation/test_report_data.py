# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from enum import Enum
from typing import TypeVar

import numpy as np
import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.evaluate import EvaluationParameters, TimeSeriesEvaluationParameters
from nemo_safe_synthesizer.config.evaluation_report_data import (
    REPORT_DATA_SCHEMA_VERSION,
    EvaluationReportData,
)
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters
from nemo_safe_synthesizer.evaluation.components.attribute_inference_protection import AttributeInferenceProtection
from nemo_safe_synthesizer.evaluation.components.autocorrelation_similarity import AutocorrelationSimilarity
from nemo_safe_synthesizer.evaluation.components.column_distribution import ColumnDistribution
from nemo_safe_synthesizer.evaluation.components.component import Component
from nemo_safe_synthesizer.evaluation.components.membership_inference_protection import MembershipInferenceProtection
from nemo_safe_synthesizer.evaluation.components.window_group_mia import WindowGroupMIA
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import EvaluationScore
from nemo_safe_synthesizer.evaluation.report_context import report_context
from nemo_safe_synthesizer.evaluation.report_data import build_report_data, to_json_safe
from nemo_safe_synthesizer.evaluation.reports.multimodal.multimodal_report import MultimodalReport
from nemo_safe_synthesizer.pii_replacer.transform_result import ColumnStatistics

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
    assert [(c.column, c.protection, c.grade) for c in metrics.attribute_inference_protection.columns] == [
        ("b", 2.0, "Poor"),
        ("a", 9.5, "Excellent"),
    ]


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


def test_transform_columns_show_when_the_only_transformed_column_is_not_evaluated() -> None:
    report = _tabular_report()
    distribution = _component(report, ColumnDistribution)
    distribution.evaluation_fields = [f for f in distribution.evaluation_fields if f.name != "contact"]

    data = build_report_data(report)

    assert "contact" not in [c.name for c in data.columns]
    assert data.any_column_transformed is True
    assert report_context(data)["with_transform"] is True


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


def test_export_of_a_time_series_report_has_typed_details_without_training_only_values() -> None:
    payload = json.loads(build_report_data(_time_series_report()).model_dump_json(), parse_constant=_reject_constant)
    metrics = EvaluationReportData.model_validate(payload).metrics
    assert metrics.autocorrelation_similarity is not None
    assert metrics.window_membership_inference_protection is not None
    autocorrelation = metrics.autocorrelation_similarity.details
    window = metrics.window_membership_inference_protection.details

    assert autocorrelation.profiles
    assert autocorrelation.groups_only_in_training_count == 2
    assert '"g0"' not in json.dumps(payload["metrics"]["autocorrelation_similarity"])
    assert window.attack is not None
    assert window.training_group_count == 8
    assert 0 < len(window.per_group) <= 8
    assert window.closest_match is not None
    assert window.closest_match.exact_copy is False
    assert (
        payload["metrics"]["window_membership_inference_protection"]["details"]["closest_match"]["training_values"]
        is None
    )


def test_time_series_export_caps_group_rows_at_what_the_report_shows() -> None:
    per_group = [
        {"group": f"g{index:02d}", "windows": 20, "exact_copy_rate": 0.0, "flag_rate": index / 20}
        for index in range(15)
    ]
    skipped = [{"group": f"g{index:02d}", "column": "value", "reason": "too short"} for index in range(30)]
    report = _report_with(
        WindowGroupMIA(score=EvaluationScore(), details={"evaluation_mode": "full", "per_group": per_group}),
        AutocorrelationSimilarity(score=EvaluationScore(), details={"skipped": skipped}),
    )

    metrics = build_report_data(report).metrics

    assert metrics.window_membership_inference_protection is not None
    window = metrics.window_membership_inference_protection.details
    assert window.training_group_count == 15
    assert [group.group for group in window.per_group] == [f"g{index:02d}" for index in range(14, 2, -1)]
    assert metrics.autocorrelation_similarity is not None
    assert len(metrics.autocorrelation_similarity.details.skipped) == 12


def test_time_series_export_keeps_an_overflowing_distance_as_null() -> None:
    attack = {
        "threshold": float("inf"),
        "training_flag_rate": 0.1,
        "holdout_flag_rate": 0.1,
        "privacy_loss_beyond_chance": 0.0,
        "holdout_exact_match_rate": 0.0,
        "exact_copy_privacy_loss": 0.0,
    }
    match = {"distance": float("inf"), "exact_copy": False, "synthetic_values": [[1.0]], "training_values": [[2.0]]}
    report = _report_with(
        WindowGroupMIA(
            score=EvaluationScore(),
            details={"evaluation_mode": "full", "attack": attack, "closest_match": match, "per_group": []},
        )
    )

    metric = build_report_data(report).metrics.window_membership_inference_protection

    assert metric is not None
    assert metric.details.attack is not None
    assert metric.details.attack.threshold is None
    assert metric.details.closest_match is not None
    assert metric.details.closest_match.distance is None
    assert metric.details.closest_match.training_values is None


class _Color(Enum):
    RED = "red"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), None),
        (np.array([1.0, np.inf]), [1.0, None]),
        (_Color.RED, "red"),
        (pd.Timestamp("2026-01-02T03:04:05"), "2026-01-02T03:04:05"),
        (pd.NaT, None),
    ],
)
def test_to_json_safe_converts_to_strict_json(value: object, expected: object) -> None:
    converted = to_json_safe(value)

    assert converted == expected
    json.dumps(converted, allow_nan=False)


def test_to_json_safe_rejects_unknown_types() -> None:
    with pytest.raises(TypeError, match="object"):
        to_json_safe(object())
