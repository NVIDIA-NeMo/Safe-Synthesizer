# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nemo_safe_synthesizer.config.evaluate import (
    AutocorrelationSimilarityParameters,
    EvaluationParameters,
    TimeSeriesEvaluationParameters,
)
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters
from nemo_safe_synthesizer.defaults import PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.evaluation.components.autocorrelation_similarity import AutocorrelationSimilarity
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets
from nemo_safe_synthesizer.training.timeseries_preprocessing import process_timeseries_data


def _config(metric: AutocorrelationSimilarityParameters | None = None) -> SafeSynthesizerParameters:
    return SafeSynthesizerParameters(
        time_series=TimeSeriesParameters(is_timeseries=True, timestamp_column="time"),
        evaluation=EvaluationParameters(
            time_series=TimeSeriesEvaluationParameters(autocorrelation=metric or AutocorrelationSimilarityParameters())
        ),
    )


def _datasets(training_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> EvaluationDatasets:
    return EvaluationDatasets.from_dataframes(training_df, synthetic_df, enable_sampling=False)


def _grouped_rows(groups: list[tuple[str, int]], points: int = 12) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for group, offset in groups:
        for index in range(points):
            rows.append({"group": group, "time": index, "value": offset + index})
    return rows


def test_autocorrelation_similarity_formula_matches_mean_absolute_acf_difference_divided_by_two():
    training_df = pd.DataFrame({"time": range(8), "value": [0, 1, 2, 3, 2, 1, 0, -1]})
    synthetic_df = pd.DataFrame({"time": range(8), "value": [0, 1, 0, -1, 0, 1, 0, -1]})
    component = AutocorrelationSimilarity.from_evaluation_datasets(_datasets(training_df, synthetic_df), _config())

    atomic = component.details["atomics"][0]
    expected = np.mean(np.abs(np.array(atomic["training_acf"]) - np.array(atomic["synthetic_acf"]))) / 2.0
    assert atomic["error"] == pytest.approx(expected, abs=1e-6)
    assert component.score.score == pytest.approx(10 * (1 - expected), abs=0.1)


def test_autocorrelation_similarity_identical_grouped_series_are_scored_atomically():
    training_df = pd.DataFrame(_grouped_rows([("B", 100), ("A", 0)])).sample(frac=1.0, random_state=7)
    config = _config(AutocorrelationSimilarityParameters(group_column="group"))
    component = AutocorrelationSimilarity.from_evaluation_datasets(_datasets(training_df, training_df.copy()), config)

    assert component.score.score == 10
    assert component.details["counts"]["groups"] == 2
    assert component.details["counts"]["atomic_scores"] == 2
    assert [row["group"] for row in component.details["per_group"]] == ["A", "B"]


def test_autocorrelation_similarity_treats_inherited_pseudo_group_as_global_sequence():
    training_df = pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0, 2.0, 1.0]})
    config = SafeSynthesizerParameters.from_params(
        is_timeseries=True,
        timestamp_interval_seconds=1,
        rope_scaling_factor=1,
    )
    processed_df, config = process_timeseries_data(training_df.copy(), config)
    synthetic_df = processed_df.drop(columns=PSEUDO_GROUP_COLUMN).sample(frac=1.0, random_state=7)

    component = AutocorrelationSimilarity.from_evaluation_datasets(
        _datasets(training_df, synthetic_df),
        config,
    )

    assert component.score.score == 10
    assert component.details["evaluation_mode"] == "global"
    assert component.details["group_column"] is None
    assert component.details["timestamp_column"] == "elapsed_seconds"
    assert component.details["atomics"][0]["training_acf"] == component.details["atomics"][0]["synthetic_acf"]


def test_autocorrelation_similarity_excludes_inherited_pseudo_group_from_value_columns():
    frame = pd.DataFrame(
        {
            PSEUDO_GROUP_COLUMN: [0] * 6,
            "time": range(6),
            "value": [0.0, 1.0, 2.0, 3.0, 2.0, 1.0],
        }
    )
    config = _config()
    config.data.group_training_examples_by = PSEUDO_GROUP_COLUMN

    component = AutocorrelationSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), config)

    assert component.score.score == 10
    assert [row["column"] for row in component.details["per_column"]] == ["value"]


def test_autocorrelation_similarity_reports_missing_explicit_group_column():
    frame = pd.DataFrame({"time": range(6), "value": [0.0, 1.0, 2.0, 3.0, 2.0, 1.0]})
    config = _config(AutocorrelationSimilarityParameters(group_column="missing_group"))

    component = AutocorrelationSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), config)

    assert component.score.score is None
    assert component.score.notes == "Configured group column 'missing_group' is missing from a dataset."


def test_autocorrelation_similarity_short_and_constant_series_are_unavailable_instead_of_perfect():
    training_df = pd.DataFrame({"time": range(4), "value": [1.0] * 4})
    component = AutocorrelationSimilarity.from_evaluation_datasets(
        _datasets(training_df, training_df.copy()), _config()
    )

    assert component.score.score is None
    assert component.details["skipped"][0]["reason"] == "training series is constant or near-constant"


def test_autocorrelation_similarity_handles_aligned_non_finite_gaps():
    training_df = pd.DataFrame({"time": range(7), "value": [0.0, 1.0, np.inf, 2.0, 3.0, 2.0, 1.0]})
    synthetic_df = pd.DataFrame({"time": range(7), "value": [0.0, 1.0, -np.inf, 2.0, 3.0, 2.0, 1.0]})

    component = AutocorrelationSimilarity.from_evaluation_datasets(
        _datasets(training_df, synthetic_df),
        _config(),
    )

    assert component.score.score == 10
    atomic = component.details["atomics"][0]
    assert np.isfinite(atomic["training_acf"]).all()
    assert np.isfinite(atomic["synthetic_acf"]).all()


def test_autocorrelation_similarity_preserves_non_finite_positions():
    training_df = pd.DataFrame({"time": range(8), "value": [0.0, np.nan, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0]})
    synthetic_df = pd.DataFrame({"time": range(8), "value": [0.0, 1.0, np.nan, 2.0, 3.0, 2.0, 1.0, 0.0]})

    component = AutocorrelationSimilarity.from_evaluation_datasets(
        _datasets(training_df, synthetic_df),
        _config(),
    )

    assert component.score.score is not None and component.score.score < 10
    atomic = component.details["atomics"][0]
    assert atomic["training_acf"] != atomic["synthetic_acf"]


def test_autocorrelation_similarity_scores_synthetic_constant_collapse_as_failure():
    training_df = pd.DataFrame(_grouped_rows([("A", 0), ("B", 100)]))
    synthetic_df = training_df.copy()
    synthetic_df.loc[synthetic_df["group"] == "B", "value"] = 100
    config = _config(AutocorrelationSimilarityParameters(group_column="group"))

    component = AutocorrelationSimilarity.from_evaluation_datasets(_datasets(training_df, synthetic_df), config)

    assert component.score.score == 5
    collapsed = next(item for item in component.details["atomics"] if item["group"] == "B")
    assert collapsed["similarity"] == 0
    assert collapsed["reason"] == "synthetic series is constant or near-constant"
    assert component.details["counts"]["skipped"] == 0


def test_autocorrelation_similarity_revalidates_usable_length_with_non_finite_values():
    frame = pd.DataFrame({"time": range(4), "value": [0.0, np.inf, np.nan, 1.0]})

    component = AutocorrelationSimilarity.from_evaluation_datasets(
        _datasets(frame, frame.copy()),
        _config(),
    )

    assert component.score.score is None
    assert component.details["skipped"][0]["reason"] == "fewer than 4 points"


def test_autocorrelation_similarity_explicit_false_disables_auto_enabled_metric():
    frame = pd.DataFrame({"time": range(5), "value": range(5)})
    config = _config(AutocorrelationSimilarityParameters(enabled=False))

    component = AutocorrelationSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), config)

    assert component.score.score is None
    assert component.score.notes is not None
    assert "disabled" in component.score.notes

    forced_config = SafeSynthesizerParameters(
        evaluation=EvaluationParameters(
            time_series=TimeSeriesEvaluationParameters(
                autocorrelation=AutocorrelationSimilarityParameters(
                    enabled=True,
                    value_columns=["value"],
                )
            )
        )
    )
    forced = AutocorrelationSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), forced_config)
    assert forced.score.score == 10


def test_autocorrelation_similarity_isolates_unexpected_metric_failures(monkeypatch):
    frame = pd.DataFrame({"time": range(5), "value": range(5)})
    datasets = _datasets(frame, frame.copy())

    def fail_sort(*_args, **_kwargs):
        raise RuntimeError("unexpected metric failure")

    monkeypatch.setattr(pd.DataFrame, "sort_values", fail_sort)

    config = _config(AutocorrelationSimilarityParameters(value_columns=["value"]))
    component = AutocorrelationSimilarity.from_evaluation_datasets(datasets, config)

    assert component.score.score is None
    assert component.score.notes == "unexpected metric failure"


def test_autocorrelation_similarity_preserves_bare_timestamp_column_override():
    config = SafeSynthesizerParameters.from_params(
        is_timeseries=True,
        group_training_examples_by="sequence",
        timestamp_column="event_time",
        rope_scaling_factor=1,
    )

    assert config.time_series.timestamp_column == "event_time"


def test_autocorrelation_similarity_group_cap_uses_deterministic_hash_selection():
    rows = []
    group_labels = ["A", "B", "C", "D", "E", "F"]
    for group_index, group in enumerate(group_labels):
        rows.extend(
            {
                "group": group,
                "time": index,
                "x": group_index * 10 + index,
                "y": (group_index * 10 + index) ** 2,
            }
            for index in range(6)
        )
    frame = pd.DataFrame(rows)
    config = _config(
        AutocorrelationSimilarityParameters(group_column="group", value_columns=["y"], max_groups=2, max_lag=2)
    )

    first = AutocorrelationSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), config)
    second = AutocorrelationSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), config)

    assert first.details == second.details
    selected_groups = [row["group"] for row in first.details["per_group"]]
    assert len(selected_groups) == 2
    assert selected_groups != group_labels[:2]
    assert [row["column"] for row in first.details["per_column"]] == ["y"]
    assert first.details["group_selection"] == {
        "shared_groups": 6,
        "evaluated_groups": 2,
        "omitted_groups": 4,
        "policy": "deterministic_hash",
    }
    assert first.score.notes is not None
    assert "Evaluated 2 of 6 shared groups" in first.score.notes


def test_autocorrelation_similarity_documentation_examples_cover_presentation_bands():
    time = np.arange(240)
    training_df = pd.DataFrame({"time": time, "value": np.sin(2 * np.pi * time / 8)})
    examples = {
        "high": np.sin(2 * np.pi * time / 8),
        "medium": np.sin(2 * np.pi * time / 16),
        "low": np.sin(2 * np.pi * time / 40),
    }
    config = _config(AutocorrelationSimilarityParameters(max_lag=5))

    scores = {
        label: AutocorrelationSimilarity.from_evaluation_datasets(
            _datasets(training_df, pd.DataFrame({"time": time, "value": values})), config
        ).score.score
        for label, values in examples.items()
    }

    assert scores["low"] is not None and scores["low"] < 5.0
    assert scores["medium"] is not None and 5.0 <= scores["medium"] < 7.0
    assert scores["high"] is not None and scores["high"] >= 7.0
