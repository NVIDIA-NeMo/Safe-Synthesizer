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
from nemo_safe_synthesizer.evaluation.components.autocorrelation_similarity import AutocorrelationSimilarity
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets


def _config(metric: AutocorrelationSimilarityParameters | None = None) -> SafeSynthesizerParameters:
    return SafeSynthesizerParameters(
        time_series=TimeSeriesParameters(is_timeseries=True, timestamp_column="time"),
        evaluation=EvaluationParameters(
            time_series=TimeSeriesEvaluationParameters(autocorrelation=metric or AutocorrelationSimilarityParameters())
        ),
    )


def _datasets(training_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> EvaluationDatasets:
    return EvaluationDatasets.from_dataframes(training_df, synthetic_df, enable_sampling=False)


def test_autocorrelation_similarity_formula_matches_mean_absolute_acf_difference_divided_by_two():
    training_df = pd.DataFrame({"time": range(8), "value": [0, 1, 2, 3, 2, 1, 0, -1]})
    synthetic_df = pd.DataFrame({"time": range(8), "value": [0, 1, 0, -1, 0, 1, 0, -1]})
    component = AutocorrelationSimilarity.from_evaluation_datasets(_datasets(training_df, synthetic_df), _config())

    atomic = component.details["atomics"][0]
    expected = np.mean(np.abs(np.array(atomic["reference_acf"]) - np.array(atomic["synthetic_acf"]))) / 2.0
    assert atomic["error"] == pytest.approx(expected, abs=1e-6)
    assert component.score.score == pytest.approx(10 * (1 - expected), abs=0.1)


def test_autocorrelation_similarity_identical_grouped_series_are_scored_atomically():
    training_df = pd.DataFrame(
        [
            {"group": group, "time": index, "value": offset + index}
            for group, offset in [("B", 100), ("A", 0)]
            for index in range(12)
        ]
    ).sample(frac=1.0, random_state=7)
    config = _config(AutocorrelationSimilarityParameters(group_column="group"))
    component = AutocorrelationSimilarity.from_evaluation_datasets(_datasets(training_df, training_df.copy()), config)

    assert component.score.score == 10
    assert component.details["counts"]["groups"] == 2
    assert component.details["counts"]["atomic_scores"] == 2
    assert [row["group"] for row in component.details["per_group"]] == ["A", "B"]


def test_autocorrelation_similarity_short_and_constant_series_are_unavailable_instead_of_perfect():
    training_df = pd.DataFrame({"time": range(4), "value": [1.0] * 4})
    component = AutocorrelationSimilarity.from_evaluation_datasets(
        _datasets(training_df, training_df.copy()), _config()
    )

    assert component.score.score is None
    assert component.details["skipped"][0]["reason"] == "constant or near-constant series"


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
                    timestamp_column="time",
                    value_columns=["value"],
                )
            )
        )
    )
    forced = AutocorrelationSimilarity.from_evaluation_datasets(_datasets(frame, frame.copy()), forced_config)
    assert forced.score.score == 10


def test_autocorrelation_similarity_column_and_group_caps_are_deterministic():
    rows = []
    for group_index, group in enumerate(["C", "A", "B"]):
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
    assert [row["group"] for row in first.details["per_group"]] == ["A", "B"]
    assert [row["column"] for row in first.details["per_column"]] == ["y"]


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
