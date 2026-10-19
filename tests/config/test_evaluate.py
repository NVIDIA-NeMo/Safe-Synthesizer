# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.evaluate import (
    EvaluationParameters,
    RollingStatsSimilarityParameters,
    SpectralSimilarityParameters,
    TimeSeriesEvaluationParameters,
)
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters


def test_time_series_evaluation_gate_is_unset_by_default() -> None:
    assert EvaluationParameters().time_series.enabled is None


def test_time_series_evaluation_follows_time_series_mode_by_default() -> None:
    tabular = SafeSynthesizerParameters()
    time_series = SafeSynthesizerParameters(
        time_series=TimeSeriesParameters(is_timeseries=True, timestamp_interval_seconds=60),
    )

    assert tabular.time_series_evaluation_enabled is False
    assert time_series.time_series_evaluation_enabled is True


def test_time_series_evaluation_can_be_disabled_in_time_series_mode() -> None:
    config = SafeSynthesizerParameters(
        evaluation=EvaluationParameters(time_series=TimeSeriesEvaluationParameters(enabled=False)),
        time_series=TimeSeriesParameters(is_timeseries=True, timestamp_interval_seconds=60),
    )

    assert config.time_series_evaluation_enabled is False


def test_global_evaluation_disable_turns_off_time_series_evaluation() -> None:
    config = SafeSynthesizerParameters(
        evaluation=EvaluationParameters(enabled=False),
        time_series=TimeSeriesParameters(is_timeseries=True, timestamp_interval_seconds=60),
    )

    assert config.time_series_evaluation_enabled is False


def test_time_series_evaluation_requires_time_series_mode() -> None:
    with pytest.raises(ValidationError) as exc_info:
        SafeSynthesizerParameters(
            evaluation=EvaluationParameters(time_series=TimeSeriesEvaluationParameters(enabled=True))
        )
    error = exc_info.value.errors()[0]
    assert error["type"] == "value_error"
    assert "requires time_series.is_timeseries=True" in error["msg"]


def test_time_series_evaluation_accepts_time_series_mode() -> None:
    config = SafeSynthesizerParameters(
        evaluation=EvaluationParameters(time_series=TimeSeriesEvaluationParameters(enabled=True)),
        time_series=TimeSeriesParameters(is_timeseries=True, timestamp_interval_seconds=60),
    )

    assert config.evaluation.time_series.enabled is True


def test_global_evaluation_disable_overrides_invalid_time_series_gate_combination() -> None:
    config = SafeSynthesizerParameters(
        evaluation=EvaluationParameters(
            enabled=False,
            time_series=TimeSeriesEvaluationParameters(enabled=True),
        )
    )

    assert config.evaluation.enabled is False
    assert config.evaluation.time_series.enabled is True


def test_rolling_stats_parameters_default_to_all_numeric_columns_and_a_20_point_window() -> None:
    rolling_stats = EvaluationParameters().time_series.rolling_stats

    assert rolling_stats.value_columns is None
    assert rolling_stats.window_size == 20
    assert rolling_stats.min_points == 4
    assert rolling_stats.max_groups == 128


def test_rolling_stats_window_must_hold_at_least_three_points() -> None:
    with pytest.raises(ValidationError):
        RollingStatsSimilarityParameters(window_size=2)


def test_spectral_parameters_default_to_all_numeric_columns_and_cycles_up_to_128_points() -> None:
    spectral = EvaluationParameters().time_series.spectral

    assert spectral.value_columns is None
    assert spectral.max_cycle_length == 128
    assert spectral.min_points == 16
    assert spectral.max_groups == 128


@pytest.mark.parametrize("overrides", [{"max_cycle_length": 3}, {"min_points": 7}])
def test_spectral_parameters_reject_settings_too_small_for_a_spectrum(overrides: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        SpectralSimilarityParameters.model_validate(overrides)
