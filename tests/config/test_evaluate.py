# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.evaluate import EvaluationParameters, TimeSeriesEvaluationParameters
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
