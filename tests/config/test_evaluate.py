# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.evaluate import EvaluationParameters, TimeSeriesEvaluationParameters
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters


def test_time_series_evaluation_is_disabled_by_default() -> None:
    assert EvaluationParameters().time_series.enabled is False


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
