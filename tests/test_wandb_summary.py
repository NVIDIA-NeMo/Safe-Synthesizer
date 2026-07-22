# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""W&B final-scalar summary contracts."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from nemo_safe_synthesizer.config.external_results import SafeSynthesizerSummary, SafeSynthesizerTiming


def test_summary_log_wandb_updates_run_summary_without_history() -> None:
    """Final scalar metrics update W&B summary and preserve ``None`` values."""
    summary = SafeSynthesizerSummary(
        timing=SafeSynthesizerTiming(generation_time_sec=None),
        num_completion_tokens=None,
    )
    mock_wandb = MagicMock()
    mock_wandb.run = MagicMock()

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        summary.log_wandb()

    mock_wandb.log.assert_not_called()
    payload = mock_wandb.run.summary.update.call_args.args[0]
    assert payload["gen/generation_time_sec"] is None
    assert payload["gen/num_completion_tokens"] is None


def test_timing_log_wandb_updates_summary_and_preserves_none() -> None:
    """Timing-only W&B output also avoids history and does not coerce ``None``."""
    timing = SafeSynthesizerTiming(evaluation_time_sec=None)
    run = MagicMock()

    timing.log_wandb(run)

    run.log.assert_not_called()
    assert run.summary.update.call_args.args[0]["evaluation_time_sec"] is None


def test_summary_log_wandb_failure_is_best_effort() -> None:
    """Final summary publication cannot turn a successful synthesis into a failure."""
    summary = SafeSynthesizerSummary(timing=SafeSynthesizerTiming())
    mock_wandb = MagicMock()
    mock_wandb.run.summary.update.side_effect = RuntimeError("wandb down")

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        summary.log_wandb()


def test_timing_log_wandb_failure_is_best_effort() -> None:
    """Timing summary publication has the same non-fatal contract."""
    run = MagicMock()
    run.summary.update.side_effect = RuntimeError("wandb down")
    SafeSynthesizerTiming().log_wandb(run)
