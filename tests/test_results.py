# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd

from nemo_safe_synthesizer.config import SafeSynthesizerTiming
from nemo_safe_synthesizer.generation.results import GenerateJobResults
from nemo_safe_synthesizer.generation.utils import GenerationStatus
from nemo_safe_synthesizer.results import make_nss_summary


def _stub_generate_results(*, with_token_stats: bool = True) -> GenerateJobResults:
    """Build a minimal GenerateJobResults, optionally with token stats."""
    if with_token_stats:
        return GenerateJobResults(
            df=pd.DataFrame({"col": [1, 2]}),
            status=GenerationStatus.COMPLETE,
            num_valid_records=2,
            num_invalid_records=1,
            num_prompts=4,
            valid_record_fraction=2 / 3,
            batch_valid_record_fractions=[0.67],
            elapsed_time=2.0,
            num_completion_tokens=1000,
            num_valid_record_tokens=700,
            num_invalid_record_tokens=200,
            num_non_record_tokens=100,
            tokens_per_prompt=250.0,
            tokens_per_second=500.0,
            valid_tokens_per_second=350.0,
            tokenization_overhead_sec=0.05,
        )

    return GenerateJobResults(
        df=pd.DataFrame({"col": [1, 2]}),
        status=GenerationStatus.COMPLETE,
        num_valid_records=2,
        num_invalid_records=1,
        num_prompts=4,
        valid_record_fraction=2 / 3,
        batch_valid_record_fractions=[0.67],
        elapsed_time=2.0,
    )


class TestMakeNssSummaryTokenStats:
    """Verify make_nss_summary propagates token stats from GenerateJobResults."""

    def test_propagates_token_stats(self):
        timing = SafeSynthesizerTiming(generation_time_sec=2.0)
        results = _stub_generate_results(with_token_stats=True)
        summary = make_nss_summary(timing, results)

        assert summary.num_completion_tokens == 1000
        assert summary.num_valid_record_tokens == 700
        assert summary.num_invalid_record_tokens == 200
        assert summary.num_non_record_tokens == 100
        assert summary.valid_record_token_fraction == 700 / 1000
        assert summary.tokens_per_prompt == 250.0
        assert summary.tokens_per_second == 500.0
        assert summary.valid_tokens_per_second == 350.0
        assert summary.tokenization_overhead_sec == 0.05

    def test_no_token_stats_leaves_fields_none(self):
        timing = SafeSynthesizerTiming(generation_time_sec=2.0)
        results = _stub_generate_results(with_token_stats=False)
        summary = make_nss_summary(timing, results)

        assert summary.num_completion_tokens is None
        assert summary.tokens_per_second is None
        assert summary.valid_tokens_per_second is None

    def test_dataframe_input_leaves_token_fields_none(self):
        timing = SafeSynthesizerTiming(generation_time_sec=2.0)
        summary = make_nss_summary(timing, pd.DataFrame({"col": [1]}))

        assert summary.num_completion_tokens is None
        assert summary.tokens_per_second is None

    def test_none_input_leaves_token_fields_none(self):
        timing = SafeSynthesizerTiming(generation_time_sec=2.0)
        summary = make_nss_summary(timing, None)

        assert summary.num_completion_tokens is None


class TestSafeSynthesizerSummaryLogWandb:
    """Verify log_wandb includes token metrics in the W&B payload."""

    def test_log_wandb_includes_token_metrics(self):
        timing = SafeSynthesizerTiming(generation_time_sec=2.0)
        results = _stub_generate_results(with_token_stats=True)
        summary = make_nss_summary(timing, results)

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            summary.log_wandb()

        mock_wandb.log.assert_not_called()
        mock_wandb.run.summary.update.assert_called_once()
        logged = mock_wandb.run.summary.update.call_args.args[0]

        assert logged["gen/num_completion_tokens"] == 1000
        assert logged["gen/num_valid_record_tokens"] == 700
        assert logged["gen/num_invalid_record_tokens"] == 200
        assert logged["gen/num_non_record_tokens"] == 100
        assert logged["gen/valid_record_token_fraction"] == 700 / 1000
        assert logged["gen/tokens_per_prompt"] == 250.0
        assert logged["gen/tokens_per_second"] == 500.0
        assert logged["gen/valid_tokens_per_second"] == 350.0
        assert logged["gen/tokenization_overhead_sec"] == 0.05

    def test_log_wandb_logs_none_token_metrics(self):
        """None-valued token metrics are logged (not filtered) to keep dashboards consistent."""
        timing = SafeSynthesizerTiming(generation_time_sec=2.0)
        results = _stub_generate_results(with_token_stats=False)
        summary = make_nss_summary(timing, results)

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            summary.log_wandb()

        mock_wandb.log.assert_not_called()
        logged = mock_wandb.run.summary.update.call_args.args[0]
        assert logged["gen/num_completion_tokens"] is None
        assert logged["gen/tokens_per_second"] is None

    def test_log_wandb_noop_when_no_run(self):
        timing = SafeSynthesizerTiming(generation_time_sec=2.0)
        summary = make_nss_summary(timing, None)

        mock_wandb = MagicMock()
        mock_wandb.run = None
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            summary.log_wandb()

        mock_wandb.log.assert_not_called()
