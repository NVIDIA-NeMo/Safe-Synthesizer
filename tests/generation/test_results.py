# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``GenerationBatches`` probe-batch and adaptive sizing logic."""

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.config.generate import ValidationParameters
from nemo_safe_synthesizer.data_processing.record_utils import ParsedRecord
from nemo_safe_synthesizer.defaults import EPS, MAX_NUM_PROMPTS_PER_BATCH
from nemo_safe_synthesizer.generation.batch import Batch
from nemo_safe_synthesizer.generation.processors import ParsedResponse, TabularDataProcessor
from nemo_safe_synthesizer.generation.results import (
    ADAPTIVE_MAX_PROMPTS_CEILING,
    INITIAL_PROBE_PROMPTS,
    NUM_PROMPT_BUFFER,
    GenerationBatches,
)


@pytest.fixture
def fixture_processor() -> TabularDataProcessor:
    """Concrete ``TabularDataProcessor`` -- exercised only as the ``Batch`` owner."""
    return TabularDataProcessor(schema={"properties": {}}, config=ValidationParameters())


def _make_batch(
    processor: TabularDataProcessor,
    *,
    num_prompts: int,
    num_valid: int = 0,
    num_invalid: int = 0,
) -> Batch:
    """Return a ``Batch`` whose aggregate counts match the requested totals.

    Distributes ``num_valid`` valid records across the first prompt and
    ``num_invalid`` invalid records across the same prompt so that
    ``Batch.num_prompts == num_prompts`` regardless of the counts.
    """
    batch = Batch(processor=processor)
    for prompt_index in range(num_prompts):
        records: list[ParsedRecord] = []
        if prompt_index == 0:
            records.extend(
                ParsedRecord(text=f'{{"col": "valid-{prompt_index}-{i}"}}', parsed={"col": f"valid-{prompt_index}-{i}"})
                for i in range(num_valid)
            )
            records.extend(
                ParsedRecord(text=f"invalid-{prompt_index}-{i}", error=("validation failed", "schema"))
                for i in range(num_invalid)
            )
        batch._responses.append(ParsedResponse(records=records, prompt_number=prompt_index))
    return batch


class TestInitialProbeBatch:
    """The first batch from a fresh accumulator is sized to ``INITIAL_PROBE_PROMPTS``."""

    def test_results_first_batch_uses_probe_size(self):
        """Fresh accumulator with ``target_num_records`` returns the probe size."""
        batches = GenerationBatches(target_num_records=10_000)

        assert batches.get_next_num_prompts() == INITIAL_PROBE_PROMPTS

    def test_results_first_batch_clipped_by_remaining_records(self):
        """The probe count is also bounded by ``records_remaining + NUM_PROMPT_BUFFER``."""
        batches = GenerationBatches(target_num_records=3)

        # The records-remaining bound is ``3 + NUM_PROMPT_BUFFER`` -- larger than the probe,
        # so the probe still wins; this test guards the three-way ``min`` against regressions.
        assert batches.get_next_num_prompts() == min(INITIAL_PROBE_PROMPTS, 3 + NUM_PROMPT_BUFFER)

    def test_results_first_batch_without_target_returns_full_budget(self):
        """Without ``target_num_records``, the helper returns the full per-batch cap (no probe)."""
        batches = GenerationBatches(target_num_records=None)

        assert batches.get_next_num_prompts() == batches.max_num_prompts_per_batch


class TestEscalateAfterFailedProbe:
    """A probe that produced no valid records escalates to the full prompt budget."""

    def test_results_escalates_to_full_budget_after_zero_valid_probe(self, fixture_processor):
        """Stopping is intentionally disabled so the next call exercises the escalation branch."""
        batches = GenerationBatches(
            target_num_records=10_000,
            invalid_fraction_threshold=1.0,
            patience=10,
        )
        first_batch = _make_batch(fixture_processor, num_prompts=INITIAL_PROBE_PROMPTS, num_valid=0, num_invalid=5)
        batches.add_batch(first_batch)

        next_count = batches.get_next_num_prompts()

        # Escalation: full ``max_num_prompts_per_batch`` clipped only by remaining records.
        records_remaining = 10_000 - batches.num_valid_records
        assert next_count == min(batches.max_num_prompts_per_batch, records_remaining + NUM_PROMPT_BUFFER)
        assert next_count > INITIAL_PROBE_PROMPTS

    def test_results_uses_records_per_prompt_estimate_after_successful_probe(self, fixture_processor):
        """Once valid records exist, sizing follows the records-per-prompt estimate."""
        batches = GenerationBatches(target_num_records=1_000)
        # 5 prompts * 1 valid record each = 5 valid; future budget = remaining / 1 per prompt.
        first_batch = _make_batch(fixture_processor, num_prompts=5, num_valid=5, num_invalid=0)
        batches.add_batch(first_batch)

        next_count = batches.get_next_num_prompts()

        records_remaining = 1_000 - 5
        valid_per_prompt = 5 / 5  # 1.0
        expected = min(
            batches.max_num_prompts_per_batch,
            round(records_remaining / (valid_per_prompt + EPS)) + NUM_PROMPT_BUFFER,
        )
        assert next_count == expected


class TestAdaptiveMaxNumPromptsPerBatch:
    """``max_num_prompts_per_batch=None`` derives the cap from ``target_num_records``."""

    @pytest.mark.parametrize(
        "target_num_records, expected_cap",
        [
            pytest.param(None, MAX_NUM_PROMPTS_PER_BATCH, id="no_target_uses_default"),
            pytest.param(500, MAX_NUM_PROMPTS_PER_BATCH, id="small_target_floors_at_default"),
            pytest.param(1_000, MAX_NUM_PROMPTS_PER_BATCH, id="exactly_at_floor"),
            pytest.param(5_000, 250, id="moderate_target_scales_up"),
            pytest.param(100_000, ADAPTIVE_MAX_PROMPTS_CEILING, id="huge_target_capped_at_ceiling"),
        ],
    )
    def test_results_adaptive_default_scales_with_target(self, target_num_records, expected_cap):
        """Adaptive default = ``min(2000, max(MAX_NUM_PROMPTS_PER_BATCH, target_num_records // 20))``."""
        batches = GenerationBatches(target_num_records=target_num_records)

        assert batches.max_num_prompts_per_batch == expected_cap

    @pytest.mark.parametrize(
        "explicit_value",
        [42, 1, MAX_NUM_PROMPTS_PER_BATCH, 5_000],
    )
    def test_results_explicit_max_num_prompts_per_batch_is_honored(self, explicit_value):
        """Backwards-compat: an explicit value bypasses the adaptive derivation."""
        batches = GenerationBatches(
            target_num_records=10_000,
            max_num_prompts_per_batch=explicit_value,
        )

        assert batches.max_num_prompts_per_batch == explicit_value
