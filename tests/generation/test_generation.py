# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pandas as pd
import pytest

from nemo_safe_synthesizer.data_processing.actions.data_actions import (
    ActionExecutor,
    CategoricalCol,
    DateConstraint,
    data_actions_fn,
)
from nemo_safe_synthesizer.errors import GenerationError
from nemo_safe_synthesizer.generation.batch import Batch
from nemo_safe_synthesizer.generation.processors import ParsedResponse
from nemo_safe_synthesizer.generation.results import NUM_PROMPT_BUFFER, GenerationBatches, GenerationStatus


# Purpose: Builds reusable good/bad Batch sets for generation tests.
# Data: good: one batch with 2 prompts (3 valid each) and another with 1 prompt; bad: one batch with 0 valid.
# Output: Tuple[List[Batch], List[Batch]]. This fixture is used in the following tests:
# - test_generation_add_batch_in_progress_status
# - test_generation_add_batch_stop_no_records_status_last_batch
# - test_generation_add_batch_stop_metric_reached_status
@pytest.fixture()
def fixture_stub_batches(
    fixture_mock_processor, fixture_mock_processor_without_valid_records
) -> tuple[list[Batch], list[Batch]]:
    batch_1 = Batch(fixture_mock_processor)
    batch_1.process(1, "stub")
    batch_1.process(2, "stub")
    batch_2 = Batch(fixture_mock_processor)
    batch_2.process(3, "stub")

    batch_3 = Batch(fixture_mock_processor_without_valid_records)
    batch_3.process(4, "stub")
    return [batch_1, batch_2], [batch_3]


# Purpose: Adding a good batch updates counts and keeps status IN_PROGRESS.
# Data: Start with one good batch; add another good batch.
# Asserts: num_batches=2, num_prompts=3, invalid=3, valid=9, status IN_PROGRESS.
def test_generation_add_batch_in_progress_status(fixture_stub_batches):
    good_batches, bad_batches = fixture_stub_batches
    generation = GenerationBatches(target_num_records=50, batches=[good_batches[0]])
    generation.add_batch(good_batches[1])
    assert generation.num_batches == 2
    assert generation.num_prompts == 3
    assert generation.num_invalid_records == 3
    assert generation.num_valid_records == 9
    assert generation.status == GenerationStatus.IN_PROGRESS


# Purpose: When last added batch yields no records, status is STOP_NO_RECORDS.
# Data: Start with good batch; add bad batch (0 valid).
# Asserts: status STOP_NO_RECORDS.
def test_generation_add_batch_stop_no_records_status_last_batch(fixture_stub_batches):
    good_batches, bad_batches = fixture_stub_batches
    generation = GenerationBatches(target_num_records=50, batches=[good_batches[0]])
    generation.add_batch(bad_batches[0])
    assert generation.status == GenerationStatus.STOP_NO_RECORDS


# Purpose: Stop metric reached based on thresholds/targets after adding a good batch.
# Data: Low invalid threshold and small target.
# Asserts: status STOP_METRIC_REACHED.
def test_generation_add_batch_stop_metric_reached_status(fixture_stub_batches):
    good_batches, _ = fixture_stub_batches
    generation_with_stop_params = GenerationBatches(
        target_num_records=5,
        invalid_fraction_threshold=0.01,
        patience=1,
    )
    generation_with_stop_params.add_batch(good_batches[0])
    assert generation_with_stop_params.status == GenerationStatus.STOP_METRIC_REACHED


# Purpose: First batch yields no records under lenient patience; stop early with STOP_NO_RECORDS.
# Data: Only bad batch added first.
# Asserts: status STOP_NO_RECORDS.
def test_generation_add_batch_stop_no_records_status_first_batch(fixture_stub_batches):
    _, bad_batches = fixture_stub_batches

    generation_with_stop_params = GenerationBatches(
        target_num_records=5,
        invalid_fraction_threshold=0.9,
        patience=3,
    )
    generation_with_stop_params.add_batch(bad_batches[0])
    assert generation_with_stop_params.status == GenerationStatus.STOP_NO_RECORDS


# Purpose: Sequence good → bad → good under stricter policy triggers STOP_METRIC_REACHED mid-flight.
# Data: Threshold 0.2, patience 3.
# Asserts: status STOP_METRIC_REACHED after third add.
def test_generation_add_batch_stop_metric_reached_status_middle_batch_all_invalid(
    fixture_stub_batches,
):
    good_batches, bad_batches = fixture_stub_batches
    generation_with_stop_params = GenerationBatches(
        target_num_records=50,
        invalid_fraction_threshold=0.2,
        patience=3,
    )
    generation_with_stop_params.add_batch(good_batches[0])
    generation_with_stop_params.add_batch(bad_batches[0])
    generation_with_stop_params.add_batch(good_batches[1])
    assert generation_with_stop_params.status == GenerationStatus.STOP_METRIC_REACHED


# Purpose: Under lenient policy, same sequence good → bad → good remains IN_PROGRESS.
# Data: Threshold 0.9, patience 3.
# Asserts: status IN_PROGRESS after third add.
def test_generation_add_batch_in_progress_status_middle_batch_all_invalid(
    fixture_stub_batches,
):
    good_batches, bad_batches = fixture_stub_batches
    generation_with_stop_params = GenerationBatches(
        target_num_records=50,
        invalid_fraction_threshold=0.9,
        patience=3,
    )
    generation_with_stop_params.add_batch(good_batches[0])
    generation_with_stop_params.add_batch(bad_batches[0])
    generation_with_stop_params.add_batch(good_batches[1])
    assert generation_with_stop_params.status == GenerationStatus.IN_PROGRESS


# Purpose: Finalize job status based on target satisfaction.
# Data: Cases: target satisfied; no target; target not met.
# Asserts: COMPLETE for satisfied/no-target; INCOMPLETE otherwise.
def test_job_complete(fixture_stub_batches):
    good_batches, bad_batches = fixture_stub_batches
    generation_complete = GenerationBatches(target_num_records=2, batches=[good_batches[0]])
    generation_complete.job_complete()
    assert generation_complete.status == GenerationStatus.COMPLETE

    generation_complete_no_target = GenerationBatches(batches=[good_batches[0]])
    generation_complete_no_target.job_complete()
    assert generation_complete_no_target.status == GenerationStatus.COMPLETE

    generation_incomplete = GenerationBatches(target_num_records=20, batches=[good_batches[0]])
    generation_incomplete.job_complete()
    assert generation_incomplete.status == GenerationStatus.INCOMPLETE


# Purpose: Compute next number of prompts to request based on target and previous yield.
# Data: No target → default buffer; With target → estimate using 3 records/prompt and buffer minimums.
# Asserts: 100 by default; 16 with target and prior yield.
def test_get_next_num_prompts(fixture_stub_batches):
    good_batches, bad_batches = fixture_stub_batches

    generation_no_target = GenerationBatches(batches=[good_batches[0]])
    assert generation_no_target.get_next_num_prompts() == 100

    generation_with_target = GenerationBatches(target_num_records=20, batches=[good_batches[1]])
    # first batch with one prompt generated 3 good records, so we need 17 more
    # so far the average is 3 records per prompt so we only need around 6 more but
    # we are expecting a higher number of prompts because of the NUM_PROMPT_BUFFER minimum
    assert generation_with_target.get_next_num_prompts() == 16


# Purpose: First batch (no history) caps prompts to target + buffer instead of max.
# Data: Empty GenerationBatches with small target_num_records and no prior batches.
# Asserts: Returns target + NUM_PROMPT_BUFFER when that's less than max; returns max otherwise.
@pytest.mark.parametrize(
    "target, expected",
    [
        (10, 10 + NUM_PROMPT_BUFFER),
        (50, 50 + NUM_PROMPT_BUFFER),
        (200, 100),  # target + buffer exceeds max (100), so capped
    ],
)
def test_get_next_num_prompts_first_batch(target, expected):
    generation = GenerationBatches(target_num_records=target)
    assert generation.get_next_num_prompts() == expected


# Purpose: Build a DataFrame of valid records across batches, honoring max record cap and validity.
# Data: Only invalid → empty; only valid → all rows; mixed → valid-only; with max_num_records enforced.
# Asserts: Columns preserved; lengths are 0, 9, 6; cap respected at 2.
def test_to_dataframe(fixture_stub_batches):
    good_batches, bad_batches = fixture_stub_batches
    generation_no_records = GenerationBatches(batches=bad_batches)
    assert len(generation_no_records.to_dataframe(columns=["some"])) == 0

    generation_with_valid_records = GenerationBatches(batches=good_batches)
    df = generation_with_valid_records.to_dataframe(columns=["some"])
    assert list(df.columns.values)[0] == "some"
    assert len(df) == 9

    generation_with_valid_and_invalid_records = GenerationBatches(batches=[good_batches[0], bad_batches[0]])
    df = generation_with_valid_and_invalid_records.to_dataframe(columns=["some"])
    # Only valid records should be included in the dataframe:
    assert list(df.columns.values)[0] == "some"
    assert len(df) == 6

    df_with_max_records = generation_with_valid_records.to_dataframe(columns=["some"], max_num_records=2)
    assert len(df_with_max_records) == 2


# Purpose: Integration of data_actions filtering and logging with Generation/Batches.
# Data: Two prompts with records; actions: CategoricalCol on role, DateConstraint enforcing start_date < end_date.
# Asserts: Log mentions date_constraint; final counts: 2 valid, 3 invalid.
def test_apply_data_actions(fixture_mock_processor, caplog):
    caplog.set_level(logging.INFO)

    # We expect only 2 valid records, with the 0th and 1st records being
    # rejected due to DateConstraint, and `contractor` being rejected
    # as a valid CategoricalCl
    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "first_name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "start_date": pd.date_range("2024-01-01", periods=5),
            "end_date": pd.date_range("2024-05-01", periods=5),
            "role": ["intern", "fulltime", "intern", "contractor", "fulltime"],
        }
    )
    data.loc[0, "end_date"] = pd.to_datetime("2023-01-01")
    data.loc[1, "end_date"] = pd.to_datetime("2023-01-01")
    batch = Batch(fixture_mock_processor)
    batch._responses = [
        ParsedResponse(
            valid_records=data.iloc[:3].to_dict("records"),
            invalid_records=[],
            errors=[],
            prompt_number=1,
        ),
        ParsedResponse(
            valid_records=data.iloc[3:].to_dict("records"),
            invalid_records=[],
            errors=[],
            prompt_number=2,
        ),
    ]

    action_executor = ActionExecutor(
        actions=[
            CategoricalCol(name="role", values=["intern", "fulltime"]),
            DateConstraint(colA="start_date", colB="end_date", operator="lt"),
        ]
    )

    daf = data_actions_fn(action_executor)
    generation = GenerationBatches(data_actions_fn=daf)
    generation.add_batch(batch)
    batch.log_summary()

    # Extract ctx data from log records (logging passes extra data to record attributes)
    ctx_data = [getattr(record, "ctx", None) for record in caplog.records if hasattr(record, "ctx")]
    # Find error statistics record (contains tabular_data with error messages as keys)
    error_ctx = next((ctx for ctx in ctx_data if ctx and ctx.get("title") == "Error Statistics"), None)
    assert error_ctx is not None, "Expected error statistics log record"
    error_data = error_ctx["tabular_data"]
    assert any("Failed data_config validation due to [date_constraint]" in key for key in error_data)

    assert generation.num_valid_records == 2
    assert generation.num_invalid_records == 3


# Purpose: When early stop condition is reached, log_status raises GenerationError.
# Data: Sequence good → bad → good under stricter policy triggers STOP_METRIC_REACHED.
# Asserts: GenerationError is raised by log_status.
def test_log_status_raises_generation_error_on_early_stop(fixture_stub_batches):
    good_batches, bad_batches = fixture_stub_batches
    generation = GenerationBatches(
        target_num_records=50,
        invalid_fraction_threshold=0.2,
        patience=3,
    )
    generation.add_batch(good_batches[0])
    generation.add_batch(bad_batches[0])
    generation.add_batch(good_batches[1])
    assert generation.status == GenerationStatus.STOP_METRIC_REACHED
    with pytest.raises(GenerationError):
        generation.log_status()


# Purpose: When no valid records are produced in the first batch, log_status raises GenerationError.
# Data: Start with a bad batch (0 valid), no stop_params required.
# Asserts: Status is STOP_NO_RECORDS and log_status raises GenerationError.
def test_log_status_raises_generation_error_on_no_records(fixture_stub_batches):
    _, bad_batches = fixture_stub_batches
    generation = GenerationBatches(target_num_records=5)
    generation.add_batch(bad_batches[0])
    assert generation.status == GenerationStatus.STOP_NO_RECORDS
    with pytest.raises(GenerationError):
        generation.log_status()


# ---------------------------------------------------------------------------
# Token-aggregation tests
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock

from nemo_safe_synthesizer.generation.results import GenerateJobResults


def _make_batch_with_tokens(
    valid_counts: list[int],
    invalid_counts: list[int],
    completion_tokens: int,
    tok_time: float = 0.01,
) -> Batch:
    """Build a Batch with a mock processor that returns specific token counts."""
    n_valid = len(valid_counts)
    n_invalid = len(invalid_counts)
    mock_proc = MagicMock()
    mock_proc.return_value = ParsedResponse(
        valid_records=[{"a": i} for i in range(n_valid)],
        invalid_records=[f"bad{i}" for i in range(n_invalid)],
        errors=[("err", "err")] * n_invalid,
        valid_record_token_counts=valid_counts,
        invalid_record_token_counts=invalid_counts,
        tokenization_time_sec=tok_time,
    )
    batch = Batch(processor=mock_proc)
    batch.process(0, "stub", completion_tokens=completion_tokens)
    return batch


class TestGenerationBatchesTokenAggregation:
    """GenerationBatches aggregates token properties across batches."""

    def test_total_completion_tokens(self):
        b1 = _make_batch_with_tokens([10, 20], [5], completion_tokens=100)
        b2 = _make_batch_with_tokens([30], [10, 15], completion_tokens=200)
        gen = GenerationBatches(batches=[b1, b2])
        assert gen.total_completion_tokens == 300

    def test_total_valid_record_tokens(self):
        b1 = _make_batch_with_tokens([10, 20], [5], completion_tokens=100)
        b2 = _make_batch_with_tokens([30], [10], completion_tokens=200)
        gen = GenerationBatches(batches=[b1, b2])
        assert gen.total_valid_record_tokens == 60  # 10+20+30

    def test_total_invalid_record_tokens(self):
        b1 = _make_batch_with_tokens([10], [5, 15], completion_tokens=100)
        gen = GenerationBatches(batches=[b1])
        assert gen.total_invalid_record_tokens == 20

    def test_total_non_record_tokens(self):
        b1 = _make_batch_with_tokens([10], [5], completion_tokens=100)
        gen = GenerationBatches(batches=[b1])
        assert gen.total_non_record_tokens == 85  # 100 - 10 - 5

    def test_total_tokenization_time_sec(self):
        b1 = _make_batch_with_tokens([10], [5], completion_tokens=100, tok_time=0.01)
        b2 = _make_batch_with_tokens([20], [10], completion_tokens=200, tok_time=0.02)
        gen = GenerationBatches(batches=[b1, b2])
        assert gen.total_tokenization_time_sec == pytest.approx(0.03)


class TestGenerateJobResultsFromBatches:
    """GenerateJobResults.from_batches populates token fields correctly."""

    def test_with_token_data(self):
        b = _make_batch_with_tokens([10, 20], [5], completion_tokens=100, tok_time=0.05)
        gen = GenerationBatches(batches=[b])
        gen.job_complete()
        results = GenerateJobResults.from_batches(gen, max_num_records=None, columns=["a"], elapsed_time=2.0)

        assert results.num_completion_tokens == 100
        assert results.num_valid_record_tokens == 30
        assert results.num_invalid_record_tokens == 5
        assert results.num_non_record_tokens == 65
        assert results.tokens_per_completion == pytest.approx(100.0)  # 100/1 prompt
        assert results.tokens_per_second == pytest.approx(50.0)  # 100/2.0
        assert results.valid_tokens_per_second == pytest.approx(15.0)  # 30/2.0
        assert results.tokenization_overhead_sec == pytest.approx(0.05)

    def test_without_token_data(self):
        """When completion_tokens=0, all token fields stay None."""
        b = _make_batch_with_tokens([], [], completion_tokens=0)
        gen = GenerationBatches(batches=[b])
        gen.job_complete()
        results = GenerateJobResults.from_batches(gen, max_num_records=None, columns=["a"], elapsed_time=2.0)

        assert results.num_completion_tokens is None
        assert results.tokens_per_second is None
        assert results.valid_tokens_per_second is None
        assert results.tokenization_overhead_sec is None
