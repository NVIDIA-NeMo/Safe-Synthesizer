# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Tuple

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
from nemo_safe_synthesizer.generation.results import GenerationBatches, GenerationStatus


# Purpose: Builds reusable good/bad Batch sets for generation tests.
# Data: good: one batch with 2 prompts (3 valid each) and another with 1 prompt; bad: one batch with 0 valid.
# Output: Tuple[List[Batch], List[Batch]]. This fixture is used in the following tests:
# - test_generation_add_batch_in_progress_status
# - test_generation_add_batch_stop_no_records_status_last_batch
# - test_generation_add_batch_stop_metric_reached_status
@pytest.fixture()
def fixture_stub_batches(
    fixture_mock_processor, fixture_mock_processor_without_valid_records
) -> Tuple[List[Batch], List[Batch]]:
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
