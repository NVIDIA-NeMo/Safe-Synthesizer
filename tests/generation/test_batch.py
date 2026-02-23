# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from logging import INFO
from unittest import mock
from unittest.mock import MagicMock

import pandas as pd
import pytest

import nemo_safe_synthesizer
from nemo_safe_synthesizer.data_processing.actions.utils import MetadataColumns
from nemo_safe_synthesizer.generation.batch import Batch
from nemo_safe_synthesizer.generation.processors import ParsedResponse
from nemo_safe_synthesizer.generation.results import rejected_record_to_error


# Purpose: Provides a processor mock with a heavy distribution of invalid records and categorized errors
# to exercise error statistics and logging.
# Data: 2 valid records, 20 invalid_records ("invalidjson"), diverse error tuples including groupby, required, enum.
@pytest.fixture()
def fixture_mock_processor_errors():
    stub_valid_records = [
        dict(some="value0", other=1),
        dict(some="value1", other=2),
    ]
    mock_processor = MagicMock()
    mock_processor.return_value = ParsedResponse(
        valid_records=stub_valid_records,
        invalid_records=["invalidjson"] * 20,
        errors=[
            ("err2", "const"),
            ("err3", "Invalid JSON"),
            ("err4", "groupby"),
            ("err4", "groupby"),
            ("err4", "groupby"),
            ("err4", "groupby"),
            ("err4", "groupby"),
            ("'col1' is a required property", "required"),
            ("'col2' is a required property", "required"),
            ("'col3' is a required property", "required"),
            (
                "0 is not one of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32 33, 34, 35, 36, 37, 38, 39, 40]",
                "enum",
            ),
            (
                "0 is not one of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32 33, 34, 35, 36, 37, 38, 39, 40]",
                "enum",
            ),
            ("0 is not one of [4, 5, 6]", "enum"),
            ("1 is not one of [4, 5, 6]", "enum"),
            ("2 is not one of [4, 5, 6]", "enum"),
            ("3 is not one of [4, 5, 6]", "enum"),
            ("7 is not one of [4, 5, 6]", "enum"),
            ("8 is not one of [4, 5, 6]", "enum"),
            ("9 is not one of [4, 5, 6]", "enum"),
            ("10 is not one of [4, 5, 6]", "enum"),
        ],
        prompt_number=1,
    )
    return mock_processor


# Purpose: Extends the error-heavy processor with synthetically "rejected by data_config" records.
# Data: Adds 5 rejected records with reasons reason1/reason2 to invalid_records and errors.
@pytest.fixture
def fixture_mock_processor_rejected_records(fixture_mock_processor_errors: MagicMock):
    reject_reason = MetadataColumns.REJECT_REASON.value
    rejected_records = [
        {"a": "a1", reject_reason: "reason1"},
        {"a": "a2", reject_reason: "reason1"},
        {"a": "a3", reject_reason: "reason2"},
        {"a": "a4", reject_reason: "reason2"},
        {"a": "a5", reject_reason: "reason2"},
    ]
    errors = [rejected_record_to_error(r) for r in rejected_records]
    invalid_records = [str(r) for r in rejected_records]

    fixture_mock_processor_errors.return_value.invalid_records.extend(invalid_records)
    fixture_mock_processor_errors.return_value.errors.extend(errors)
    return fixture_mock_processor_errors


# Purpose: Validates Batch.process aggregates counts and derived metrics for a single prompt.
# Data: fixture_mock_processor yields 3 valid, 1 invalid per call.
# Asserts: 1 prompt processed; 3 valid, 1 invalid; fractions/metrics computed (0.75 valid, 0.25 stopping).
def test_batch_process(fixture_mock_processor):
    batch = Batch(processor=fixture_mock_processor)
    batch.process(prompt_number=1, text="stub text")
    assert batch.num_prompts == 1
    assert batch.num_invalid_records == 1
    assert batch.num_valid_records == 3
    assert round(batch.valid_record_fraction, 2) == 0.75
    assert round(batch.stopping_metric, 2) == 0.25


# Purpose: Validates Batch.to_dataframe concatenates valid records across prompts.
# Data: Two prompts, each with 3 valid records.
# Asserts: DataFrame length is 6 (valid-only).
def test_batch_to_dataframe(fixture_mock_processor):
    batch = Batch(processor=fixture_mock_processor)
    batch.process(prompt_number=1, text="stub text")
    batch.process(prompt_number=2, text="stub text")
    df = batch.to_dataframe()
    assert len(df) == 6


# Purpose: Ensures Batch.to_dataframe returns None when no valid records exist.
# Data: Processor returns 0 valid records.
# Asserts: to_dataframe() is None.
def test_batch_to_dataframe_without_valid_records(
    fixture_mock_processor_without_valid_records,
):
    batch = Batch(fixture_mock_processor_without_valid_records)
    batch.process(prompt_number=1, text="stub text")
    df = batch.to_dataframe()
    assert df is None


# Purpose: Verifies error aggregation in Batch.error_statistics for detailed vs summarized modes.
# Data: Error-heavy processor with crafted counts for grouping into categories.
# Asserts: DataFrame of percentages and labels matches expectations per mode.
@pytest.mark.parametrize("detailed_errors", [True, False])
def test_batch_error_statistics(detailed_errors, fixture_mock_processor_errors):
    batch = Batch(processor=fixture_mock_processor_errors)
    batch.process(prompt_number=1, text="stub text")
    if detailed_errors:
        expected = pd.DataFrame([(0.4,), (0.25,), (0.15,), (0.1,), (0.05,), (0.05,)], columns=["Percentage"])
        expected.index = [
            "Grouped error message: 0/1/10/2/3/... is not one of [4, 5, 6]",
            "err4",
            "Grouped error message: 'col1'/'col2'/'col3' is a required property",
            "0 is not one of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, ...",
            "err2",
            "err3",
        ]
    else:
        expected = pd.DataFrame([(0.55,), (0.25,), (0.15,), (0.05,)], columns=["Percentage"])
        expected.index = [
            "Invalid field value",
            "Groupby generation failed",
            "Missing required field",
            "Invalid JSON",
        ]
    result = batch.error_statistics(detailed_errors)

    pd.testing.assert_frame_equal(result, expected)


# Purpose: Validates Batch.log_summary emits counts, percentages, and error details depending on detail level.
# Data: Error-heavy processor.
# Asserts: Log contains correct counts and error summaries; emojis/labels differ per detailed mode.
@pytest.mark.parametrize("detailed_errors", [True, False])
def test_log_summary(detailed_errors, caplog, fixture_mock_processor_errors):
    caplog.set_level(INFO)
    batch = Batch(processor=fixture_mock_processor_errors)
    batch.process(prompt_number=1, text="stub text")
    batch.log_summary(detailed_errors=detailed_errors)

    # Extract ctx data from log records (logging passes extra data to record attributes)
    ctx_data = [getattr(record, "ctx", None) for record in caplog.records if hasattr(record, "ctx")]
    assert len(ctx_data) == 2, f"Expected 2 log records with ctx, got {len(ctx_data)}"

    # First record is the summary
    summary_ctx = ctx_data[0]
    assert summary_ctx["tabular_data"]["num_valid_records"] == 2
    assert summary_ctx["tabular_data"]["num_invalid_records"] == 20
    assert summary_ctx["tabular_data"]["valid_record_fraction"] == 0.09

    # Second record is the error statistics
    error_ctx = ctx_data[1]
    error_data = error_ctx["tabular_data"]
    if detailed_errors:
        assert "err2" in error_data
        assert "err3" in error_data
        assert "err4" in error_data
        assert "Grouped error message: 'col1'/'col2'/'col3' is a required property" in error_data
        assert "Grouped error message: 0/1/10/2/3/... is not one of [4, 5, 6]" in error_data
        assert (
            "0 is not one of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, ..."
            in error_data
        )
    else:
        assert "Invalid field value" in error_data
        assert "Invalid JSON" in error_data
        assert "Groupby generation failed" in error_data
        assert "Missing required field" in error_data


# Purpose: Ensures data_config rejection reasons are surfaced even when log is truncated.
# Data: Processor with added rejected records; LOG_NUM_ERRORS patched to 3.
# Asserts: Summary includes total valid/invalid and count of data_config-invalid; reasons displayed.
def test_log_summary_data_config(caplog, fixture_mock_processor_rejected_records):
    caplog.set_level(INFO)
    batch = Batch(processor=fixture_mock_processor_rejected_records)
    batch.process(prompt_number=1, text="stub text")

    # ensure we display data_config errors even if there's more than LOG_NUM_ERRORS
    with mock.patch.object(nemo_safe_synthesizer.generation.batch, "LOG_NUM_ERRORS", 3):
        batch.log_summary()

    # Extract ctx data from log records (logging passes extra data to record attributes)
    ctx_data = [getattr(record, "ctx", None) for record in caplog.records if hasattr(record, "ctx")]
    assert len(ctx_data) == 2, f"Expected 2 log records with ctx, got {len(ctx_data)}"

    # First record is the summary
    summary_ctx = ctx_data[0]
    assert summary_ctx["tabular_data"]["num_valid_records"] == 2
    assert summary_ctx["tabular_data"]["num_invalid_records"] == 25
    assert summary_ctx["tabular_data"]["num_data_config_rejected_records"] == 5

    # Second record is the error statistics - check for data_config rejection reason
    error_ctx = ctx_data[1]
    error_data = error_ctx["tabular_data"]
    assert any("Failed data_config validation due to [reason1]" in key for key in error_data)
