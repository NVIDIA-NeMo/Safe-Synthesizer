# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO

import pandas as pd
import pytest

# Skip all tests in this module if transformers is not available
transformers = pytest.importorskip(
    "transformers", reason="transformers is required for these tests (install with: uv sync --extra cpu)"
)
AutoTokenizer = transformers.AutoTokenizer
PreTrainedTokenizer = transformers.PreTrainedTokenizer


from nemo_safe_synthesizer.data_processing.dataset import make_json_schema  # noqa: E402
from nemo_safe_synthesizer.data_processing.record_utils import records_to_jsonl  # noqa: E402


@pytest.fixture
def fixture_tokenizer(fixture_stub_tokenizer_path) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(fixture_stub_tokenizer_path)


@pytest.fixture
def fixture_lmsys_dataset_jsonl_and_schema(
    fixture_lmsys_chat_non_english_dataset,
) -> tuple[str, dict]:
    return records_to_jsonl(fixture_lmsys_chat_non_english_dataset), make_json_schema(
        fixture_lmsys_chat_non_english_dataset
    )


# Purpose: Returns first 5 Iris rows as JSONL and its inferred JSON schema for processor/regex tests.
# Used by:
#   - tests/generation/test_processors.py::test_tabular_data_processor
#   - tests/generation/test_processors.py::test_grouped_data_processor_with_no_groups
#   - tests/generation/test_processors.py::test_grouped_data_processor_with_invalid_json
#   - tests/generation/test_processors.py::test_grouped_data_processor_with_non_unique_group_by
#   - tests/generation/test_processors.py::test_grouped_data_processor_out_of_order_records
#   - tests/generation/test_processors.py::test_grouped_data_processor_multiple_group_by
#   - tests/generation/test_processors.py::test_grouped_data_processor_multiple_group_by_error
#   - tests/generation/test_regex_manager.py::test_build_json_based_regex
@pytest.fixture
def fixture_valid_iris_dataset_jsonl_and_schema(
    fixture_iris_dataset,
) -> tuple[str, dict]:
    sample_df = pd.DataFrame(fixture_iris_dataset[:5])
    str_buffer = StringIO()
    sample_df.to_json(str_buffer, orient="records", lines=True)
    return str_buffer.getvalue(), make_json_schema(sample_df)
