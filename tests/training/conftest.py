# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

# Skip all tests in this module if transformers is not available
transformers = pytest.importorskip(
    "transformers", reason="transformers is required for these tests (install with: uv sync --extra cpu)"
)
AutoTokenizer = transformers.AutoTokenizer
PreTrainedTokenizer = transformers.PreTrainedTokenizer


@pytest.fixture
def fixture_tokenizer(fixture_smollm3_tokenizer) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(fixture_smollm3_tokenizer)
