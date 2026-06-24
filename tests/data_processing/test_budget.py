# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``data_processing.budget`` token-budget arithmetic."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from transformers import BatchEncoding, PreTrainedTokenizerBase
from typing_extensions import override

from nemo_safe_synthesizer.data_processing.budget import (
    compute_max_new_tokens,
    compute_schema_prompt_ids,
    tokenize_records,
)
from nemo_safe_synthesizer.defaults import PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.llm.metadata import ModelMetadata


class _RecordingTokenizer(PreTrainedTokenizerBase):
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.encoded_text = ""

    @override
    def __call__(self, texts: list[str], *, add_special_tokens: bool) -> dict[str, list[list[int]]]:
        assert add_special_tokens is False
        self.texts = texts
        return {"input_ids": [[ord(char) for char in text] for text in texts]}

    @override
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.encoded_text = text
        return [ord(char) for char in text]


@pytest.mark.unit
def test_compute_max_new_tokens_subtracts_schema_and_special_tokens():
    """Budget = context - schema - 2 * NUM_SPECIAL_TOKENS.

    With NUM_SPECIAL_TOKENS=2: 2048 - 100 - 4 = 1944.
    """
    assert compute_max_new_tokens(list(range(100)), 2048) == 1944


@pytest.mark.unit
def test_compute_max_new_tokens_negative_when_schema_exceeds_context():
    """A schema larger than the context window produces a negative budget."""
    assert compute_max_new_tokens(list(range(2050)), 2048) < 0


@pytest.mark.unit
def test_tokenize_records_excludes_columns():
    """Excluded columns should serialize exactly like a frame without those columns."""
    df = pd.DataFrame({PSEUDO_GROUP_COLUMN: ["group-1"], "value": ["visible"]})
    expected_df = pd.DataFrame({"value": ["visible"]})
    tokenizer = _RecordingTokenizer()
    expected_tokenizer = _RecordingTokenizer()

    token_ids = tokenize_records(df, tokenizer, exclude_columns=(PSEUDO_GROUP_COLUMN,))
    expected_token_ids = tokenize_records(expected_df, expected_tokenizer)

    assert token_ids == expected_token_ids
    assert PSEUDO_GROUP_COLUMN not in tokenizer.texts[0]


class _BatchEncodingTokenizer(PreTrainedTokenizerBase):
    """Stub that mimics a real HF tokenizer: ``__call__`` returns ``BatchEncoding``.

    ``BatchEncoding`` subclasses ``UserDict``, not ``dict``, which is the exact
    case ``tokenize_records`` must accept for the batch fast path. This stub
    also tracks per-record ``encode()`` invocations so the test can assert the
    fast path was actually taken (zero ``encode`` calls).
    """

    def __init__(self) -> None:
        self.encode_calls = 0

    @override
    def __call__(self, texts: list[str], *, add_special_tokens: bool) -> BatchEncoding:
        assert add_special_tokens is False
        return BatchEncoding({"input_ids": [[ord(char) for char in text] for text in texts]})

    @override
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.encode_calls += 1
        return [ord(char) for char in text]


@pytest.mark.unit
def test_tokenize_records_takes_batch_fast_path_for_batchencoding():
    """Real HF tokenizers return ``BatchEncoding`` (a ``UserDict`` subclass).

    Regression: an earlier ``isinstance(tokenized, dict)`` guard skipped the
    batch path for every real tokenizer because ``BatchEncoding`` does not
    subclass ``dict``, silently degrading to per-record ``encode()`` calls.
    """
    df = pd.DataFrame({"value": ["a", "b", "c"]})
    tokenizer = _BatchEncodingTokenizer()

    token_ids = tokenize_records(df, tokenizer)

    assert len(token_ids) == 3
    assert all(isinstance(ids, list) for ids in token_ids)
    assert tokenizer.encode_calls == 0, "batch path was bypassed; fell through to per-record encode()"


@pytest.mark.unit
def test_tokenize_records_no_exclude_default_keeps_all_columns():
    """The shared helper stays policy-free unless callers pass exclude_columns."""
    df = pd.DataFrame({PSEUDO_GROUP_COLUMN: ["group-1"], "value": ["visible"]})
    tokenizer = _RecordingTokenizer()

    tokenize_records(df, tokenizer)

    assert PSEUDO_GROUP_COLUMN in tokenizer.texts[0]


@pytest.mark.unit
def test_compute_schema_prompt_ids_excludes_columns():
    tokenizer = _RecordingTokenizer()
    metadata = cast(
        ModelMetadata,
        SimpleNamespace(
            tokenizer=tokenizer,
            instruction="Generate: ",
            prompt_config=SimpleNamespace(template="{instruction}{schema}{prefill}"),
        ),
    )

    compute_schema_prompt_ids(["value", PSEUDO_GROUP_COLUMN], metadata, exclude_columns=(PSEUDO_GROUP_COLUMN,))

    assert '"value":<unk>' in tokenizer.encoded_text
    assert PSEUDO_GROUP_COLUMN not in tokenizer.encoded_text
