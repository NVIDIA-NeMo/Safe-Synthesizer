# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared assembler/preflight token-budget behavior."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_safe_synthesizer.data_processing.budget import (
    compute_max_new_tokens,
    compute_prompt_encoding,
    tokenize_record,
    tokenize_records,
)
from nemo_safe_synthesizer.defaults import PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.errors import GenerationError
from nemo_safe_synthesizer.llm.metadata import ModelMetadata
from nemo_safe_synthesizer.tokenization import WorkloadKind, bind_tokenizer


def _values(tokenizers_dir: Path):
    source = tokenizers_dir / "smollm3b"
    native = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(source))
    metadata = ModelMetadata.from_str_or_path(source, tokenizer=native)
    tokenization = bind_tokenizer(native, metadata, workload_kind=WorkloadKind.TABULAR)
    return native, metadata, tokenization


def test_prompt_budget_preserves_order_and_excludes_internal_columns(tokenizers_dir: Path) -> None:
    _, metadata, tokenization = _values(tokenizers_dir)

    prompt = compute_prompt_encoding(
        ["b", PSEUDO_GROUP_COLUMN, "a"],
        metadata,
        tokenization,
        exclude_columns=[PSEUDO_GROUP_COLUMN],
    )

    assert '"b":<unk>,"a":<unk>' in prompt.text
    assert PSEUDO_GROUP_COLUMN not in prompt.text


def test_max_new_tokens_is_exact_one_sequence_capacity(tokenizers_dir: Path) -> None:
    _, metadata, tokenization = _values(tokenizers_dir)
    prompt = compute_prompt_encoding(["value"], metadata, tokenization)

    result = compute_max_new_tokens(prompt, metadata.max_seq_length, tokenization)

    assert result == metadata.max_seq_length - len(prompt.input_ids) - 2


def test_negative_record_capacity_is_reported_without_clamping(tokenizers_dir: Path) -> None:
    _, metadata, tokenization = _values(tokenizers_dir)
    prompt = compute_prompt_encoding(["value"], metadata, tokenization)

    result = compute_max_new_tokens(prompt, len(prompt.input_ids) + 1, tokenization)

    assert result == -1


def test_single_and_batch_record_paths_preserve_pandas_row_dtypes(tokenizers_dir: Path) -> None:
    _, _, tokenization = _values(tokenizers_dir)
    frame = pd.DataFrame([{"b": 2, "a": 0.12345678901234567}])

    single = tokenize_record(frame.iloc[0], tokenization)
    batch = tokenize_records(frame, tokenization)

    single_text = '{"b":2.0,"a":0.123456789}\n'
    batch_text = '{"b":2,"a":0.123456789}\n'
    assert single == list(tokenization.native.encode(single_text, add_special_tokens=False))
    assert batch == [list(tokenization.native.encode(batch_text, add_special_tokens=False))]


def test_batch_record_path_preserves_unicode_and_order(tokenizers_dir: Path) -> None:
    _, _, tokenization = _values(tokenizers_dir)
    frame = pd.DataFrame({"second": ["λ", "雪"], "first": [1, 2]})

    result = tokenize_records(frame, tokenization)

    expected = [
        tokenization.native.encode('{"second":"λ","first":1}\n', add_special_tokens=False),
        tokenization.native.encode('{"second":"雪","first":2}\n', add_special_tokens=False),
    ]
    assert result == expected


def test_batch_record_path_excludes_internal_columns(tokenizers_dir: Path) -> None:
    _, _, tokenization = _values(tokenizers_dir)
    frame = pd.DataFrame({PSEUDO_GROUP_COLUMN: ["g"], "value": [1]})

    result = tokenize_records(frame, tokenization, exclude_columns=[PSEUDO_GROUP_COLUMN])

    assert result == [tokenization.native.encode('{"value":1}\n', add_special_tokens=False)]


def test_empty_dataframe_produces_no_record_ids(tokenizers_dir: Path) -> None:
    _, _, tokenization = _values(tokenizers_dir)

    assert tokenize_records(pd.DataFrame({"value": []}), tokenization) == []


def test_prompt_overflow_uses_shared_user_error(tokenizers_dir: Path) -> None:
    _, metadata, tokenization = _values(tokenizers_dir)
    prompt = compute_prompt_encoding(["long_column_name"], metadata, tokenization)

    with pytest.raises(GenerationError, match="dataset schema requires more tokens"):
        tokenization.validate_prompt_capacity(
            prompt,
            context_limit=len(prompt.input_ids) - 1,
            rope_scaling_factor=1,
        )


def test_time_series_budget_includes_initial_prefill(tokenizers_dir: Path) -> None:
    source = tokenizers_dir / "smollm3b"
    native = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(source))
    metadata = ModelMetadata.from_str_or_path(source, tokenizer=native)
    tokenization = bind_tokenizer(native, metadata, workload_kind=WorkloadKind.TIME_SERIES)
    base = tokenization.render_prompt(["t", "v"], metadata.instruction)
    prefilled = tokenization.render_prompt(
        ["t", "v"],
        metadata.instruction,
        current_prefill=' {"t":1,"v":2}\n',
    )

    assert len(prefilled.input_ids) > len(base.input_ids)
    assert compute_max_new_tokens(prefilled, metadata.max_seq_length, tokenization) < compute_max_new_tokens(
        base,
        metadata.max_seq_length,
        tokenization,
    )
