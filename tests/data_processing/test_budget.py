# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``data_processing.budget`` token-budget arithmetic."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pandas as pd
import pytest
from datasets import Dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase

from nemo_safe_synthesizer.data_processing import budget as budget_module
from nemo_safe_synthesizer.data_processing.assembler import TabularDataExampleAssembler
from nemo_safe_synthesizer.data_processing.budget import (
    compute_max_new_tokens,
    compute_schema_prompt_ids,
    tokenize_records,
)
from nemo_safe_synthesizer.defaults import DEFAULT_EXCLUDE_COLUMNS, PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.llm.metadata import ModelMetadata
from nemo_safe_synthesizer.preflight.checks._helpers import check_sampled_record_budget
from nemo_safe_synthesizer.tokenization import NssTokenizer, PromptEncoding, WorkloadKind, create_runtime_nss_tokenizer


class _RecordingTokenizer(PreTrainedTokenizerBase):
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.encoded_text = ""

    def __call__(self, texts: list[str], *, add_special_tokens: bool) -> dict[str, list[list[int]]]:
        assert add_special_tokens is False
        self.texts = texts
        return {"input_ids": [[ord(char) for char in text] for text in texts]}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.encoded_text = text
        return [ord(char) for char in text]


class _NssRecordAdapter:
    """Minimal NSS-shaped adapter that keeps budget tests focused on delegation."""

    def __init__(self, native: PreTrainedTokenizerBase) -> None:
        self.native = native

    def encode_records(self, records, *, exclude_columns=()):
        excluded = frozenset(exclude_columns)
        texts = [
            json.dumps(
                {key: value for key, value in record.items() if key not in excluded},
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for record in records
        ]
        result = self.native(texts, add_special_tokens=False)
        return SimpleNamespace(input_ids=tuple(tuple(row) for row in result["input_ids"]))


@pytest.mark.unit
def test_compute_max_new_tokens_subtracts_schema_and_special_tokens():
    tokenizer = MagicMock()
    tokenizer.capacity_for.return_value = SimpleNamespace(record_token_capacity=1944)
    prompt = PromptEncoding("prompt", tuple(range(102)), (1,) * 102)

    assert compute_max_new_tokens(prompt, 2048, cast(NssTokenizer, tokenizer)) == 1944
    tokenizer.capacity_for.assert_called_once_with(prompt, context_limit=2048, sequence_count=1)


@pytest.mark.unit
def test_compute_max_new_tokens_negative_when_schema_exceeds_context():
    tokenizer = MagicMock()
    tokenizer.capacity_for.return_value = SimpleNamespace(record_token_capacity=-4)

    prompt = PromptEncoding("prompt", (), ())

    assert compute_max_new_tokens(prompt, 2048, cast(NssTokenizer, tokenizer)) < 0


@pytest.mark.unit
def test_tokenize_records_excludes_columns():
    """Excluded columns should serialize exactly like a frame without those columns."""
    df = pd.DataFrame({PSEUDO_GROUP_COLUMN: ["group-1"], "value": ["visible"]})
    expected_df = pd.DataFrame({"value": ["visible"]})
    tokenizer = _RecordingTokenizer()
    expected_tokenizer = _RecordingTokenizer()

    token_ids = tokenize_records(
        df,
        cast(NssTokenizer, _NssRecordAdapter(tokenizer)),
        exclude_columns=(PSEUDO_GROUP_COLUMN,),
    )
    expected_token_ids = tokenize_records(
        expected_df,
        cast(NssTokenizer, _NssRecordAdapter(expected_tokenizer)),
    )

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

    def __call__(self, texts: list[str], *, add_special_tokens: bool) -> BatchEncoding:
        assert add_special_tokens is False
        return BatchEncoding({"input_ids": [[ord(char) for char in text] for text in texts]})

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

    token_ids = tokenize_records(df, cast(NssTokenizer, _NssRecordAdapter(tokenizer)))

    assert len(token_ids) == 3
    assert all(isinstance(ids, list) for ids in token_ids)
    assert tokenizer.encode_calls == 0, "batch path was bypassed; fell through to per-record encode()"


@pytest.mark.unit
def test_tokenize_records_no_exclude_default_keeps_all_columns():
    """The shared helper stays policy-free unless callers pass exclude_columns."""
    df = pd.DataFrame({PSEUDO_GROUP_COLUMN: ["group-1"], "value": ["visible"]})
    tokenizer = _RecordingTokenizer()

    tokenize_records(df, cast(NssTokenizer, _NssRecordAdapter(tokenizer)))

    assert PSEUDO_GROUP_COLUMN in tokenizer.texts[0]


def test_assembler_budget_and_preflight_share_exact_ordered_record_encoding(
    fixture_tokenizer,
    fixture_smollm3_tokenizer,
    tmp_path,
    monkeypatch,
) -> None:
    dataset = Dataset.from_dict(
        {
            "b": ["first", "second", "third"],
            PSEUDO_GROUP_COLUMN: ["hidden-1", "hidden-2", "hidden-3"],
            "a": [1, 2, 3],
        }
    )
    frame = cast(pd.DataFrame, dataset.to_pandas())
    metadata = ModelMetadata.from_str_or_path(
        model_name_or_path=fixture_smollm3_tokenizer,
        tokenizer=fixture_tokenizer,
    )
    record_tokenizer = create_runtime_nss_tokenizer(
        fixture_tokenizer,
        metadata,
        workload_kind=WorkloadKind.TABULAR,
    )
    captured = []
    original_encode_records = NssTokenizer.encode_records

    def capture_records(self, records, *, exclude_columns=()):
        batch = original_encode_records(self, records, exclude_columns=exclude_columns)
        captured.append(batch)
        return batch

    monkeypatch.setattr(NssTokenizer, "encode_records", capture_records)
    assembler = TabularDataExampleAssembler(
        dataset=dataset,
        tokenizer=fixture_tokenizer,
        record_tokenizer=record_tokenizer,
        metadata=metadata,
        cache_file_path=tmp_path,
        seed=1,
    )
    assembler_batch = captured[-1]
    captured.clear()

    budget_ids = tokenize_records(
        frame,
        record_tokenizer,
        exclude_columns=DEFAULT_EXCLUDE_COLUMNS,
    )
    budget_batch = captured[-1]
    captured.clear()

    collector = MagicMock()
    check_sampled_record_budget(
        collector,
        frame,
        metadata,
        record_tokenizer,
        max_new_tokens=10_000,
        sample_size_limit=len(frame),
    )
    preflight_batch = captured[-1]

    expected_text = ('{"b":"first","a":1}\n', '{"b":"second","a":2}\n', '{"b":"third","a":3}\n')
    expected_ids = fixture_tokenizer(list(expected_text), add_special_tokens=False)["input_ids"]
    expected_masks = [[1] * len(row) for row in expected_ids]
    arrow = assembler.tokenized_records.to_dict()

    assert arrow == {
        "text": list(expected_text),
        "input_ids": expected_ids,
        "attention_mask": expected_masks,
    }
    assert list(assembler_batch.input_ids) == [tuple(row) for row in expected_ids]
    assert budget_ids == expected_ids
    assert budget_batch.input_ids == preflight_batch.input_ids == tuple(tuple(row) for row in expected_ids)
    assert budget_batch.attention_mask == preflight_batch.attention_mask
    assert assembler.stats["tokens_per_record"].mean == sum(map(len, expected_ids)) / len(expected_ids)
    collector.error.assert_not_called()


@pytest.mark.unit
def test_compute_schema_prompt_ids_excludes_columns():
    tokenizer = MagicMock()
    tokenizer.render_training_prompt.return_value = SimpleNamespace(text='Generate: "value":<unk>')
    tokenizer.encode_no_special.return_value = (1, 2, 3)
    metadata = cast(
        ModelMetadata,
        SimpleNamespace(
            instruction="Generate: ",
            prompt_config=SimpleNamespace(template="{instruction}{schema}{prefill}"),
        ),
    )

    result = compute_schema_prompt_ids(
        ["value", PSEUDO_GROUP_COLUMN],
        metadata,
        tokenizer,
        exclude_columns=(PSEUDO_GROUP_COLUMN,),
    )

    assert result == [1, 2, 3]
    tokenizer.render_training_prompt.assert_called_once_with(("value",), "Generate: ")
    tokenizer.encode_no_special.assert_called_once_with('Generate: "value":<unk>')


def test_prompt_budget_uses_authoritative_nss_prompt(fixture_tokenizer, fixture_smollm3_tokenizer) -> None:
    metadata = ModelMetadata.from_str_or_path(
        model_name_or_path=fixture_smollm3_tokenizer,
        tokenizer=fixture_tokenizer,
    )
    nss_tokenizer = create_runtime_nss_tokenizer(
        fixture_tokenizer,
        metadata,
        workload_kind=WorkloadKind.TABULAR,
    )

    prompt = budget_module.compute_prompt_encoding(
        ["b", PSEUDO_GROUP_COLUMN, "a"],
        metadata,
        nss_tokenizer,
        exclude_columns=DEFAULT_EXCLUDE_COLUMNS,
    )

    assert prompt == nss_tokenizer.render_training_prompt(("b", "a"), metadata.instruction)


def test_legacy_budget_helpers_delegate_to_nss_capacity(fixture_tokenizer, fixture_smollm3_tokenizer) -> None:
    metadata = ModelMetadata.from_str_or_path(
        model_name_or_path=fixture_smollm3_tokenizer,
        tokenizer=fixture_tokenizer,
    )
    nss_tokenizer = create_runtime_nss_tokenizer(
        fixture_tokenizer,
        metadata,
        workload_kind=WorkloadKind.TABULAR,
    )
    prompt = nss_tokenizer.render_training_prompt(("value",), metadata.instruction)

    prompt_ids = compute_schema_prompt_ids(["value"], metadata, nss_tokenizer)
    budget = compute_max_new_tokens(prompt, metadata.max_seq_length, nss_tokenizer)

    assert prompt_ids == list(nss_tokenizer.encode_no_special(prompt.text))
    assert (
        budget
        == nss_tokenizer.capacity_for(
            prompt,
            context_limit=metadata.max_seq_length,
            sequence_count=1,
        ).record_token_capacity
    )
