# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared token budget computation used by both the assembler and preflight."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from .record_utils import extract_records_from_jsonl_string, records_to_jsonl

if TYPE_CHECKING:
    from ..llm.metadata import ModelMetadata

NUM_SPECIAL_TOKENS = 2


def compute_schema_prompt_ids(
    columns: list[str],
    metadata: ModelMetadata,
) -> list[int]:
    """Tokenize the full schema prompt using the same path as the assembler.

    Args:
        columns: Column names (excluding internal pseudo-columns).
        metadata: Model metadata with tokenizer, instruction, and prompt config.

    Returns:
        Token IDs for the schema prompt (no special tokens).
    """
    from ..utils import create_schema_prompt

    if metadata.tokenizer is None:
        raise RuntimeError("compute_schema_prompt_ids requires a loaded tokenizer on ModelMetadata")
    schema_prompt = create_schema_prompt(
        columns,
        instruction=metadata.instruction,
        prompt_template=metadata.prompt_config.template,
    )
    return metadata.tokenizer.encode(schema_prompt, add_special_tokens=False)


def compute_max_new_tokens(
    schema_prompt_ids: list[int],
    max_seq_length: int,
) -> int:
    """Max tokens available for record content after schema and special tokens.

    Uses the same formula as assembler._tokenize_records:
    ``max_seq_length - len(schema_prompt_ids) - 2 * NUM_SPECIAL_TOKENS``.
    """
    return max_seq_length - len(schema_prompt_ids) - 2 * NUM_SPECIAL_TOKENS


def tokenize_record(row: pd.Series, tokenizer: Any) -> list[int]:
    """Tokenize a single record using the same JSONL serialization as the assembler.

    Args:
        row: A single DataFrame row.
        tokenizer: HuggingFace tokenizer instance.

    Returns:
        Token IDs for the record (no special tokens).
    """
    jsonl = records_to_jsonl(pd.DataFrame([row.to_dict()]))
    record_text = extract_records_from_jsonl_string(jsonl)[0]
    return tokenizer.encode(record_text, add_special_tokens=False)


def tokenize_records(df: pd.DataFrame, tokenizer: Any) -> list[list[int]]:
    """Tokenize multiple records using shared JSONL serialization.

    Uses batch tokenization when available, and falls back to per-record
    ``encode()`` for tokenizers that only expose single-record APIs.

    Args:
        df: DataFrame whose rows represent records to tokenize.
        tokenizer: HuggingFace tokenizer instance.

    Returns:
        List of token-id lists, one per input row.
    """
    if df.empty:
        return []

    jsonl = records_to_jsonl(df.to_dict(orient="list"))
    record_texts = extract_records_from_jsonl_string(jsonl)

    if callable(tokenizer):
        tokenized = tokenizer(record_texts, add_special_tokens=False)
        if isinstance(tokenized, dict):
            input_ids = tokenized.get("input_ids")
            if (
                isinstance(input_ids, list)
                and len(input_ids) == len(record_texts)
                and all(isinstance(ids, list) for ids in input_ids)
            ):
                return input_ids

    if hasattr(tokenizer, "encode"):
        return [tokenizer.encode(text, add_special_tokens=False) for text in record_texts]

    msg = "Tokenizer must support batch __call__() or encode()."
    raise TypeError(msg)
