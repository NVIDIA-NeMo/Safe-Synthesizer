# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared token budget computation used by both the assembler and preflight."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import pandas as pd

from ..errors import ParameterError
from ..tokenization import PromptEncoding
from ..tokenization.core import _BoundTokenization
from ..tokenization.records import columnar_batch_to_records

if TYPE_CHECKING:
    from ..llm.metadata import ModelMetadata


def compute_prompt_encoding(
    columns: Sequence[str],
    metadata: ModelMetadata,
    tokenization: _BoundTokenization,
    *,
    exclude_columns: Sequence[str] = (),
) -> PromptEncoding:
    """Render the exact NSS training prompt for budget and preflight use."""
    excluded = frozenset(exclude_columns)
    ordered_columns = tuple(column for column in columns if column not in excluded)
    return tokenization.render_prompt(ordered_columns, metadata.instruction)


def compute_max_new_tokens(
    prompt: PromptEncoding,
    max_seq_length: int,
    tokenization: _BoundTokenization,
) -> int:
    """Return exact one-sequence record capacity from the NSS contract."""
    return tokenization.capacity_for(
        prompt,
        context_limit=max_seq_length,
        sequence_count=1,
    ).record_token_capacity


def tokenize_record(row: pd.Series, tokenization: _BoundTokenization) -> list[int]:
    """Tokenize a single record using the same JSONL serialization as the assembler.

    Args:
        row: A single DataFrame row.
        tokenizer: NSS tokenizer owning ordered record encoding.

    Returns:
        Token IDs for the record (no special tokens).
    """
    raw_record = row.to_dict()
    if not all(isinstance(column, str) for column in raw_record):
        raise ParameterError("Record column names must be strings.")
    record = {cast(str, column): value for column, value in raw_record.items()}
    return list(tokenization.encode_records([record]).input_ids[0])


def tokenize_records(
    df: pd.DataFrame,
    tokenization: _BoundTokenization,
    *,
    exclude_columns: Sequence[str] = (),
) -> list[list[int]]:
    """Tokenize multiple records using shared JSONL serialization.

    Delegates serialization and one-call native batching to the NSS contract.

    Args:
        df: DataFrame whose rows represent records to tokenize.
        tokenizer: NSS tokenizer owning ordered record encoding.
        exclude_columns: Column names to omit from serialized records.

    Returns:
        List of token-id lists, one per input row.
    """
    raw_columns = df.to_dict(orient="list")
    if not all(isinstance(column, str) for column in raw_columns):
        raise ParameterError("Record column names must be strings.")
    columns = {cast(str, column): cast(Sequence[object], values) for column, values in raw_columns.items()}
    rows = columnar_batch_to_records(columns, row_count=len(df))
    return [
        list(input_ids) for input_ids in tokenization.encode_records(rows, exclude_columns=exclude_columns).input_ids
    ]
