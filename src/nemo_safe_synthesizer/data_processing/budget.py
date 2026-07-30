# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared token budget computation used by both the assembler and preflight."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import pandas as pd

from ..errors import ParameterError
from ..tokenization import NssTokenizer
from ..tokenization.records import columnar_batch_to_records
from ..tokenization.types import JsonValue

if TYPE_CHECKING:
    from ..llm.metadata import ModelMetadata

NUM_SPECIAL_TOKENS = 2


def compute_schema_prompt_ids(
    columns: list[str],
    metadata: ModelMetadata,
    *,
    exclude_columns: Sequence[str] = (),
) -> list[int]:
    """Tokenize the full schema prompt using the same path as the assembler.

    Args:
        columns: Column names.
        metadata: Model metadata with tokenizer, instruction, and prompt config.
        exclude_columns: Column names to omit from the schema prompt.

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
        exclude_columns=list(exclude_columns),
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


def tokenize_record(row: pd.Series, tokenizer: NssTokenizer) -> list[int]:
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
    record = {cast(str, column): cast(JsonValue, value) for column, value in raw_record.items()}
    return list(tokenizer.encode_records([record]).input_ids[0])


def tokenize_records(
    df: pd.DataFrame,
    tokenizer: NssTokenizer,
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
    return [list(input_ids) for input_ids in tokenizer.encode_records(rows, exclude_columns=exclude_columns).input_ids]
