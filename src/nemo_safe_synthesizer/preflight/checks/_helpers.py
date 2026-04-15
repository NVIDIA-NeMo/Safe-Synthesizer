# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers used by more than one core check."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ...data_processing.budget import compute_max_new_tokens, compute_schema_prompt_ids, tokenize_records
from ..types import IssueCollector, PreflightContext

if TYPE_CHECKING:
    from ...llm.metadata import ModelMetadata


def resolved_record_count(ctx: PreflightContext) -> int | None:
    """Return ``num_input_records_to_sample`` once resolved to a concrete int.

    ``num_input_records_to_sample`` may still carry a sentinel like ``"auto"``
    if ``AutoConfigResolver`` has not normalized it. Checks that only make
    sense against a concrete count should short-circuit on ``None``.
    """
    n_records = ctx.config.training.num_input_records_to_sample
    return n_records if isinstance(n_records, int) else None


def check_schema_prompt_budget(
    collector: IssueCollector,
    columns: list[str],
    metadata: ModelMetadata,
) -> int | None:
    """Validate schema prompt against context length; return token budget.

    Delegates to ``data_processing.budget`` for parity with the assembler.
    """
    schema_prompt_ids = compute_schema_prompt_ids(columns, metadata)
    max_new_tokens = compute_max_new_tokens(schema_prompt_ids, metadata.max_seq_length)
    if max_new_tokens <= 0:
        collector.error(
            "schema_exceeds_context",
            (
                f"Schema prompt ({len(schema_prompt_ids)} tokens) "
                f"exceeds model context window ({metadata.max_seq_length})."
            ),
        )
        return None
    return max_new_tokens


def check_sampled_record_budget(
    collector: IssueCollector,
    data: pd.DataFrame,
    metadata: ModelMetadata,
    max_new_tokens: int,
    *,
    sample_size_limit: int,
) -> None:
    """Validate sampled records against token budget."""
    if metadata.tokenizer is None:
        raise RuntimeError("check_sampled_record_budget requires a loaded tokenizer on ModelMetadata")

    sample_size = min(len(data), sample_size_limit)
    sample = data.sample(n=sample_size, random_state=42) if sample_size < len(data) else data
    tokenized_records = tokenize_records(sample, metadata.tokenizer)
    exceeded = sum(1 for token_ids in tokenized_records if len(token_ids) > max_new_tokens)
    if exceeded:
        collector.error(
            "record_exceeds_context",
            (f"{exceeded} of {sample_size} sampled records exceed the token budget ({max_new_tokens} tokens)."),
        )


def check_group_budget(
    collector: IssueCollector,
    data: pd.DataFrame,
    group_col: str,
    metadata: ModelMetadata,
    max_new_tokens: int,
    *,
    top_n: int,
) -> None:
    """Validate largest groups against token budget."""
    if metadata.tokenizer is None:
        raise RuntimeError("check_group_budget requires a loaded tokenizer on ModelMetadata")

    group_sizes = data.groupby(group_col).size().sort_values(ascending=False)
    top_groups = group_sizes.head(top_n).index
    groups_exceeded = 0
    # Stop at the first flagged group: the message only says "N of the
    # largest groups exceed" as a diagnostic -- we don't need an exact
    # count, and tokenizing every group on a large dataset is expensive.
    # Within a group, tokenize in chunks and bail as soon as the running
    # total exceeds the budget so a single enormous group doesn't force
    # a full pass through every row.
    chunk_size = 1024
    for grp_key in top_groups:
        grp_df = data[data[group_col] == grp_key]
        running_tokens = 0
        exceeded = False
        for start in range(0, len(grp_df), chunk_size):
            chunk = grp_df.iloc[start : start + chunk_size]
            chunk_token_ids = tokenize_records(chunk, metadata.tokenizer)
            running_tokens += sum(len(token_ids) for token_ids in chunk_token_ids)
            if running_tokens > max_new_tokens:
                exceeded = True
                break
        if exceeded:
            groups_exceeded += 1
            break
    if groups_exceeded:
        collector.error(
            "group_exceeds_context",
            (
                f"At least one of the {len(top_groups)} largest groups "
                f"exceeds the token budget ({max_new_tokens} tokens)."
            ),
        )
