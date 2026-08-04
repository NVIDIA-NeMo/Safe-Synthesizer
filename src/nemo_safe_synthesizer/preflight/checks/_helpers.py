# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers used by more than one core check."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

from ...data_processing.budget import compute_prompt_encoding, tokenize_records
from ...defaults import DEFAULT_EXCLUDE_COLUMNS
from ...errors import GenerationError
from ...tokenization.core import _BoundTokenization
from ..base import IssueCollector

if TYPE_CHECKING:
    from ...config.parameters import SafeSynthesizerParameters
    from ...llm.metadata import ModelMetadata


class _CtxWithConfig(Protocol):
    """Minimal structural protocol satisfied by all stage views and ``PreflightContext``."""

    config: SafeSynthesizerParameters


def resolved_record_count(ctx: _CtxWithConfig) -> int | None:
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
    tokenization: _BoundTokenization,
) -> int | None:
    """Validate schema prompt against context length; return token budget.

    Delegates to ``data_processing.budget`` for parity with the assembler.
    """
    prompt = compute_prompt_encoding(
        columns,
        metadata,
        tokenization,
        exclude_columns=DEFAULT_EXCLUDE_COLUMNS,
    )
    try:
        capacity = tokenization.capacity_for(
            prompt,
            context_limit=metadata.max_seq_length,
            sequence_count=1,
        )
    except GenerationError:
        collector.error(
            "schema_exceeds_context",
            (
                f"Schema prompt ({len(prompt.input_ids)} tokens) "
                f"exceeds model context window ({metadata.max_seq_length})."
            ),
        )
        return None
    return capacity.record_token_capacity


def check_sampled_record_budget(
    collector: IssueCollector,
    data: pd.DataFrame,
    metadata: ModelMetadata,
    tokenization: _BoundTokenization,
    max_new_tokens: int,
    *,
    sample_size_limit: int,
) -> None:
    """Validate sampled records against token budget."""
    sample_size = min(len(data), sample_size_limit)
    sample = data.sample(n=sample_size, random_state=42) if sample_size < len(data) else data
    tokenized_records = tokenize_records(sample, tokenization, exclude_columns=DEFAULT_EXCLUDE_COLUMNS)
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
    tokenization: _BoundTokenization,
    max_new_tokens: int,
    *,
    top_n: int,
) -> None:
    """Validate largest groups against token budget."""
    # Stop at the first flagged group: the message only says "N of the
    # largest groups exceed" as a diagnostic, so we don't need an exact
    # count, and tokenizing every group on a large dataset is expensive.
    # Within a group, tokenize in chunks and bail as soon as the running
    # total exceeds the budget so a single enormous group doesn't force
    # a full pass through every row.
    chunk_size = 1024
    top_positions = _top_group_positions(data, group_col, top_n=top_n)
    n_top = len(top_positions)

    def _check_group(positions: np.ndarray) -> bool:
        running_tokens = 0
        for start in range(0, positions.size, chunk_size):
            chunk = data.iloc[positions[start : start + chunk_size]]
            chunk_token_ids = tokenize_records(chunk, tokenization, exclude_columns=DEFAULT_EXCLUDE_COLUMNS)
            running_tokens += sum(len(token_ids) for token_ids in chunk_token_ids)
            if running_tokens > max_new_tokens:
                return True
        return False

    groups_exceeded = 0
    for positions in top_positions:
        if _check_group(positions):
            groups_exceeded += 1
            break

    if groups_exceeded:
        collector.error(
            "group_exceeds_context",
            f"At least one of the {n_top} largest groups exceeds the token budget ({max_new_tokens} tokens).",
        )


def _top_group_positions(data: pd.DataFrame, group_col: str, *, top_n: int) -> list[np.ndarray]:
    """Return integer positions for the ``top_n`` largest groups, sorted by size descending.

    Dispatches on ``group_col`` dtype: string equality scans Python objects
    per element, so the one-shot ``groupby.indices`` hash pays for itself;
    numeric equality is C-fast, so a per-key ``np.flatnonzero`` is cheaper
    than the upfront groupby at small ``top_n``.
    """
    # 1M rows / 10k zipf-distributed groups: ~6x faster than a per-group
    # ``data[mask]`` scan on string keys; within noise on numeric keys.
    # See PR #406 discussion for the benchmark table.
    if not pd.api.types.is_numeric_dtype(data[group_col]):
        raw = data.groupby(group_col, sort=False, observed=True).indices
        indices = {k: np.asarray(v) for k, v in raw.items()}
        top_keys = sorted(indices, key=lambda k: -indices[k].size)[:top_n]
        return [indices[k] for k in top_keys]

    sizes = data.groupby(group_col, sort=False).size().sort_values(ascending=False)
    top_keys = sizes.head(top_n).index
    group_arr = data[group_col].to_numpy()
    return [np.flatnonzero(group_arr == key) for key in top_keys]
