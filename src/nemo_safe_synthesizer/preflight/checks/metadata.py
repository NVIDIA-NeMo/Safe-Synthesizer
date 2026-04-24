# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata-stage checks that require a loaded tokenizer / model metadata."""

from __future__ import annotations

from ...defaults import PSEUDO_GROUP_COLUMN
from ..base import MetadataCheck
from ..base import IssueCollector
from ..types import MetadataView
from ._helpers import check_group_budget, check_sampled_record_budget, check_schema_prompt_budget

__all__ = ["TokenBudgetCheck"]


class TokenBudgetCheck(MetadataCheck):
    """Verify that records and groups fit within the model's context window.

    This check is a heuristic approximation of what the training assembler
    will see, not an exact simulation. Known sources of drift:

    - Sampling: only the first ``token_sample_size`` rows (default 5000)
      and the largest ``top_groups_to_check`` groups (default 100) are
      tokenized; a long-tail outlier outside the sample can still fail at
      assembly time.
    - Top-by-records bias: groups are ranked by row count, but token
      budget is driven by serialized text length -- a group with fewer rows
      but very wide columns could exceed the budget without being flagged.
    - PII-replacement drift: on ``--validate`` the data has *not* been
      PII-replaced, so token counts reflect the raw input rather than the
      replaced text the assembler actually sees. Replacement tokens can be
      shorter or longer than the originals.

    Treat the output as a strong signal, not a guarantee; a clean result
    means the sampled rows and top groups fit, not that every row will.
    """

    name = "token_budget"
    label = "Token budget"
    # Intentionally no ``requires``: the schema-prompt and sampled-record
    # budget checks don't depend on the group-by column, so a failed
    # ``columns.groupby`` must not skip them. The group-budget branch
    # guards itself with ``group_col in data.columns`` below.
    token_sample_size: int = 5000
    top_groups_to_check: int = 100

    def check(self, ctx: MetadataView, collector: IssueCollector) -> None:
        data = ctx.data
        config = ctx.config
        metadata = ctx.metadata
        if metadata.tokenizer is None:
            collector.warning(
                "tokenizer_unavailable",
                "Tokenizer not available; token budget checks skipped.",
            )
            return

        columns = [column for column in data.columns if column != PSEUDO_GROUP_COLUMN]
        max_new_tokens = check_schema_prompt_budget(collector, columns, metadata)
        if max_new_tokens is None:
            return

        check_sampled_record_budget(collector, data, metadata, max_new_tokens, sample_size_limit=self.token_sample_size)

        # Only run the per-group budget when group-by is configured AND the
        # column is actually present (GroupbyColumnCheck may have flagged
        # a missing column as an error but we still want schema/record
        # checks to run; guarding here keeps them independent).
        group_col = config.data.group_training_examples_by
        if group_col is not None and group_col in data.columns:
            check_group_budget(collector, data, group_col, metadata, max_new_tokens, top_n=self.top_groups_to_check)
