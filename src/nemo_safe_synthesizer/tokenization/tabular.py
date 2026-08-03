# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tabular NSS tokenizer implementation."""

from __future__ import annotations

from typing import cast, final

from typing_extensions import override

from ..errors import ParameterError
from .base import NssTokenizer, NssTokenizerCore, PromptRenderer
from .spec import PolicyEpochs
from .types import JsonObject, TabularContext, TokenizerCapabilities, WorkloadKind


@final
class TabularNssTokenizer(NssTokenizer[TabularContext]):
    """Stateless tabular prompt and record policy."""

    IMPLEMENTATION_ID = "nvidia.safe-synthesizer:tabular"
    IMPLEMENTATION_VERSION = "1"
    WORKLOAD_KIND = WorkloadKind.TABULAR
    POLICY_EPOCHS = PolicyEpochs(prompt=1, cache=1)

    @property
    @override
    def capabilities(self) -> TokenizerCapabilities:
        """Declare tabular v1 capabilities."""
        return TokenizerCapabilities(
            record_jsonl=True,
            prompt_encoding=True,
            rolling_prefill=False,
            training_prompt=True,
        )

    @override
    def _prompt_parts(self, context: TabularContext) -> tuple[str, str, str]:
        if not isinstance(context, TabularContext):
            raise ParameterError("TabularNssTokenizer requires TabularContext.")
        schema = ",".join(f'"{column}":<unk>' for column in context.ordered_columns)
        return context.instruction, schema, ""

    @override
    def _training_context(
        self,
        ordered_columns: tuple[str, ...],
        instruction: str,
        current_prefill: str,
    ) -> TabularContext:
        if current_prefill:
            raise ParameterError("Tabular training prompts do not support a prefill.")
        return TabularContext(ordered_columns, instruction)

    @classmethod
    @override
    def _default_workload_payload(cls) -> JsonObject:
        return {"payload_version": 1}

    @classmethod
    @override
    def _validate_workload_payload(cls, payload: object) -> JsonObject:
        if payload != {"payload_version": 1}:
            raise ParameterError("Invalid tabular tokenizer workload payload.")
        return {"payload_version": 1}


def as_tabular_renderer(tokenizer: NssTokenizerCore) -> PromptRenderer[TabularContext]:
    """Narrow a persisted tokenizer to the checked tabular prompt contract."""
    if tokenizer.spec.workload_kind is not WorkloadKind.TABULAR or not isinstance(tokenizer, NssTokenizer):
        raise ParameterError("The selected NSS tokenizer does not support tabular prompt rendering.")
    return cast(PromptRenderer[TabularContext], tokenizer)
