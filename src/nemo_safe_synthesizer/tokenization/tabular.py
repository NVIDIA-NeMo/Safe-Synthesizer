# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tabular NSS tokenizer implementation."""

from __future__ import annotations

import json

from ..errors import ParameterError
from .base import NssTokenizer
from .types import JsonObject, TabularContext, TokenizerCapabilities, WorkloadKind


class TabularNssTokenizer(NssTokenizer[TabularContext]):
    """Stateless tabular prompt and record policy."""

    IMPLEMENTATION_ID = "nvidia.safe-synthesizer:tabular"
    IMPLEMENTATION_VERSION = "1"
    WORKLOAD_KIND = WorkloadKind.TABULAR

    @property
    def capabilities(self) -> TokenizerCapabilities:
        """Declare tabular v1 capabilities."""
        return TokenizerCapabilities(record_jsonl=True, prompt_encoding=True, rolling_prefill=False)

    def _prompt_parts(self, context: TabularContext) -> tuple[str, str, str]:
        if not isinstance(context, TabularContext):
            raise ParameterError("TabularNssTokenizer requires TabularContext.")
        schema = ",".join(f"{json.dumps(column, ensure_ascii=False)}:<unk>" for column in context.ordered_columns)
        return context.instruction, schema, ""

    @classmethod
    def _default_workload_payload(cls) -> JsonObject:
        return {"payload_version": 1}

    @classmethod
    def _validate_workload_payload(cls, payload: object) -> JsonObject:
        if payload != {"payload_version": 1}:
            raise ParameterError("Invalid tabular tokenizer workload payload.")
        return {"payload_version": 1}
