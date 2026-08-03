# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateless time-series NSS tokenizer implementation."""

from __future__ import annotations

from ..errors import ParameterError
from .base import NssTokenizer
from .spec import PolicyEpochs
from .types import FrozenJsonObject, JsonObject, TimeSeriesContext, TokenizerCapabilities, WorkloadKind


class TimeSeriesNssTokenizer(NssTokenizer[TimeSeriesContext]):
    """Convert immutable time-series snapshots into canonical prompts and IDs."""

    IMPLEMENTATION_ID = "nvidia.safe-synthesizer:time-series"
    IMPLEMENTATION_VERSION = "1"
    WORKLOAD_KIND = WorkloadKind.TIME_SERIES
    POLICY_EPOCHS = PolicyEpochs(prompt=1, cache=1)

    @property
    def capabilities(self) -> TokenizerCapabilities:
        """Declare time-series v1 capabilities."""
        return TokenizerCapabilities(
            record_jsonl=True,
            prompt_encoding=True,
            rolling_prefill=True,
            training_prompt=True,
        )

    def _prompt_parts(self, context: TimeSeriesContext) -> tuple[str, str, str]:
        if not isinstance(context, TimeSeriesContext):
            raise ParameterError("TimeSeriesNssTokenizer requires TimeSeriesContext.")
        schema = context.schema.to_dict()
        properties = schema.get("properties")
        if not isinstance(properties, dict) or not all(isinstance(column, str) for column in properties):
            raise ParameterError("Time-series schema must contain a string-keyed properties object.")
        fragment = ",".join(f'"{column}":<unk>' for column in context.schema.property_order)
        return context.instruction, fragment, context.current_prefill

    def _training_context(
        self,
        ordered_columns: tuple[str, ...],
        instruction: str,
        current_prefill: str,
    ) -> TimeSeriesContext:
        schema = FrozenJsonObject.from_value({"properties": {column: {} for column in ordered_columns}})
        return TimeSeriesContext(schema, instruction, current_prefill)

    @classmethod
    def _default_workload_payload(cls) -> JsonObject:
        return {"payload_version": 1, "stateful": False}

    @classmethod
    def _validate_workload_payload(cls, payload: object) -> JsonObject:
        if payload != {"payload_version": 1, "stateful": False}:
            raise ParameterError("Invalid time-series tokenizer workload payload.")
        return {"payload_version": 1, "stateful": False}
