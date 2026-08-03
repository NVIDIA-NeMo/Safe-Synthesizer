# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable value objects for NSS tokenizer contracts."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from string import Formatter
from typing import TypeAlias, cast

from ..errors import ParameterError

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Serialize a JSON value using the NSS canonical JSON algorithm."""
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ParameterError("Canonical JSON values must be finite JSON data.") from exc


def _validated_json_object(value: object, *, name: str) -> JsonObject:
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ParameterError(f"{name} must be a finite JSON object.") from exc
    if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
        raise ParameterError(f"{name} must be a JSON object with string keys.")
    return cast(JsonObject, decoded)


@dataclass(frozen=True, slots=True)
class FrozenJsonObject:
    """Deeply immutable JSON object with schema property order retained."""

    canonical: str
    property_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            value = json.loads(self.canonical)
        except json.JSONDecodeError as exc:
            raise ParameterError("Frozen JSON must contain a canonical JSON object.") from exc
        validated = _validated_json_object(value, name="schema")
        if canonical_json_bytes(validated).decode() != self.canonical:
            raise ParameterError("Frozen JSON must use canonical JSON.")
        properties = validated.get("properties")
        expected = set(properties) if isinstance(properties, dict) else set()
        if len(set(self.property_order)) != len(self.property_order) or set(self.property_order) != expected:
            raise ParameterError("Frozen JSON property order must exactly match schema properties.")

    @classmethod
    def from_value(cls, value: object) -> FrozenJsonObject:
        """Validate and freeze a JSON object."""
        validated = _validated_json_object(value, name="schema")
        properties = validated.get("properties")
        property_order = tuple(properties) if isinstance(properties, dict) else ()
        return cls(canonical_json_bytes(validated).decode(), property_order)

    def to_dict(self) -> JsonObject:
        """Return an independent mutable representation."""
        return cast(JsonObject, json.loads(self.canonical))


class WorkloadKind(StrEnum):
    """Tokenizer workload families supported by the v1 API."""

    TABULAR = "tabular"
    TIME_SERIES = "time-series"


class WorkloadContext:
    """Marker contract for immutable workload-specific input snapshots."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class TabularContext(WorkloadContext):
    """Immutable tabular prompt inputs."""

    ordered_columns: tuple[str, ...]
    instruction: str

    def __post_init__(self) -> None:
        if not isinstance(self.ordered_columns, tuple) or not all(
            isinstance(column, str) for column in self.ordered_columns
        ):
            raise ParameterError("Tabular ordered_columns must be an immutable tuple of strings.")
        if not isinstance(self.instruction, str):
            raise ParameterError("Tabular instruction must be a string.")
        if len(set(self.ordered_columns)) != len(self.ordered_columns):
            raise ParameterError("Tabular columns must be unique and ordered.")


@dataclass(frozen=True, slots=True)
class TimeSeriesContext(WorkloadContext):
    """Immutable time-series prompt inputs copied from backend state.

    ``current_prefill`` is opaque caller-owned text that the tokenizer inserts
    verbatim. Non-empty initial and rolling prefills share one shape: a leading
    space followed by newline-terminated records. Rolling producers reuse the
    model's emitted record text, or canonical serializer output when emitted
    text is unavailable; they never reserialize parsed values ad hoc. This
    value has no record-count or length bound here because consumers compute
    capacity from the complete rendered prompt.
    """

    schema: FrozenJsonObject
    instruction: str
    current_prefill: str

    def __post_init__(self) -> None:
        if not isinstance(self.schema, FrozenJsonObject):
            raise ParameterError("Time-series schema must be a FrozenJsonObject.")
        if not isinstance(self.instruction, str) or not isinstance(self.current_prefill, str):
            raise ParameterError("Time-series instruction and current_prefill must be strings.")


@dataclass(frozen=True, slots=True)
class TokenizerCapabilities:
    """Capabilities consumers may request without inspecting concrete classes."""

    record_jsonl: bool
    prompt_encoding: bool
    rolling_prefill: bool
    no_special_encoding: bool = True
    training_prompt: bool = False


@dataclass(frozen=True, slots=True)
class FramingPolicy:
    """Immutable NSS-owned prompt and record framing policy."""

    prompt_template: str
    add_bos_token_to_prompt: bool
    add_eos_token_to_prompt: bool
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    bos_token: str
    eos_token: str
    pad_token: str

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_template, str):
            raise ParameterError("Prompt template must be a string.")
        if not isinstance(self.add_bos_token_to_prompt, bool) or not isinstance(self.add_eos_token_to_prompt, bool):
            raise ParameterError("Prompt special-token flags must be booleans.")
        if self.pad_token_id is None:
            raise ParameterError("A pad token ID is required; zero is a valid pad token ID.")
        if self.eos_token_id is None:
            raise ParameterError("An EOS token ID is required.")
        if self.bos_token_id is None:
            raise ParameterError("A BOS token ID is required for sequence framing.")
        if not all(isinstance(token, str) and token for token in (self.bos_token, self.eos_token, self.pad_token)):
            raise ParameterError("BOS, EOS, and pad token strings are required.")
        for name, token_id in (
            ("bos_token_id", self.bos_token_id),
            ("eos_token_id", self.eos_token_id),
            ("pad_token_id", self.pad_token_id),
        ):
            if token_id is not None and (not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0):
                raise ParameterError(f"{name} must be a non-negative integer.")
        for placeholder in ("{instruction}", "{schema}", "{prefill}"):
            if placeholder not in self.prompt_template:
                raise ParameterError(f"Prompt template must contain active field {placeholder}.")
        try:
            fields = [
                field_name
                for _, field_name, format_spec, conversion in Formatter().parse(self.prompt_template)
                if field_name is not None
                and not format_spec
                and conversion is None
                and field_name in {"instruction", "schema", "prefill"}
            ]
            all_fields = [field_name for _, field_name, _, _ in Formatter().parse(self.prompt_template) if field_name]
        except ValueError as exc:
            raise ParameterError("Prompt template is malformed.") from exc
        expected = Counter({"instruction": 1, "schema": 1, "prefill": 1})
        if Counter(fields) != expected or Counter(all_fields) != expected:
            raise ParameterError(
                "Prompt template must contain each of instruction, schema, and prefill exactly once "
                "without conversions or format specifiers."
            )


@dataclass(frozen=True, slots=True)
class PromptEncoding:
    """Stable rendered prompt output."""

    text: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    segment_offsets: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.segment_offsets, tuple) or any(
            not isinstance(segment, tuple) or len(segment) != 2 for segment in self.segment_offsets
        ):
            raise ParameterError("Prompt segment offsets must name positions within the prompt IDs.")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
            or offset > len(self.input_ids)
            for name, offset in self.segment_offsets
        ):
            raise ParameterError("Prompt segment offsets must name positions within the prompt IDs.")


@dataclass(frozen=True, slots=True)
class RecordEncoding:
    """Stable encoding for one terminal-LF JSONL record."""

    utf8: bytes
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RecordBatch:
    """Stable variable-length batch of JSONL record encodings."""

    records: tuple[RecordEncoding, ...]

    @property
    def input_ids(self) -> tuple[tuple[int, ...], ...]:
        """Return record token IDs in input order."""
        return tuple(record.input_ids for record in self.records)

    @property
    def attention_mask(self) -> tuple[tuple[int, ...], ...]:
        """Return record attention masks in input order."""
        return tuple(record.attention_mask for record in self.records)


@dataclass(frozen=True, slots=True)
class PaddedTokenBatch:
    """Stable rectangular token batch."""

    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class TokenBatch:
    """Stable variable-length token batch."""

    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class TrainingEncoding:
    """Stable framed training output for later assembler migration."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TrainingCapacity:
    """Exact record-token capacity for one immutable training frame."""

    context_limit: int
    prompt_tokens: int
    sequence_count: int
    delimiter_tokens_per_sequence: int
    delimiter_tokens: int
    record_token_capacity: int


@dataclass(frozen=True, slots=True)
class TokenizerProbe:
    """One deterministic native tokenizer parity probe."""

    text: str
    input_ids: tuple[int, ...]
    decoded: str


@dataclass(frozen=True, slots=True)
class NativeTokenizerSnapshot:
    """Normalized native identity and parity observations."""

    class_name: str
    vocab_size: int
    total_size: int
    added_vocabulary: tuple[tuple[str, int], ...]
    special_token_ids: tuple[tuple[str, int | tuple[int, ...] | None], ...]
    probes: tuple[TokenizerProbe, ...]
    fingerprint: str
