# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt construction helpers for time-series generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from ..data_processing.record_utils import ParsedRecord, records_to_jsonl
from ..defaults import PSEUDO_GROUP_COLUMN
from ..errors import GenerationError
from ..llm.metadata import LLMPromptConfig

__all__ = [
    "build_partial_record_prefix",
    "build_rolling_record_prefill",
    "build_training_compatible_prompt_token_ids",
]


class EncodeOnlyTokenizer(Protocol):
    """Tokenizer interface required for time-series prompt construction."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Encode text without injecting tokenizer-defined special tokens."""


def _schema_types(schema: Mapping[str, object], column: str) -> list[str]:
    """Return the declared JSON types for a schema column.

    Prefix seed values do not always retain their training-data types: group
    IDs become strings when stored as dictionary keys, and configured
    timestamps may be strings even when the schema expects a number. Reading
    and normalizing both scalar and list-valued JSON Schema ``type`` fields
    lets the caller restore those values to the training-compatible JSON type.

    Args:
        schema: Saved JSON schema containing a ``properties`` mapping.
        column: Column whose declared types should be returned.

    Returns:
        Declared type names, or an empty list when the column has no standard
        string or list-valued ``type`` declaration.

    Raises:
        GenerationError: If the schema has no properties mapping or does not
            define the requested column.
    """
    properties_value = schema.get("properties")
    if not isinstance(properties_value, Mapping):
        raise GenerationError("The saved schema has no valid 'properties' mapping.")
    properties = cast(Mapping[str, object], properties_value)
    column_schema_value = properties.get(column)
    if not isinstance(column_schema_value, Mapping):
        raise GenerationError(f"The saved schema has no definition for time-series prefix column {column!r}.")
    column_schema = cast(Mapping[str, object], column_schema_value)
    schema_type = column_schema.get("type")
    if isinstance(schema_type, str):
        return [schema_type]
    if isinstance(schema_type, list) and all(isinstance(item, str) for item in schema_type):
        return cast(list[str], schema_type)
    return []


def _coerce_prefix_value(schema: Mapping[str, object], column: str, value: object) -> object:
    """Coerce a known prefix value to the column's JSON schema type.

    Args:
        schema: Saved JSON schema used to determine the target type.
        column: Prefix column receiving the value.
        value: Group or timestamp value to serialize.

    Returns:
        The value converted to the declared integer, number, or string type.
        Values with other or unspecified schema types are returned unchanged.

    Raises:
        GenerationError: If schema lookup fails or a numeric conversion is not
            possible.
    """
    schema_types = _schema_types(schema, column)
    try:
        if "integer" in schema_types:
            return int(value)  # ty: ignore[invalid-argument-type]
        if "number" in schema_types:
            return float(value)  # ty: ignore[invalid-argument-type]
    except (TypeError, ValueError) as exc:
        raise GenerationError(
            f"Could not serialize value {value!r} for time-series prefix column {column!r} as {schema_types!r}."
        ) from exc
    if "string" in schema_types and value is not None:
        return str(value)
    return value


def build_partial_record_prefix(
    *,
    columns: Sequence[str],
    schema: Mapping[str, object],
    group_column: str | None,
    group_id: object,
    timestamp_column: str | None,
    start_timestamp: object,
) -> str:
    """Build a training-dialect incomplete first record for generation.

    Args:
        columns: Saved JSON schema columns in generation order.
        schema: Saved JSON schema used to coerce known values.
        group_column: Configured group column, including the pseudo-group value.
        group_id: Group value for this generation stream.
        timestamp_column: Configured timestamp column.
        start_timestamp: First timestamp to generate.

    Returns:
        An incomplete JSON record ending with the opening quote of the next
        field name. Including the training ``,"`` token keeps the standalone
        prefix tokenization identical to the beginning of a complete training
        record. The record begins directly with ``{`` because training places
        the sequence BOS token immediately before the first JSON byte.

    Raises:
        GenerationError: If the saved artifact cannot support a partial prefix.
    """
    if timestamp_column is None:
        raise GenerationError("Partial-record time-series generation requires a timestamp column.")
    if start_timestamp is None:
        raise GenerationError("Partial-record time-series generation requires a resolved start timestamp.")

    seed_values: dict[str, object] = {}
    if group_column != PSEUDO_GROUP_COLUMN:
        if group_column is None:
            raise GenerationError("Grouped time-series metadata is missing its group column.")
        seed_values[group_column] = group_id
    seed_values[timestamp_column] = start_timestamp

    missing_columns = [column for column in seed_values if column not in columns]
    if missing_columns:
        raise GenerationError(
            "The saved schema is missing required partial-prefix columns: "
            f"{', '.join(repr(column) for column in missing_columns)}. Retrain the time-series artifact."
        )

    prefix_columns = [column for column in columns if column in seed_values]
    if list(columns[: len(prefix_columns)]) != prefix_columns:
        raise GenerationError(
            "The saved schema does not begin with the time-series identity columns "
            f"{prefix_columns!r}. Retrain the artifact so group and timestamp columns are first."
        )
    if len(prefix_columns) == len(columns):
        raise GenerationError(
            "Partial-record time-series generation requires at least one non-identity column after the prefix."
        )

    ordered_values = {column: _coerce_prefix_value(schema, column, seed_values[column]) for column in prefix_columns}
    serialized = records_to_jsonl([ordered_values]).rstrip("\n")
    if not serialized.endswith("}"):
        raise GenerationError("Could not serialize the initial time-series partial record.")
    return f'{serialized[:-1]},"'


def build_rolling_record_prefill(records: Sequence[ParsedRecord]) -> str:
    """Build rolling prompt context from exact model-emitted record text.

    Training inserts the sequence BOS token immediately before the first record,
    with no intervening whitespace. The caller adds that BOS token separately,
    so this helper returns only the exact newline-terminated record bytes.

    Args:
        records: Accepted records in chronological order.

    Returns:
        A sequence of newline-terminated records, or an empty string when no
        records are provided.
    """
    if not records:
        return ""
    return "".join(f"{record.text}\n" for record in records)


def build_training_compatible_prompt_token_ids(
    *,
    tokenizer: EncodeOnlyTokenizer,
    prompt_config: LLMPromptConfig,
    instruction: str,
    schema_fragment: str,
    prefill: str | Sequence[str],
) -> list[int]:
    """Build the token prefix seen before record continuation during training.

    This mirrors ``Example`` construction: encode the schema prompt without
    tokenizer-defined special tokens, apply the configured prompt BOS/EOS
    tokens, add the sequence BOS token, then append the partial or rolling
    record context. Building IDs explicitly is necessary for tokenizers such as
    SmolLM3's, which do not add ``<|im_start|>`` automatically at inference.

    Args:
        tokenizer: Tokenizer used by the generation engine.
        prompt_config: Saved prompt template and special-token settings.
        instruction: Training instruction inserted into the prompt template.
        schema_fragment: Ordered column placeholder fragment.
        prefill: Partial first record, or separately encoded rolling records.
            Encoding rolling records individually mirrors training, which
            tokenizes each newline-terminated JSON record before concatenation.

    Returns:
        Prompt token IDs whose boundary exactly matches a training example.
    """
    prompt = prompt_config.template.format(
        instruction=instruction,
        schema=schema_fragment,
        prefill="",
    )
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    if prompt_config.add_bos_token_to_prompt:
        prompt_ids.insert(0, prompt_config.bos_token_id)
    if prompt_config.add_eos_token_to_prompt:
        prompt_ids.append(prompt_config.eos_token_id)

    prompt_ids.append(prompt_config.bos_token_id)
    prefill_segments = [prefill] if isinstance(prefill, str) else prefill
    for segment in prefill_segments:
        prompt_ids.extend(tokenizer.encode(segment, add_special_tokens=False))
    return prompt_ids
