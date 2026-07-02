# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prompt helpers for time-series cold-start experiments."""

from __future__ import annotations

import json
from collections.abc import Sequence

from ..defaults import PSEUDO_GROUP_COLUMN


def format_cold_start_instruction(
    *,
    base_instruction: str,
    template: str | None,
    group_id: object,
    group_column: str | None,
    timestamp_column: str,
    start_timestamp: object,
    stop_timestamp: object,
    timestamp_interval_seconds: int | None,
) -> str:
    """Build the per-group instruction used by the start-instruction strategy."""
    instruction_template = template or (
        "Start group {group_id} at timestamp {start_timestamp}. Generate complete JSONL records."
        " for the configured schema through {stop_timestamp}, using {timestamp_interval_seconds}-second intervals."
    )
    suffix = instruction_template.format(
        group_id=group_id,
        group_column=group_column,
        timestamp_column=timestamp_column,
        start_timestamp=start_timestamp,
        stop_timestamp=stop_timestamp,
        timestamp_interval_seconds=timestamp_interval_seconds,
    )
    return f"{base_instruction.rstrip()} {suffix.strip()}"


def build_partial_record_prefix(
    *,
    columns: Sequence[str],
    schema: dict,
    group_column: str | None,
    group_id: object,
    timestamp_column: str | None,
    start_timestamp: object,
) -> str:
    """Build an incomplete first JSON record in schema order."""
    known_fields: dict[str, object] = {}
    if group_column and group_column != PSEUDO_GROUP_COLUMN and group_column in columns:
        known_fields[group_column] = group_id
    if timestamp_column and timestamp_column in columns:
        known_fields[timestamp_column] = start_timestamp
    if not known_fields:
        return ""
    fragments = [
        format_partial_prefix_field(column, known_fields[column], schema)
        for column in columns
        if column in known_fields
    ]
    if not fragments:
        return ""
    return "{" + ",".join(fragments) + ","


def format_partial_prefix_field(column: str, value: object, schema: dict) -> str:
    """Format one schema-ordered partial-prefix field as compact JSON."""
    coerced_value = coerce_partial_prefix_value(column, value, schema)
    key_json = json.dumps(column, ensure_ascii=False, separators=(",", ":"))
    value_json = json.dumps(coerced_value, ensure_ascii=False, separators=(",", ":"))
    return f"{key_json}:{value_json}"


def coerce_partial_prefix_value(column: str, value: object, schema: dict) -> object:
    """Coerce known prefix values to match the JSON schema type."""
    column_schema = schema.get("properties", {}).get(column, {})
    schema_type = column_schema.get("type")
    schema_types = schema_type if isinstance(schema_type, list) else [schema_type]

    if "integer" in schema_types:
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if "number" in schema_types:
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if "string" in schema_types and value is not None:
        return str(value)
    return value
