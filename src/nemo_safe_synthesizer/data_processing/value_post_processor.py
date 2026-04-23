# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-column post-processing for schema-external (positional) generation.

Under the positional serialization format the model emits raw values only
(no JSON keys, no braces); the detected JSON schema is imposed here. Each
column's raw string value is coerced, clamped, or snapped according to its
schema entry so the output row is schema-valid by construction.

Type handling:

* ``integer`` / ``number``: parsed as ``int``/``float``, clamped to
  ``[minimum, maximum]`` when bounds are present.
* ``string`` with ``enum``: exact → case-insensitive → close-match (edit
  distance) against the vocabulary. Non-match within cutoff raises.
* ``string`` without ``enum``: passthrough, optionally truncated to
  ``maxLength``.
* List-of-types (e.g. ``["integer", "null"]``): any registered null
  sentinel returns ``None``; otherwise dispatched by the first non-null
  type.

Failure semantics match the existing JSON path: an unparseable or
non-matching value raises ``ValuePostProcessingError`` and the row is
dropped downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any

from ..observability import get_logger

logger = get_logger()


NULL_SENTINELS: frozenset[str] = frozenset({"", "null", "None", "NaN", "nan", "NULL"})
ENUM_CLOSE_MATCH_CUTOFF: float = 0.6


class ValuePostProcessingError(ValueError):
    """Raised when a raw string cannot be coerced to its column's schema."""


@dataclass(frozen=True)
class _ColumnSpec:
    name: str
    types: tuple[str, ...]
    nullable: bool
    minimum: float | None
    maximum: float | None
    min_length: int | None
    max_length: int | None
    enum: tuple[Any, ...] | None
    enum_str_to_value: dict[str, Any] | None
    enum_str_lower_to_value: dict[str, Any] | None


class ValuePostProcessor:
    """Imposes a detected JSON schema on raw string values.

    Pre-computes per-column specs at construction time so ``process`` is
    cheap in the inner generation loop. The schema is the same dict
    produced by :func:`data_processing.dataset.make_json_schema`.
    """

    def __init__(self, schema: dict, enum_close_match_cutoff: float = ENUM_CLOSE_MATCH_CUTOFF) -> None:
        self._specs: dict[str, _ColumnSpec] = {}
        self._enum_cutoff = enum_close_match_cutoff

        properties = schema.get("properties", {})
        for col_name, col_schema in properties.items():
            self._specs[col_name] = self._build_spec(col_name, col_schema)

    @property
    def columns(self) -> list[str]:
        """Column names in schema-declared order."""
        return list(self._specs.keys())

    def process(self, col_name: str, raw: str) -> Any:
        """Coerce ``raw`` to the value expected by ``col_name``'s schema.

        Raises :class:`ValuePostProcessingError` if the string cannot be
        coerced; callers are expected to drop the row.
        """
        spec = self._specs.get(col_name)
        if spec is None:
            raise ValuePostProcessingError(f"unknown column: {col_name!r}")

        stripped = raw.strip()
        if spec.nullable and stripped in NULL_SENTINELS:
            return None

        if spec.enum is not None:
            return self._coerce_enum(stripped, spec)

        primary_type = self._primary_type(spec)
        if primary_type in ("integer", "number"):
            return self._coerce_numeric(stripped, spec, primary_type)
        if primary_type == "string":
            return self._coerce_string(stripped, spec)
        if primary_type == "boolean":
            return self._coerce_bool(stripped)

        raise ValuePostProcessingError(
            f"column {col_name!r}: no handler for type {primary_type!r}"
        )

    def _build_spec(self, col_name: str, col_schema: dict) -> _ColumnSpec:
        raw_type = col_schema.get("type")
        if raw_type is None and "enum" in col_schema:
            types: tuple[str, ...] = ("string",)
        elif isinstance(raw_type, list):
            types = tuple(raw_type)
        elif isinstance(raw_type, str):
            types = (raw_type,)
        else:
            types = ("string",)

        nullable = "null" in types

        enum_values = col_schema.get("enum")
        enum_tuple: tuple[Any, ...] | None = tuple(enum_values) if enum_values is not None else None
        enum_str_to_value: dict[str, Any] | None = None
        enum_str_lower_to_value: dict[str, Any] | None = None
        if enum_tuple is not None:
            enum_str_to_value = {str(v): v for v in enum_tuple}
            enum_str_lower_to_value = {str(v).lower(): v for v in enum_tuple}

        return _ColumnSpec(
            name=col_name,
            types=types,
            nullable=nullable,
            minimum=_as_float(col_schema.get("minimum")),
            maximum=_as_float(col_schema.get("maximum")),
            min_length=_as_int(col_schema.get("minLength")),
            max_length=_as_int(col_schema.get("maxLength")),
            enum=enum_tuple,
            enum_str_to_value=enum_str_to_value,
            enum_str_lower_to_value=enum_str_lower_to_value,
        )

    def _primary_type(self, spec: _ColumnSpec) -> str:
        for t in spec.types:
            if t != "null":
                return t
        return "string"

    def _coerce_numeric(self, raw: str, spec: _ColumnSpec, primary_type: str) -> int | float:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValuePostProcessingError(
                f"column {spec.name!r}: cannot parse numeric value {raw!r}"
            ) from exc

        if math.isnan(value) or math.isinf(value):
            raise ValuePostProcessingError(
                f"column {spec.name!r}: numeric value {raw!r} is not finite"
            )

        if spec.minimum is not None and value < spec.minimum:
            value = spec.minimum
        if spec.maximum is not None and value > spec.maximum:
            value = spec.maximum

        if primary_type == "integer":
            return int(round(value))
        return value

    def _coerce_string(self, raw: str, spec: _ColumnSpec) -> str:
        if spec.min_length is not None and len(raw) < spec.min_length:
            raise ValuePostProcessingError(
                f"column {spec.name!r}: value length {len(raw)} below minLength={spec.min_length}"
            )
        if spec.max_length is not None and len(raw) > spec.max_length:
            return raw[: spec.max_length]
        return raw

    def _coerce_bool(self, raw: str) -> bool:
        lowered = raw.lower()
        if lowered in ("true", "1", "yes", "y", "t"):
            return True
        if lowered in ("false", "0", "no", "n", "f"):
            return False
        raise ValuePostProcessingError(f"cannot parse boolean value {raw!r}")

    def _coerce_enum(self, raw: str, spec: _ColumnSpec) -> Any:
        assert spec.enum_str_to_value is not None and spec.enum_str_lower_to_value is not None

        if raw in spec.enum_str_to_value:
            return spec.enum_str_to_value[raw]

        lowered = raw.lower()
        if lowered in spec.enum_str_lower_to_value:
            return spec.enum_str_lower_to_value[lowered]

        candidates = get_close_matches(
            raw, list(spec.enum_str_to_value.keys()), n=1, cutoff=self._enum_cutoff
        )
        if candidates:
            return spec.enum_str_to_value[candidates[0]]

        raise ValuePostProcessingError(
            f"column {spec.name!r}: value {raw!r} does not match any enum value"
        )


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
