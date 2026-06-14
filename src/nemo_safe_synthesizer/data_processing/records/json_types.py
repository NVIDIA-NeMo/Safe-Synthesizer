# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared recursive JSON type aliases and runtime shape guards."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from typing_extensions import TypeIs

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | dict[str, "JsonValue"] | list["JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
JsonArray: TypeAlias = list[JsonValue]
JsonContainer: TypeAlias = JsonObject | JsonArray
JsonSchema: TypeAlias = Mapping[str, JsonValue]


def is_json_value(value: object) -> TypeIs[JsonValue]:
    """Return whether ``value`` is representable as JSON."""
    match value:
        case str() | int() | float() | bool() | None:
            return True
        case list() as values:
            return all(is_json_value(item) for item in values)
        case dict() as values:
            return all(isinstance(key, str) and is_json_value(item) for key, item in values.items())
        case _:
            return False


def is_json_object(value: object) -> TypeIs[JsonObject]:
    """Return whether ``value`` is a JSON object with string keys."""
    return isinstance(value, dict) and all(isinstance(key, str) and is_json_value(item) for key, item in value.items())
