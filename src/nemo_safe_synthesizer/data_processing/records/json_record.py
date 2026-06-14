# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""JSON record flattening and field-level tracking.

Provides ``JSONRecord`` for unpacking arbitrarily nested JSON objects
(or plain strings) into flat ``KVPair`` lists, and the ``flatten``
utility that recursively collapses nested dicts/lists into a single-level
dict using the ``NESTING_DELIM`` / ``ARRAY_POS`` markers.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import chain, starmap
from typing import Any, Optional

from typing_extensions import override

from . import base
from .json_types import JsonArray, JsonContainer, JsonObject, JsonScalar, JsonValue
from .value_path import (
    value_path,
    value_path_to_field_name,
)

__all__ = [
    "JSONRecord",
    "JsonArray",
    "JsonContainer",
    "JsonObject",
    "JsonScalar",
    "JsonValue",
    "convert_flat_dict_to_kv_pairs",
    "flatten",
    "remove_array_markers",
]


def flatten(raw: JsonContainer, array_marker: str = base.ARRAY_POS) -> dict[object, object]:
    """Recursively flatten a nested dict/list into a single-level dict.

    Keys are joined with ``NESTING_DELIM``; array indices are encoded as
    ``{array_marker}{index}``. Top-level lists are wrapped in a dict with
    a None key before flattening.

    Args:
        raw: Nested dict or list to flatten.
        array_marker: Prefix used to mark array indices in keys.

    Returns:
        A flat dict mapping composite keys to scalar values.
    """
    match raw:
        case list() as values:
            # if the whole JSON document is an array, we wrap it in dict
            flattened: dict[object, object] = {None: values}
        case dict() as values:
            flattened = {key: value for key, value in values.items()}
        case _:
            raise TypeError("flatten expects a JSON object or array")

    def unpack_level(parent_key: object, parent_val: object) -> Iterator[tuple[object, object]]:
        match parent_val:
            case dict() as values:
                for key, value in values.items():
                    yield str(parent_key) + base.NESTING_DELIM + str(key), value
            case list() as values:
                for i, value in enumerate(values):
                    if parent_key is None:
                        tmp = array_marker + str(i)
                    else:
                        tmp = str(parent_key) + base.NESTING_DELIM + array_marker + str(i)

                    yield tmp, value
            case scalar:
                yield parent_key, scalar

    while True:
        flattened = dict(chain.from_iterable(starmap(unpack_level, flattened.items())))
        if not any(isinstance(value, dict) for value in flattened.values()) and not any(
            isinstance(value, list) for value in flattened.values()
        ):
            break

    return flattened


def remove_array_markers(data: str) -> tuple[str, int, base.ValuePath]:
    """Strip array-position markers from a composite key and build a ``ValuePath``.

    Returns:
        A tuple of (dot-joined field name, array nesting depth, structural value path).
    """
    array_count = 0
    parts = data.split(base.NESTING_DELIM)
    path_items: list[str | int] = []
    for part in parts:
        if part.startswith(base.ARRAY_POS):
            array_count += 1
            path_items.append(int(part[len(base.ARRAY_POS) :]))
            continue
        path_items.append(str(part))

    path = value_path(*path_items)
    return value_path_to_field_name(path), array_count, path


def convert_flat_dict_to_kv_pairs(data: dict[Any, Any]) -> list[base.KVPair]:
    """Convert a flattened dict (from ``flatten``) into a list of ``KVPair`` objects."""
    out: list[base.KVPair] = []
    for k, v in data.items():
        k = str(k)
        new_key, array_count, value_path = remove_array_markers(k)
        flat = base.KVPair(new_key, v, base.get_type_as_string(v), array_count, value_path)
        out.append(flat)
    return out


class JSONRecord(base.BaseRecord):
    """Record backed by a JSON object (dict) or bare string.

    On construction, the original value is flattened into ``KVPair`` entries.
    Provides lookup by JSONPath or ``ValuePath``.
    """

    def __init__(self, original: Any):
        super().__init__(original)
        self._unpack_json()

    def _unpack_json(self) -> None:
        flattened_dict = flatten({"": self.original} if isinstance(self.original, str) else self.original)

        kv_pairs = convert_flat_dict_to_kv_pairs(flattened_dict)
        for pair in kv_pairs:
            self.fields.add(pair.field)
            self.kv_pairs.append(pair)

    @override
    def unpack(self) -> None:
        self.kv_pairs = []
        self.fields = set()
        self._unpack_json()

    def value_for_json_path(self, json_path: str) -> Optional[str]:
        """Return the string value at ``json_path``, or None if not found."""
        for pair in self.kv_pairs:
            if pair.json_path == json_path:
                return str(pair.value)
        return None

    def value_for_value_path(self, path: base.ValuePath) -> Optional[str]:
        """Return the string value at ``path``, or None if not found."""
        for pair in self.kv_pairs:
            if pair.value_path == path:
                return str(pair.value)
        return None

    def flattened(self) -> dict[base.ValuePath, object]:
        """Return a dict mapping each ``ValuePath`` to its scalar value."""
        return {x.value_path: x.value for x in self.kv_pairs}
