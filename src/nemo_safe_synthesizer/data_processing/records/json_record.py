# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""JSON record flattening and field-level tracking.

Provides ``JSONRecord`` for unpacking arbitrarily nested JSON objects
(or plain strings) into flat ``KVPair`` lists, and the ``flatten``
utility that recursively collapses nested dicts/lists into a single-level
dict using the ``NESTING_DELIM`` / ``ARRAY_POS`` markers.
"""

from itertools import chain, starmap
from typing import Optional

from . import base
from .value_path import (
    value_path,
    value_path_to_field_name,
)


def flatten(raw, array_marker=base.ARRAY_POS):
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
    if isinstance(raw, list):
        # if the whole JSON document is an array, we wrap it in dict
        raw = {None: raw}

    def unpack_level(parent_key, parent_val):
        if isinstance(parent_val, dict):
            for key, value in parent_val.items():
                tmp = str(parent_key) + base.NESTING_DELIM + key
                yield tmp, value
        elif isinstance(parent_val, list):
            i = 0
            for value in parent_val:
                if parent_key is None:
                    tmp = array_marker + str(i)
                else:
                    tmp = str(parent_key) + base.NESTING_DELIM + array_marker + str(i)

                yield tmp, value
                i += 1
        else:
            yield parent_key, parent_val

    while True:
        raw = dict(chain.from_iterable(starmap(unpack_level, raw.items())))
        if not any(isinstance(value, dict) for value in raw.values()) and not any(
            isinstance(value, list) for value in raw.values()
        ):
            break

    return raw


def remove_array_markers(data: str) -> tuple[str, int, base.ValuePath]:
    """Strip array-position markers from a composite key and build a ``ValuePath``.

    Returns:
        A tuple of (dot-joined field name, array nesting depth, structural value path).
    """
    array_count = 0
    parts = data.split(base.NESTING_DELIM)
    path_items = []
    for part in parts:
        if part.startswith(base.ARRAY_POS):
            array_count += 1
            path_items.append(int(part[len(base.ARRAY_POS) :]))
            continue
        path_items.append(str(part))

    path = value_path(*path_items)
    return value_path_to_field_name(path), array_count, path


def convert_flat_dict_to_kv_pairs(data: dict) -> list[base.KVPair]:
    """Convert a flattened dict (from ``flatten``) into a list of ``KVPair`` objects."""
    out = []
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

    def unpack(self):
        flattened_dict = flatten({"": self.original} if isinstance(self.original, str) else self.original)

        kv_pairs = convert_flat_dict_to_kv_pairs(flattened_dict)
        for pair in kv_pairs:
            self.fields.add(pair.field)
            self.kv_pairs.append(pair)

    def value_for_json_path(self, json_path: str) -> Optional[str]:
        """Return the string value at ``json_path``, or None if not found."""
        for pair in self.kv_pairs:
            if pair.json_path == json_path:
                return str(pair.value)

    def value_for_value_path(self, path: base.ValuePath) -> Optional[str]:
        """Return the string value at ``path``, or None if not found."""
        for pair in self.kv_pairs:
            if pair.value_path == path:
                return str(pair.value)

    def flattened(self) -> dict[base.ValuePath, object]:
        """Return a dict mapping each ``ValuePath`` to its scalar value."""
        return {x.value_path: x.value for x in self.kv_pairs}
