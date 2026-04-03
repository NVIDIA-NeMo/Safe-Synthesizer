# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import string
from typing import Any, Optional

ValuePathItem = str | int
ValuePath = tuple[ValuePathItem, ...]
"""A tuple representing a path to a value in a document.

Each item is either a field name (``str``) or an array index (``int``).

Example::

    ("user", "emails", 0, "address")

represents the path to ``"test@example.com"`` in::

    {"user": {"emails": [{"address": "test@example.com"}]}}
"""


def value_path(*args: ValuePathItem) -> ValuePath:
    """Create a ``ValuePath`` tuple from positional arguments.

    Example::

        value_path("user", "emails", 0)
    """
    return args


def value_path_to_field_name(path: ValuePath) -> str:
    """Join the string components of a ``ValuePath`` with dots, ignoring array indices."""
    return ".".join(item for item in path if isinstance(item, str))


_JSON_PATH_ALLOWED_CHARS = set(string.ascii_letters + string.digits + "_")


def _needs_json_path_bracket_notation(field_name: str) -> bool:
    """Return True if JSONPath bracket notation is needed for ``field_name``.

    Bracket notation is required when the field name is empty or contains
    characters outside ``[a-zA-Z0-9_]``.
    """
    # field_name being empty string requires bracket notation
    if field_name == "":
        return True

    return any(char not in _JSON_PATH_ALLOWED_CHARS for char in field_name)


def value_path_to_json_path(path: ValuePath) -> str:
    """Convert a ``ValuePath`` to a JSONPath string (e.g., ``$.user.emails[0].address``)."""
    json_path = ["$"]
    for item in path:
        if isinstance(item, int):
            # array index
            json_path[-1] += "[" + str(item) + "]"
        elif _needs_json_path_bracket_notation(item):
            # bracket notation
            json_path[-1] += "['" + item.replace("'", "\\'") + "']"
        else:
            # dot notation
            json_path.append(item)

    return ".".join(json_path)


class InvalidPath(Exception): ...


def unflatten(data: dict[ValuePath, Any]) -> Optional[dict | list]:
    """Reconstruct a nested dict/list from a flat ``{ValuePath: value}`` mapping.

    Args:
        data: Flat mapping of value paths to scalar values.

    Returns:
        A nested dict or list, or None if ``data`` is empty.

    Raises:
        InvalidPath: If paths are structurally inconsistent (e.g., a path
            expects a list where a dict already exists).
    """
    result = None
    for key, value in data.items():
        try:
            result = _unflatten_path(result, key, value)
        except InvalidPath as e:
            raise InvalidPath(f"Cannot unflatten path '{key}'") from e

    return result


def _ensure_array_size(result: list, item: int):
    if len(result) <= item:
        for i in range(len(result), item + 1):
            result.append(None)


def _ensure_dict_key(result: dict, item: str):
    if item not in result:
        result[item] = None


def _unflatten_path(result: Optional[dict | list], path: ValuePath, value: Any) -> dict | list:
    # Note: result will be a list when working with an array at this level of
    # the path, and thus the first element of path is an integer. Otherwise
    # working with an object at this level of the path, result will be a dict
    # and the first element of path is a string.
    items = list(path)
    item, *items = items

    if isinstance(item, int):
        if result is None:
            result = []
        if not isinstance(result, list):
            raise InvalidPath(f"Cannot unflatten: at index '_root' was expecting list, but was {type(result)}.")

        _ensure_array_size(result, item)
        _unflatten_recursive(result, item, items, value)

        return result
    else:
        if result is None:
            result = {}
        if not isinstance(result, dict):
            raise InvalidPath(f"Cannot unflatten: at index '_root' was expecting dict, but was {type(result)}.")

        _ensure_dict_key(result, item)
        _unflatten_recursive(result, item, items, value)

        return result


def _unflatten_recursive(result: Any, prev_item: ValuePathItem, items: list[ValuePathItem], value: Any):
    # Note: result will be a list when working with an array at this level of
    # the path, and thus the first element of path is an integer. Otherwise
    # working with an object at this level of the path, result will be a dict
    # and the first element of path is a string.
    if not items:
        result[prev_item] = value
        return

    item, *items = items
    if isinstance(item, int):
        if result[prev_item] is None:
            result[prev_item] = []

        current_level = result[prev_item]
        if not isinstance(current_level, list):
            raise InvalidPath(
                f"Cannot unflatten: at index '{prev_item}' was expecting list, but was {type(current_level)}."
            )

        _ensure_array_size(current_level, item)

        _unflatten_recursive(current_level, item, items, value)
    else:
        if result[prev_item] is None:
            result[prev_item] = {}

        current_level = result[prev_item]
        if not isinstance(current_level, dict):
            raise InvalidPath(
                f"Cannot unflatten: at index '{prev_item}' was expecting dict, but was {type(current_level)}."
            )

        _ensure_dict_key(current_level, item)

        _unflatten_recursive(result[prev_item], item, items, value)
