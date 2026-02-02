# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import string
from typing import Any, Dict, List, Optional, Tuple, Union

ValuePathItem = Union[str, int]
ValuePath = Tuple[ValuePathItem, ...]
"""
A tuple representing a path to a value in a document.
Each item in the tuple represents name of the nested field or array.

For example: ``("user", "emails", 0, "address")`` represents path to
the ``test@example.com`` in a structure like:

{"user": {"emails": [{"address": "test@example.com"}]}}
"""


def value_path(*args: ValuePathItem) -> ValuePath:
    """
    Convenient way of creating a value path.
    For example ``value_path("user", "emails", 0)``.
    """
    return args


def value_path_to_field_name(path: ValuePath) -> str:
    return ".".join(item for item in path if isinstance(item, str))


_JSON_PATH_ALLOWED_CHARS = set(string.ascii_letters + string.digits + "_")


def _needs_json_path_bracket_notation(field_name: str) -> bool:
    """
    Returns ``True`` if bracket notation is required for given field_name.
    For example:
    - $.my_field - dot notation can be used here
    - $['my$field'] - contains special char, so need to use bracket notation.
    """
    # field_name being empty string requires bracket notation
    if field_name == "":
        return True

    return any(char not in _JSON_PATH_ALLOWED_CHARS for char in field_name)


def value_path_to_json_path(path: ValuePath) -> str:
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


def unflatten(data: Dict[ValuePath, Any]) -> Optional[Union[dict, list]]:
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


def _unflatten_path(result: Optional[Union[dict, list]], path: ValuePath, value: Any) -> Union[dict, list]:
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
        _unflatten_recursive(result, item, items, value)  # ty: ignore[invalid-argument-type]

        return result
    else:
        if result is None:
            result = {}
        if not isinstance(result, dict):
            raise InvalidPath(f"Cannot unflatten: at index '_root' was expecting dict, but was {type(result)}.")

        _ensure_dict_key(result, item)
        _unflatten_recursive(result, item, items, value)

        return result


def _unflatten_recursive(result: Union[dict, list], prev_item: ValuePathItem, items: List[ValuePathItem], value: Any):
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
