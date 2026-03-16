# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base record representation and field-level tokenization utilities.

Provides ``BaseRecord`` -- the abstract base for record types used by the
PII replacer -- along with ``KVPair`` for representing flattened key-value
entries, and helpers for tokenizing field names (``tokenize_header``,
``tokenize_on_upper``).
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from numbers import Number

from .value_path import (
    ValuePath,
    value_path_to_json_path,
)

FIELD = "field"
FIELD_TOKENS = "field_tokens"
VALUE_PATH = "value_path"
FIELDS = "fields"
VALUE = "value"
ARRAY_COUNT = "array_count"
SCALAR_TYPE = "scalar_type"
ORIGINAL = "original"
KV_PAIRS = "kv_pairs"
STRING = "string"
BOOL = "boolean"
NUMBER = "number"
NULL = "null"
ARRAY_POS = "_nssarray_"
NESTING_DELIM = "*#N#*"
DELIM = "."

WORD_TOKENIZER = re.compile(r"(?!\d)\w+", re.IGNORECASE)


def tokenize_on_upper(data: str) -> list[str]:
    """Split a camelCase or PascalCase string into lowercase tokens.

    Args:
        data: String to tokenize.

    Returns:
        List of lowercase token strings, or an empty list if ``data`` is empty.
    """
    if not data:
        return []
    out = []
    curr = []
    curr.append(data[0])
    for i in range(1, len(data)):
        if data[i].isupper() and data[i - 1].islower():
            out.append("".join(curr).casefold())
            curr = []
        elif i < len(data) - 1 and data[i].isupper() and data[i + 1].islower():
            out.append("".join(curr).casefold())
            curr = []
        curr.append(data[i])
    out.append("".join(curr).casefold())
    return out


def tokenize_header(field: str) -> list[str]:
    """Tokenize a field/column name into lowercase word tokens.

    Underscores are treated as separators, and camelCase boundaries are
    split via ``tokenize_on_upper``.

    Args:
        field: Field name to tokenize.

    Returns:
        List of lowercase word tokens extracted from the field name.
    """
    out = []
    # For header tokenization we don't consider `_` a word character,
    # so replace with `-` to split on it.
    field = field.replace("_", "-")
    base_tokens = re.findall(WORD_TOKENIZER, field)
    for token in base_tokens:
        out.extend(tokenize_on_upper(token))
    return out


def get_type_as_string(value) -> str:
    """Return the JSON schema type name for a Python scalar value."""
    if isinstance(value, str):
        return STRING
    elif isinstance(value, bool):
        if str(value) in ("True", "False"):
            return BOOL
    elif isinstance(value, Number):
        return NUMBER
    elif value is None:
        return NULL

    return NULL


class KVPair:
    """A single flattened key-value entry from a record.

    Stores the field name, value, scalar type, nesting depth (array count),
    and the structural path to the value in the original document.

    Args:
        field: Dot-joined field name (array markers removed).
        value: The scalar value.
        scalar_type: JSON schema type string (``"string"``, ``"number"``, etc.).
        array_count: Number of array levels this value is nested within.
        value_path: Structural path tuple identifying the value's location.
    """

    __slots__ = (FIELD, VALUE, ARRAY_COUNT, SCALAR_TYPE, FIELD_TOKENS, VALUE_PATH)

    def __init__(
        self,
        field: str,
        value: str | Number,
        scalar_type: str,
        array_count: int,
        value_path: ValuePath,
    ):
        self.field = field
        self.value = value
        self.scalar_type = scalar_type
        self.array_count = array_count
        self.value_path = value_path
        self.field_tokens = tokenize_header(field)

    @property
    def json_path(self):
        """JSONPath string (e.g., ``$.user.emails[0].address``)."""
        return value_path_to_json_path(self.value_path)

    def as_dict(self):
        """Serialize to a dictionary of field, value, scalar_type, and array_count."""
        return {
            FIELD: self.field,
            VALUE: self.value,
            SCALAR_TYPE: self.scalar_type,
            ARRAY_COUNT: self.array_count,
        }


class BaseRecord(ABC):
    """Abstract base for structured record representations.

    Subclasses implement ``unpack`` to flatten the original record into a
    list of ``KVPair`` entries and a set of field names.

    Args:
        original: The raw record data (typically a dict or string).
    """

    __slots__ = (ORIGINAL, KV_PAIRS, FIELDS)

    def __init__(self, original):
        self.original = original
        self.kv_pairs = []
        self.fields = set()
        self.unpack()

    @abstractmethod
    def unpack(self):  # pragma: no cover
        """Flatten ``self.original`` into ``self.kv_pairs`` and ``self.fields``.

        Must be implemented by subclasses to handle format-specific unpacking
        (e.g., JSON objects, CSV rows).
        """
        pass

    def as_dict(self):
        """Serialize the record to a dictionary with original data, kv_pairs, and fields."""
        out = {
            ORIGINAL: self.original,
            KV_PAIRS: [p.as_dict() for p in self.kv_pairs],
            FIELDS: list(self.fields),
        }
        return out


def normalize_labels(labels: Iterable[str]) -> set[str]:
    """Normalize labels by converting them to lowercase."""
    return {normalize_label(label) for label in labels}


def normalize_label(label: str) -> str:
    """Convert a single label to lowercase."""
    return label.lower()
