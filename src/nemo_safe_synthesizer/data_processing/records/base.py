# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class that all record types can inherit from."""

import re
from abc import ABC, abstractmethod
from numbers import Number
from typing import Iterable, List, Union

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
ARRAY_POS = "_gretelarray_"
NESTING_DELIM = "*#N#*"
DELIM = "."

WORD_TOKENIZER = re.compile(r"(?!\d)\w+", re.IGNORECASE)


def tokenize_on_upper(data: str) -> List[str]:
    if not data:
        return []
    out = []
    curr = []
    curr.append(data[0])  # seed the curr word
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


def tokenize_header(field: str) -> List[str]:
    out = []
    # NOTE(jm): for the purposes of header tokenization, we
    # don't consider `_` to be a word character, so we just
    # replace them with `-` so they'll be split on
    field = field.replace("_", "-")
    base_tokens = re.findall(WORD_TOKENIZER, field)
    for token in base_tokens:
        out.extend(tokenize_on_upper(token))
    return out


def get_type_as_string(value):
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
    __slots__ = (FIELD, VALUE, ARRAY_COUNT, SCALAR_TYPE, FIELD_TOKENS, VALUE_PATH)

    def __init__(
        self,
        field: str,
        value: Union[str, Number],
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
        return value_path_to_json_path(self.value_path)

    def as_dict(self):
        return {
            FIELD: self.field,
            VALUE: self.value,
            SCALAR_TYPE: self.scalar_type,
            ARRAY_COUNT: self.array_count,
        }


class BaseRecord(ABC):
    __slots__ = (ORIGINAL, KV_PAIRS, FIELDS)

    def __init__(self, original):
        self.original = original
        self.kv_pairs = []
        self.fields = set()
        self.unpack()

    @abstractmethod
    def unpack(self):  # pragma: no cover
        """Must be implemented by sub-classes and should
        handle the unpacking and loading of the record
        data as attrs from ``self.original`` onto the
        object
        """
        pass

    def as_dict(self):
        out = {
            ORIGINAL: self.original,
            KV_PAIRS: [p.as_dict() for p in self.kv_pairs],
            FIELDS: list(self.fields),
        }
        return out


def normalize_labels(labels: Iterable[str]) -> set[str]:
    """
    Normalize labels by converting them to lowercase.
    """
    return {normalize_label(label) for label in labels}


def normalize_label(label: str) -> str:
    return label.lower()
