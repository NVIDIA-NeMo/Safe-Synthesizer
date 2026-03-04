# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import StrEnum
from functools import cached_property
from typing import Any

from pydantic import BaseModel, Field


class FieldType(StrEnum):
    EMPTY = "empty"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    OTHER = "other"
    BINARY = "binary"


class FieldFeatures(BaseModel):
    name: str
    type: FieldType

    count: int
    """
    Number of non-empty values.
    """

    unique_values_list: list[Any]
    """
    List of unique values.
    """

    unique_count: int
    """
    Number of unique values.
    """
    unique_percent: float

    missing_count: int
    """
    Number of missing values.
    It's number of records from the dataset that don't have value set for this field.
    """
    missing_percent: float

    min_str_length: int
    max_str_length: int

    avg_str_length: float
    """
    Average length of string representation of field's value.
    Note: this only includes non-missing values (i.e. we don't add 0 for fields that are missing).
    """

    # numeric
    min_value: int | float | None = Field(default=None)
    max_value: int | float | None = Field(default=None)
    min_precision: int | None = Field(default=None)
    """
    Min number of decimal spaces from all of the values for this field.
    E.g. it's 1 for number like "1.1" and 3 for "1.234".
    """
    max_precision: int | None = Field(default=None)
    """
    Max number of decimal spaces from all of the values for this field.
    """

    space_count: int | None = Field(default=None)
    """
    Number of times space character (' ') appears in field's values.
    """

    classification: dict | None = Field(default=None)
    """
    Classification information, based on labels detected in field's values.
    """

    def to_dict(self, **kwargs) -> dict:
        return self.model_dump(exclude_unset=True, exclude_none=True)


class FieldFeaturesInfo:
    """
    This class provides functionality to analyze field features. It can be used
    by other libraries instead of having to parse through the dictionaries that
    are normally stored on a manifest dataclass. This class will be init'd
    and stored on the `AnalyzerContext`
    """

    field_features: list[FieldFeatures]

    def __init__(self, field_features: list[FieldFeatures]):
        self.field_features = field_features

    @cached_property
    def text_field_count(self) -> int:
        count = 0
        for field in self.field_features:
            if field.type == FieldType.TEXT:
                count += 1
        return count

    @cached_property
    def field_count(self) -> int:
        return len(self.field_features)

    @cached_property
    def numeric_ratio(self) -> float:
        count = 0
        for field in self.field_features:
            if field.type == FieldType.NUMERIC:
                count += 1

        if count == 0:
            return 0

        return count / self.field_count
