# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data models for field type classification and per-column statistics.

Classes:

    FieldType: Enum of column types recognized by the field analyzer.
    FieldFeatures: Statistical profile of a single DataFrame column.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FieldType(StrEnum):
    """Column type classification assigned by the field analyzer.

    Used by ``evaluation`` and ``pii_replacer`` to dispatch type-specific
    processing logic (e.g., numeric metrics vs. text similarity).
    """

    EMPTY = "empty"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    OTHER = "other"
    BINARY = "binary"


class FieldFeatures(BaseModel):
    """Statistical profile of a single DataFrame column.

    Captures type classification, value distribution, missing-data rates,
    string-length statistics, and optional numeric precision. Produced by
    ``describe_field`` in the ``analyzers.field_features`` module.
    """

    name: str = Field(description="Column name in the source DataFrame.")
    type: FieldType = Field(description="Inferred column type.")

    count: int = Field(description="Number of non-null values.")

    unique_values_list: list[Any] = Field(description="Deduplicated list of non-null values.")

    unique_count: int = Field(description="Number of unique non-null values.")
    unique_percent: float = Field(description="Percentage of values that are unique, relative to non-null count.")

    missing_count: int = Field(description="Number of null/missing values.")
    missing_percent: float = Field(description="Percentage of values that are missing, relative to total count.")

    min_str_length: int = Field(description="Minimum string-representation length among non-null values.")
    max_str_length: int = Field(description="Maximum string-representation length among non-null values.")

    avg_str_length: float = Field(
        description="Mean string-representation length among non-null values.",
    )

    min_value: int | float | None = Field(
        default=None,
        description="Floor power-of-10 of the column minimum (numeric columns only).",
    )
    max_value: int | float | None = Field(
        default=None,
        description="Floor power-of-10 of the column maximum (numeric columns only).",
    )
    min_precision: int | None = Field(
        default=None,
        description="Minimum decimal digit count across float values.",
    )
    max_precision: int | None = Field(
        default=None,
        description="Maximum decimal digit count across float values.",
    )

    space_count: int | None = Field(
        default=None,
        description="Total number of space characters across all non-null values.",
    )

    classification: dict | None = Field(
        default=None,
        description="NER-based classification metadata, when available.",
    )

    def to_dict(self, **kwargs) -> dict:
        """Serialize to a dict, excluding unset and None fields."""
        return self.model_dump(exclude_unset=True, exclude_none=True)
