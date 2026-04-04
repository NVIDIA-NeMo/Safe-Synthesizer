# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A named entity detected in a column with its occurrence count."""

    name: str
    """Entity type label (e.g. ``"SSN"``, ``"email"``)."""

    count: int
    """Number of occurrences in the column."""


# TODO: write an adapter from nemo_safe_synthesizer.pii_replacer.transform_result.ColumnStatistics
# to EntityMetadata, or remove EntityMetadata and use ColumnStatistics directly in evaluation.


class EntityMetadata(BaseModel):
    """Per-column entity detection and transformation metadata for evaluation."""

    entities: list[Entity] | None = Field(
        default=None, description="Detected PII entities with occurrence counts, if any."
    )
    transformed: bool = Field(default=False, description="Whether the column was transformed during PII replacement.")
    transform_function: str | None = Field(default=None, description="Name of the transform function applied, if any.")
