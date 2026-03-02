# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A named entity detected in a column with its occurrence count.

    Attributes:
        name: Entity type label (e.g. ``"SSN"``, ``"email"``).
        count: Number of occurrences in the column.
    """

    name: str
    count: int


# TODO: write an adapter from nemo_safe_synthesizer.pii_replacer.transform_result.ColumnStatistics
# to EntityMetadata, or remove EntityMetadata and use ColumnStatistics directly in evaluation.


class EntityMetadata(BaseModel):
    """Per-column entity detection and transformation metadata for evaluation.

    Attributes:
        entities: Detected PII entities with occurrence counts, if any.
        transformed: Whether the column was transformed during PII replacement.
        transform_function: Name of the transform function applied, if any.
    """

    entities: list[Entity] | None = Field(default=None)
    transformed: bool = Field(default=False)
    transform_function: str | None = Field(default=None)
