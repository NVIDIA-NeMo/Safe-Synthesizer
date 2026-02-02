# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str
    count: int


# TODO: write an adapter from nemo_safe_synthesizer.pii_replacer.transform_result.ColumnStatistics
# to EntityMetadata, or remove EntityMetadata and use ColumnStatistics directly in evaluation.


class EntityMetadata(BaseModel):
    """Info for evaluation and report generation on entities and transformations applied to a column"""

    entities: list[Entity] | None = Field(default=None)
    transformed: bool = Field(default=False)
    transform_function: str | None = Field(default=None)
