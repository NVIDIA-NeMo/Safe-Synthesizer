# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class ColumnStatistics(BaseModel):
    """Metadata and statistics for transformations and detected entities in a column.

    Tracks assigned type and entity from classification, NER-detected counts and
    values per entity, and which transform functions were applied.
    """

    assigned_type: str | None = Field(
        description="Type assigned to the column, usually from column classification.",
    )
    assigned_entity: str | None = Field(
        description="Entity assigned to the column, usually from column classification.",
    )
    detected_entity_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Entity name to count of times NER detected it in the column.",
    )
    detected_entity_values: dict[str, set] = Field(
        default_factory=dict,
        description="Entity name to set of detected values.",
    )
    is_transformed: bool = Field(
        default=False,
        description="Whether the column was transformed.",
    )
    transform_functions: set[str] = Field(
        default_factory=set,
        description="Names of transform functions applied to the column.",
    )


class TransformResult(BaseModel):
    """Result of PII replacement: transformed data and per-column statistics.

    Produced by ``NemoPII.transform_df`` and exposed as ``NemoPII.result``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transformed_df: pd.DataFrame = Field(
        description="DataFrame with PII replaced according to config.",
    )
    column_statistics: dict[str, ColumnStatistics] = Field(
        description="Column name to ``ColumnStatistics`` for that column.",
    )
