# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from pydantic import BaseModel, ConfigDict


class ColumnStatistics(BaseModel):
    """Metadata and statistics for transformations and detected entities in a column."""

    assigned_type: str | None
    """Type assigned to the column, usually from column classification."""
    assigned_entity: str | None
    """Entity assigned to the column, usually from column classification."""
    detected_entity_counts: dict[str, int]
    """Dictionary of entity to count of how many times NER detected it in the column."""
    detected_entity_values: dict[str, set]
    """Dictionary of entity to set of detected values."""
    is_transformed: bool
    """Whether the column was transformed."""
    transform_functions: set[str]
    """Functions that were used to transform the column."""


class TransformResult(BaseModel):
    """Result object for data and statistics after applying pii replacement."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transformed_df: pd.DataFrame
    """Transformed dataframe with PII replaced."""

    column_statistics: dict[str, ColumnStatistics]
    """Dictionary of column name to statistics for that column."""
