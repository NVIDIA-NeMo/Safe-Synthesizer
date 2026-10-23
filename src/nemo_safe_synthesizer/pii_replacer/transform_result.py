# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..config.replace_pii import EntityType, PiiReplacementPlan, ReplacePiiConfig
from ..errors import InternalError

__all__ = [
    "ColumnStatistics",
    "FreeTextReplacementRecord",
    "ReplacementGenerationStatistics",
    "ReplacementMap",
    "StructuredReplacementRecord",
    "TransformResult",
]


class StructuredReplacementRecord(BaseModel):
    """One structured replacement occurrence captured for explicit evaluation use.

    The original and replacement values can contain sensitive data. They are
    excluded from ``repr`` but remain available to callers and serialization.
    """

    model_config = ConfigDict(frozen=True)

    row_position: int = Field(ge=0, description="Zero-based position of the replaced dataframe row.")
    column_name: str = Field(description="Name of the structured column that was replaced.")
    entity_type: EntityType = Field(description="Normalized entity type used to generate the replacement.")
    scope: Literal["record", "group"] = Field(description="Mapping scope used for replacement reuse.")
    original_value: str = Field(repr=False, description="Canonical original value. May contain sensitive data.")
    replacement_value: str = Field(repr=False, description="Generated replacement value.")


class FreeTextReplacementRecord(BaseModel):
    """One accepted free-text span and its applied replacement.

    Offsets are half-open and refer to the complete original cell. The original
    and replacement values can contain sensitive data. They are excluded from
    ``repr`` but remain available to callers and serialization.
    """

    model_config = ConfigDict(frozen=True)

    row_position: int = Field(ge=0, description="Zero-based position of the replaced dataframe row.")
    column_name: str = Field(description="Name of the free-text column containing the accepted span.")
    start: int = Field(ge=0, description="Inclusive start offset in the original cell.")
    end: int = Field(gt=0, description="Exclusive end offset in the original cell. Must be greater than start.")
    entity_type: EntityType = Field(description="Normalized entity type assigned to the accepted span.")
    detection_source: Literal["gliner", "regex"] = Field(description="Detector that produced the accepted span.")
    score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
        description="Detector confidence in [0, 1] when the detector supplies one.",
    )
    scope: Literal["record", "group"] = Field(description="Mapping scope used for replacement reuse.")
    original_value: str = Field(repr=False, description="Exact text covered by the span. May contain sensitive data.")
    replacement_value: str = Field(repr=False, description="Replacement inserted for the accepted span.")

    @model_validator(mode="after")
    def _require_nonempty_span(self) -> Self:
        if self.end <= self.start:
            raise InternalError(f"free-text replacement record must satisfy start < end, got {self.start}, {self.end}")
        return self


class ReplacementMap(BaseModel):
    """Sensitive, opt-in replacement provenance for evaluation and auditing.

    Structured records identify each replaced cell. Free-text records are the
    exact accepted-span trace, including original offsets and detector
    provenance. Serializing this model persists original PII, so normal
    replacement calls do not create it unless explicitly requested.
    """

    model_config = ConfigDict(frozen=True)

    structured: tuple[StructuredReplacementRecord, ...] = Field(
        default_factory=tuple,
        description="Structured replacement occurrences in execution order.",
    )
    free_text: tuple[FreeTextReplacementRecord, ...] = Field(
        default_factory=tuple,
        description="Accepted free-text spans in stable column and row order.",
    )


class ColumnStatistics(BaseModel):
    """Metadata and statistics for transformations and detected entities in a column.

    Tracks assigned type and entity, detected entity counts and values, and
    which transform functions were applied.
    """

    assigned_type: str | None = Field(
        description="Type assigned to the column.",
    )
    assigned_entity: str | None = Field(
        description="Entity assigned to the column.",
    )
    detected_entity_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Entity name to count of times it was detected in the column.",
    )
    detected_entity_values: dict[str, set] = Field(
        default_factory=dict,
        exclude=True,
        repr=False,
        description=(
            "Entity name to set of detected values. These are original PII values, kept in memory for the PII "
            "replay metric and excluded from serialization and repr so they cannot be written out."
        ),
    )
    is_transformed: bool = Field(
        default=False,
        description="Whether the column was transformed.",
    )
    transform_functions: set[str] = Field(
        default_factory=set,
        description="Names of transform functions applied to the column.",
    )
    dependency_drift_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of later rows in a group that reused the group's first replacement although a column it "
            "depends on had a different value in that row."
        ),
    )
    repeated_replacement_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of replacements that repeat an earlier one in a column whose original values are all "
            "distinct, because no unused value was found."
        ),
    )
    kept_name_part_count: int = Field(
        default=0,
        ge=0,
        description="Number of full-name replacements that keep the original's first or last name.",
    )


class ReplacementGenerationStatistics(BaseModel):
    """Aggregate statistics for the synthetic replacement generation phase."""

    generated_replacement_count: int = Field(
        ge=0,
        description="Number of distinct replacement values generated after cache reuse.",
    )
    generator_time_seconds: float = Field(
        ge=0,
        description=(
            "Wall-clock time spent inside replacement generator calls, in seconds. Excludes plan resolution, "
            "free-text detection, and writing replacements back."
        ),
    )
    dependency_drift_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of later group occurrences that reused a replacement although their dependency values "
            "differed from the group's first occurrence."
        ),
    )


class TransformResult(BaseModel):
    """Result of PII replacement: transformed data and per-column statistics.

    Shared result shape consumed by evaluation after a replacement run. The
    result keeps its own deep copy of ``resolved_config``, and
    ``replacement_plan`` is read from that copy, so later changes to the
    caller's configuration cannot make the result inconsistent.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    transformed_df: pd.DataFrame = Field(
        description="DataFrame with PII replaced according to config.",
    )
    column_statistics: dict[str, ColumnStatistics] = Field(
        description="Column name to ``ColumnStatistics`` for that column.",
    )
    resolved_config: ReplacePiiConfig = Field(
        description=(
            "Resolved PII configuration containing the executed plan and its dependency mappings. "
            "Stored as a deep copy of the value passed in."
        ),
    )
    generation_statistics: ReplacementGenerationStatistics = Field(
        description="Aggregate timing and count statistics for replacement generation.",
    )
    replacement_time_seconds: float = Field(
        ge=0,
        description=(
            "Wall-clock time for this PII replacement call, in seconds, including plan resolution, free-text "
            "detection, generation, and statistics."
        ),
    )
    replacement_map: ReplacementMap | None = Field(
        default=None,
        repr=False,
        description="Sensitive replacement provenance, present only when explicitly requested.",
    )

    @field_validator("resolved_config")
    @classmethod
    def _own_a_resolved_config_snapshot(cls, value: ReplacePiiConfig) -> ReplacePiiConfig:
        if value.inline_plan is None:
            raise InternalError("resolved_config must contain the executed replacement plan")
        return value.model_copy(deep=True)

    @property
    def replacement_plan(self) -> PiiReplacementPlan:
        """Resolved replacement plan executed for this result, read from ``resolved_config``."""
        plan = self.resolved_config.inline_plan
        if plan is None:
            raise InternalError("resolved_config must contain the executed replacement plan")
        return plan
