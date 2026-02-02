# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from nemo_safe_synthesizer.config.base import NSSBaseModel


class FieldAttribute(StrEnum):
    ID = "id"
    CATEGORICAL = "categorical"


class EntityMetadata(NSSBaseModel):
    label: str
    """Label of detected entity."""

    count: int
    """Number of times this entity was detected."""

    f_ratio: float
    """Equal to ``(number of values with this entity)/(total number of values for this field)``."""

    approx_cardinality: int
    """How many distinct values there were for this entity type."""

    sources: list[str]
    """A list of unique sources that contributed predictions
    to the entity summary.
    """

    field_label_f_ratio: float
    """The ratio of (column spanning entity matches)/(total number of field values).
    This field is used to determine if an entity should be applied
    as a field_label in transformation pipelines."""


class TypeMetadata(NSSBaseModel):
    type: str
    """
    Type of the values in the dataset.
    """

    count: int
    """Number of times this type appeared in the values of a field."""


class FieldMetadata(NSSBaseModel):
    field: str
    count: int
    """Number of times this field appeared in the dataset."""

    approx_cardinality: int
    """How many distinct values this field have in the dataset (approximate)."""

    missing: int
    """Number of records that didn't contain this field."""

    pct_missing: float
    """Percent of missing in the whole dataset [0-100]."""

    pct_total_unique: float
    """
    Percent of unique values in the whole dataset [0-100].
    This is equal to 100, when all values for this field are unique.
    """

    s_score: float
    """
    Sensitivity score [0-1].

    It's equal to:
    - 1.0, when all values are unique and there are no values missing.
    - moving toward 0.0 with missing values and/or many values that are repeated.

    The general idea was to quickly highlight columns you might want to pay attention to for special handling in
    either transforms or synthesizer, for one reason or another.
    """

    entities: list[EntityMetadata] = Field(default=list())
    """List of entities detected in values of this field."""

    types: list[TypeMetadata] = Field(default=list())
    """List of types detected in values of this field."""

    field_labels: list[str] = Field(default=list())
    """Labels detected for this field."""

    field_attributes: list[FieldAttribute] = Field(default=list())
    """Attributes detected for this field."""


class EntitySummary(NSSBaseModel):
    """Contains entity summary data that is unique by label name"""

    label: str
    """Name of the entity or label."""

    fields: list[str]
    """Fields containing the entity or label."""

    count: int
    """Total number of entities found in the dataset."""

    approx_distinct_count: int
    """Approximate total number of unique entity values
    found in the dataset. This value is collected
    using an HLL data structure.
    """

    sources: list[str]
    """A list of unique sources that contributed predictions
    to the entity summary.
    """


class FieldsMetadata(NSSBaseModel):
    fields: list[FieldMetadata] = Field(default=list())
    """
    List of fields in the dataset.
    Note: This list is ordered in the same order that original dataset was ordered.
    """

    entities: list[EntitySummary] = Field(default=list())
    """List of entities in the dataset. Unique by entity label and score."""


class DatasetMetadata(NSSBaseModel):
    project_record_count: int
    total_field_count: int

    # TODO: maybe we can simplify this later, previous structure was ["data"]["fields"], so preserving it here
    data: FieldsMetadata = Field(default=FieldsMetadata())

    def add_field(self, field_metadata: FieldMetadata):
        self.data.fields.append(field_metadata)

    def add_entity(self, entity_summary: EntitySummary):
        self.data.entities.append(entity_summary)


class FieldLabelCondition(NSSBaseModel):
    min_f_ratio: float = 0.8

    def is_met(self, entity: EntityMetadata) -> bool:
        return entity.field_label_f_ratio >= self.min_f_ratio

    def explain(self, label: str) -> str:
        return f"At least {self.min_f_ratio * 100}% of all records were labeled with {label}"
