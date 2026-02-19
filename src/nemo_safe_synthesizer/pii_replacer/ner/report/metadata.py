# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic.v1 import Field

from ..metadata import (
    DatasetMetadata,
    EntityMetadata,
    FieldMetadata,
)
from .base import ReportBaseModel


def convert_to_report(metadata: DatasetMetadata) -> DatasetMetadataReport:
    """
    Converts internal model_metadata object to the report.
    """
    fields_report = [_convert_field(field) for field in metadata.data.fields]
    entity_summary_report = [EntitySummaryReport(**e.dict()) for e in metadata.data.entities]

    return DatasetMetadataReport(
        record_count=metadata.project_record_count,
        fields=fields_report,
        entities=entity_summary_report,
    )


def _convert_field(field: FieldMetadata) -> FieldMetadataReport:
    entities = [_convert_entity(entity) for entity in field.entities]

    return FieldMetadataReport(
        name=field.field,
        count=field.count,
        approx_distinct_count=field.approx_cardinality,
        missing_count=field.missing,
        labels=field.field_labels,
        attributes=field.field_attributes,
        entities=entities,
        types=[TypeReport(type=tm.type, count=tm.count) for tm in field.types],
    )


class EntityReport(ReportBaseModel):
    label: str
    count: int
    approx_distinct_count: int
    f_ratio: float

    sources: list[str]
    """A list of unique sources that contributed predictions
    to the entity summary.
    """


class TypeReport(ReportBaseModel):
    """
    Information about scalar types of values within a given field.
    See :func:`common.records.base.get_type_as_string`
    """

    type: str
    """
    Name of the type.
    """

    count: int
    """
    Number of times value of that type has appeared.
    """


class FieldMetadataReport(ReportBaseModel):
    name: str
    """
    Name of the field.
    For data formats that support nesting, it's a dot delimited list of parents.
    E.g. "user.firstName"
    """

    count: int
    """
    Number of records that had value for this field.
    """

    approx_distinct_count: int
    """
    Approximate number of distinct values for this field.
    """

    missing_count: int
    """
    Number of records that didnt' have value for this field.
    """

    labels: list[str]
    """
    Field-level labels for this field.
    """

    attributes: list[str]
    """
    Attributes for this field.
    """

    entities: list[EntityReport] = Field(default_factory=list)
    """
    Granular information about entities detected in field's values.
    """

    types: list[TypeReport] = Field(default_factory=list)
    """
    Scalar types of values in this field.
    """


class EntitySummaryReport(ReportBaseModel):
    """Aggregate entity metrics by label."""

    label: str
    """Name of the entity or label."""

    fields: list[str]
    """List of fields the entity was seen in."""

    count: int
    """Total number of entity occurrences in the dataset
    by score.
    """

    approx_distinct_count: int
    """Approximate number of unique entity values in
    the dataset.
    """

    sources: list[str]
    """A list of unique sources that contributed predictions
    to the entity summary.
    """


def _convert_entity(entity: EntityMetadata) -> EntityReport:
    return EntityReport(
        label=entity.label,
        count=entity.count,
        approx_distinct_count=entity.approx_cardinality,
        f_ratio=entity.f_ratio,
        sources=entity.sources,
    )


class DatasetMetadataReport(ReportBaseModel):
    """
    Represents report with model_metadata about the dataset.
    """

    record_count: int
    """
    Number of records that were used to calculate model_metadata.
    """

    fields: list[FieldMetadataReport] = Field(default_factory=list)
    """
    Report about each field in the dataset.
    """

    entities: list[EntitySummaryReport] = Field(default_factory=list)
    """Aggregate entity metrics by score by label."""
