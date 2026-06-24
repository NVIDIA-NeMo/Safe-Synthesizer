# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as Field
from enum import StrEnum
from math import ceil
from typing import Any, Optional, TypedDict

from ...data_processing.records.json_record import JSONRecord
from .ner import NER, PipelineResult, Timings
from .ner_mp import NERParallel


class FieldAttribute(StrEnum):
    ID = "id"
    CATEGORICAL = "categorical"


class EntityMetadataPayload(TypedDict):
    label: str
    count: int
    f_ratio: float
    approx_cardinality: int
    sources: list[str]
    field_label_f_ratio: float


class TypeMetadataPayload(TypedDict):
    type: str
    count: int


class FieldMetadataPayload(TypedDict):
    field: str
    count: int
    approx_cardinality: int
    missing: int
    pct_missing: float
    pct_total_unique: float
    s_score: float
    entities: list[EntityMetadataPayload]
    types: list[TypeMetadataPayload]
    field_labels: list[str]
    field_attributes: list[FieldAttribute]


class EntitySummaryPayload(TypedDict):
    label: str
    fields: list[str]
    count: int
    approx_distinct_count: int
    sources: list[str]


class FieldsMetadataPayload(TypedDict):
    fields: list[FieldMetadataPayload]
    entities: list[EntitySummaryPayload]


class DatasetMetadataPayload(TypedDict):
    project_record_count: int
    total_field_count: int
    data: FieldsMetadataPayload


@dataclass(frozen=True)
class EntityMetadata:
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

    def dict(self) -> EntityMetadataPayload:
        return {
            "label": self.label,
            "count": self.count,
            "f_ratio": self.f_ratio,
            "approx_cardinality": self.approx_cardinality,
            "sources": self.sources,
            "field_label_f_ratio": self.field_label_f_ratio,
        }


@dataclass(frozen=True)
class TypeMetadata:
    type: str
    """
    Type of the values in the dataset.
    See :func:`common.records.base.get_type_as_string` for list of types.
    """

    count: int
    """Number of times this type appeared in the values of a field."""

    def dict(self) -> TypeMetadataPayload:
        return {
            "type": self.type,
            "count": self.count,
        }


@dataclass(frozen=True)
class FieldMetadata:
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

    entities: list[EntityMetadata] = Field(default_factory=list)
    """List of entities detected in values of this field."""

    types: list[TypeMetadata] = Field(default_factory=list)
    """List of types detected in values of this field."""

    field_labels: list[str] = Field(default_factory=list)
    """Labels detected for this field."""

    field_attributes: list[FieldAttribute] = Field(default_factory=list)
    """Attributes detected for this field."""

    def dict(self) -> FieldMetadataPayload:
        return {
            "field": self.field,
            "count": self.count,
            "approx_cardinality": self.approx_cardinality,
            "missing": self.missing,
            "pct_missing": self.pct_missing,
            "pct_total_unique": self.pct_total_unique,
            "s_score": self.s_score,
            "entities": [entity.dict() for entity in self.entities],
            "types": [type_metadata.dict() for type_metadata in self.types],
            "field_labels": self.field_labels,
            "field_attributes": self.field_attributes,
        }


@dataclass(frozen=True)
class EntitySummary:
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
    using an HLL datastructure.
    """

    sources: list[str]
    """A list of unique sources that contributed predictions
    to the entity summary.
    """

    def dict(self) -> EntitySummaryPayload:
        return {
            "label": self.label,
            "fields": self.fields,
            "count": self.count,
            "approx_distinct_count": self.approx_distinct_count,
            "sources": self.sources,
        }


@dataclass(frozen=True)
class FieldsMetadata:
    fields: list[FieldMetadata] = Field(default_factory=list)
    """
    List of fields in the dataset.
    Note: This list is ordered in the same order that original dataset was ordered.
    """

    entities: list[EntitySummary] = Field(default_factory=list)
    """List of entities in the dataset. Unique by entity label and score."""


@dataclass(frozen=True)
class DatasetMetadata:
    project_record_count: int
    total_field_count: int

    # TODO: maybe we can simplify this later, previous structure was ["data"]["fields"], so preserving it here
    data: FieldsMetadata = Field(default_factory=FieldsMetadata)

    def add_field(self, field_metadata: FieldMetadata):
        self.data.fields.append(field_metadata)

    def add_entity(self, entity_summary: EntitySummary):
        self.data.entities.append(entity_summary)

    def to_dict(self) -> DatasetMetadataPayload:
        return {
            "project_record_count": self.project_record_count,
            "total_field_count": self.total_field_count,
            "data": {
                "fields": [field_metadata.dict() for field_metadata in self.data.fields],
                "entities": [entity_summary.dict() for entity_summary in self.data.entities],
            },
        }


@dataclass(frozen=True)
class FieldLabelCondition:
    min_f_ratio: float = 0.8

    def is_met(self, entity: EntityMetadata) -> bool:
        return entity.field_label_f_ratio >= self.min_f_ratio

    def explain(self, label: str) -> str:
        return f"At least {self.min_f_ratio * 100}% of all records were labeled with {label}"


class _DatasetMetadataTracker:
    def __init__(self, field_label_condition: FieldLabelCondition | None = None):
        self.field_label_condition = field_label_condition or FieldLabelCondition()
        self._field_names: list[str] = []
        self._record_count = 0

    def add_field_names(self, field_names: list[str]) -> None:
        for field_name in field_names:
            if field_name not in self._field_names:
                self._field_names.append(field_name)

    def update_fields(self, records: list[JSONRecord]) -> None:
        self._record_count += len(records)

    def update_entities(self, records: list[JSONRecord], record_labels: PipelineResult) -> None:
        return None

    def get_snapshot(self) -> DatasetMetadata:
        fields = [
            FieldMetadata(
                field=field_name,
                count=0,
                approx_cardinality=0,
                missing=self._record_count,
                pct_missing=100.0 if self._record_count else 0.0,
                pct_total_unique=0.0,
                s_score=0.0,
            )
            for field_name in self._field_names
        ]
        return DatasetMetadata(
            project_record_count=self._record_count,
            total_field_count=len(self._field_names),
            data=FieldsMetadata(fields=fields),
        )

    def get_entity_detail(self, entity_label: str) -> dict[str, Any]:
        return {}


class MetadataService:
    """
    Service that provides functionality to label records and also track model_metadata across whole dataset.

    It uses NER for the labeling itself and tracks labels across fields.
    """

    def __init__(
        self,
        ner: NER | NERParallel,
        field_label_condition: FieldLabelCondition | None = None,
    ):
        self.ner = ner
        self.dataset_metadata_tracker = _DatasetMetadataTracker(field_label_condition=field_label_condition)

    def add_field_names(self, field_names: list[str]):
        """
        Adds names of all fields that should be tracked.
        This is necessary to track fields that can be present in the dataset,
        but have no values.
        For example for a CSV file, where there is a header "my_field", but the whole
        column is empty, we still want to report model_metadata on that field.

        Args:
            field_names: Names of the fields to be initialized. These names should be in the
                same order as they appear in the dataset.
        """
        self.dataset_metadata_tracker.add_field_names(field_names)

    def predict(
        self,
        records: list[JSONRecord],
        min_score: float = 0.0,
        timings_only: bool = False,
        include_labels: Optional[set[str]] = None,
    ) -> PipelineResult | dict[str, Any]:
        # potential improvements here
        # - if a field is already classified as something on a field level -> do we skip doing NER on that field?

        if timings_only:
            timings = self.ner.predict(
                records,
                dict_result=True,
                min_score=min_score,
                timings_only=True,
                include_labels=include_labels,
            )
            if not isinstance(timings, Timings):
                raise RuntimeError("NER timings result was not returned")
            return timings.to_dict()

        record_labels = self.ner.predict(
            records,
            dict_result=True,
            min_score=min_score,
            timings_only=False,
            include_labels=include_labels,
        )
        if isinstance(record_labels, Timings):
            raise RuntimeError("NER predictions were not returned")

        # Update model_metadata based on records that were classified
        self.dataset_metadata_tracker.update_fields(records)

        # Update model_metadata based on labels that were detected
        self.dataset_metadata_tracker.update_entities(records, record_labels)

        return record_labels

    def get_metadata(self) -> DatasetMetadata:
        """Returns dataset model_metadata based on records that were labeled to this point."""
        return self.dataset_metadata_tracker.get_snapshot()

    def get_entity_detail(self, entity_label: str) -> dict:
        return self.dataset_metadata_tracker.get_entity_detail(entity_label)


def _trim(v: float) -> float:
    return ceil(v * 100) / 100
