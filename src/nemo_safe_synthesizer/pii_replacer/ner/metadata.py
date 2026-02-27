# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass
from dataclasses import field as Field
from enum import StrEnum
from math import ceil
from typing import Optional

from ...data_processing.records.json_record import JSONRecord
from .ner import NER, PipelineResult
from .ner_mp import NERParallel


class FieldAttribute(StrEnum):
    ID = "id"
    CATEGORICAL = "categorical"


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


@dataclass(frozen=True)
class TypeMetadata:
    type: str
    """
    Type of the values in the dataset.
    See :func:`common.records.base.get_type_as_string` for list of types.
    """

    count: int
    """Number of times this type appeared in the values of a field."""


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

    def dict(self):
        return asdict(self)


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

    def dict(self) -> dict:
        return asdict(self)


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

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FieldLabelCondition:
    min_f_ratio: float = 0.8

    def is_met(self, entity: EntityMetadata) -> bool:
        return entity.field_label_f_ratio >= self.min_f_ratio

    def explain(self, label: str) -> str:
        return f"At least {self.min_f_ratio * 100}% of all records were labeled with {label}"


class MetadataService:
    """
    Service that provides functionality to label records and also track model_metadata across whole dataset.

    It uses NER for the labeling itself and tracks labels across fields.
    """

    def __init__(
        self,
        ner: NER | NERParallel,
        field_label_condition: FieldLabelCondition = None,
    ):
        self.ner = ner
        self.dataset_metadata_tracker = _DatasetMetadataTracker(field_label_condition=field_label_condition)  # noqa: F821

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
    ) -> PipelineResult:
        # potential improvements here
        # - if a field is already classified as something on a field level -> do we skip doing NER on that field?

        record_labels = self.ner.predict(
            records,
            dict_result=True,
            min_score=min_score,
            timings_only=timings_only,
            include_labels=include_labels,
        )

        if timings_only:
            return record_labels.to_dict()

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
