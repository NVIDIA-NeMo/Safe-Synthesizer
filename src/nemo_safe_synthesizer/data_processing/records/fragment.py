# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata fragment assembly for NER-annotated records.

Provides ``Metadata`` and ``MetadataFragment`` for aggregating per-field NER
predictions, along with helpers to merge fragments, build entity maps, and
produce API-compatible response dicts.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, NotRequired, TypeAlias, TypedDict

from ...pii_replacer.ner.entity import Score
from ...pii_replacer.ner.predictor import NERPrediction


class MetadataError(Exception):
    """Raised when metadata fragments cannot be merged (e.g., mismatched IDs)."""


SCORE_HIGH = "score_high"
SCORE_MED = "score_med"
SCORE_LOW = "score_low"
E2F = "fields_by_entity"


class NERRawPredictionPayload(TypedDict):
    """Raw ``NERPrediction.as_dict`` payload consumed by metadata helpers."""

    text: str
    start: int
    end: int
    label: str
    source: str
    score: float | None
    field: NotRequired[str | None]
    value_path: NotRequired[tuple[str | int, ...] | list[str | int] | None]
    substring_match: NotRequired[bool | None]


class NERFieldLabelPayload(TypedDict):
    """Per-field label entry in the NER model metadata payload."""

    start: int
    end: int
    label: str
    score: float | None
    source: str
    text: str


NERMetadataFieldsPayload: TypeAlias = dict[str, dict[str, dict[str, list[NERFieldLabelPayload]]]]


class NEREntityMapPayload(TypedDict):
    """Score-bucketed entity summary in the NER model metadata payload."""

    score_high: list[str]
    score_med: list[str]
    score_low: list[str]
    fields_by_entity: dict[str, list[str]]


class NERMetadataPayload(TypedDict):
    """API-facing NER metadata payload."""

    record_id: str
    fields: NERMetadataFieldsPayload
    entities: NEREntityMapPayload
    received_at: str


NERRecordPayload: TypeAlias = dict[str, Any]


class NERApiResponseRow(TypedDict):
    """One API response row for dict-based NER model predictions."""

    data: NERRecordPayload
    model_metadata: NERMetadataPayload


@dataclass
class Metadata:
    """Merged record metadata aggregated from one or more ``MetadataFragment`` objects.

    The ``fields`` dict has the structure::

        field_name -> fragment_name -> metadata_type -> [metadata_items]
    """

    record_id: str

    fields: NERMetadataFieldsPayload
    """Nested dict of per-field, per-fragment metadata."""

    entities: NEREntityMapPayload
    """Entity map produced by ``predictions_to_dict``."""

    received_at: str
    """ISO-8601 timestamp of the earliest fragment."""

    def as_dict(self) -> NERMetadataPayload:
        """Serialize to a plain dictionary."""
        return {
            "record_id": self.record_id,
            "fields": self.fields,
            "entities": self.entities,
            "received_at": self.received_at,
        }


@dataclass
class MetadataFragment:
    """A single annotation pass over a record (e.g., one NER model's output).

    Fragments are later merged via ``merge_fragments`` into a single
    ``Metadata`` object per record.

    Args:
        record_id: Unique identifier for the source record.
        fragment_ts: ISO-8601 timestamp string.
        fragment_epoch: Unix epoch of the fragment creation.
        fragment_name: Identifier for this annotation pass (e.g., ``"ner"``).
    """

    record_id: str
    fragment_ts: str
    fragment_epoch: float
    fragment_name: str
    fields: dict[str, dict[str, list[NERFieldLabelPayload]]] = field(init=False)

    def __post_init__(self):
        self.fields = defaultdict(lambda: defaultdict(list))

    @property
    def fragment_datetime(self) -> datetime:
        """Fragment creation time as a ``datetime`` object."""
        return datetime.fromtimestamp(self.fragment_epoch)

    def add_field_data(
        self,
        field_name: str,
        metadata_type: str,
        field_data: NERFieldLabelPayload | list[NERFieldLabelPayload],
    ) -> None:
        """Append metadata entries for a field.

        Args:
            field_name: Name of the field to annotate.
            metadata_type: Category of metadata (e.g., ``"labels"``).
            field_data: A dict (single entry) or list of entries to add.

        Raises:
            TypeError: If ``field_data`` is neither a dict nor a list.
        """
        if isinstance(field_data, list):
            self.fields[field_name][metadata_type].extend(field_data)
        else:
            self.fields[field_name][metadata_type].append(field_data)

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return self.__dict__


def merge_fragments(*fragments, ts: str | None = None) -> Metadata:
    """Merge one or more ``MetadataFragment`` objects into a single ``Metadata``.

    Args:
        *fragments: Fragments to merge. All must share the same ``gretel_id``.
        ts: Override timestamp for ``received_at``. Defaults to the earliest
            fragment timestamp.

    Returns:
        A single ``Metadata`` object with all fragment data merged.

    Raises:
        MetadataError: If the fragments have different ``gretel_id`` values.
    """
    if len(set([fragment.record_id for fragment in fragments])) != 1:
        raise MetadataError("cannot merge fragments from different records")
    else:
        record_id = fragments[0].record_id

    merged_fragment: dict[str, dict[str, dict[str, list[NERFieldLabelPayload]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    ts = ts or min([f.fragment_datetime for f in fragments]).isoformat() + "Z"
    fragment: MetadataFragment
    for fragment in fragments:
        for field_name, field_data in fragment.fields.items():
            for meta_type, meta_data in field_data.items():
                merged_fragment[field_name][fragment.fragment_name][meta_type].extend(meta_data)
    fields: NERMetadataFieldsPayload = {
        field_name: {
            fragment_name: {meta_type: list(meta_data) for meta_type, meta_data in fragment_data.items()}
            for fragment_name, fragment_data in field_data.items()
        }
        for field_name, field_data in merged_fragment.items()
    }
    empty_entities: NEREntityMapPayload = {
        SCORE_HIGH: [],
        SCORE_MED: [],
        SCORE_LOW: [],
        E2F: {},
    }
    return Metadata(record_id=record_id, fields=fields, received_at=ts, entities=empty_entities)


def fragment_for_record(record_id: str, fragment_name: str) -> MetadataFragment:
    """Create a new ``MetadataFragment`` timestamped to the current time."""
    epoch = time.time()
    ts = datetime.fromtimestamp(epoch).isoformat() + "Z"
    return MetadataFragment(
        record_id=record_id,
        fragment_epoch=epoch,
        fragment_ts=ts,
        fragment_name=fragment_name,
    )


def predictions_to_dict(
    predictions: list[NERPrediction],
    *,
    high_score: float = Score.HIGH,
    med_score: float = Score.MED,
) -> tuple[dict[str, list[NERFieldLabelPayload]], NEREntityMapPayload]:
    """Aggregate NER predictions into per-field results and an entity map.

    Groups predictions by field and builds a score-bucketed entity map::

        {
            "score_high": ["ip_address", ...],
            "score_med": [],
            "score_low": [],
            "fields_by_entity": {"ip_address": ["conn_str"]},
        }

    Args:
        predictions: List of NER prediction objects.
        high_score: Minimum score threshold for the ``score_high`` bucket.
        med_score: Minimum score threshold for the ``score_med`` bucket.

    Returns:
        A tuple of (predictions_by_field, entity_map).
    """
    high_entities: set[str] = set()
    medium_entities: set[str] = set()
    low_entities: set[str] = set()
    fields_by_entity: dict[str, set[str]] = defaultdict(set)
    predictions_by_key: dict[str, list[NERFieldLabelPayload]] = defaultdict(list)
    for prediction in predictions:
        if prediction.field is None:
            continue
        predictions_by_key[prediction.field].append(
            {
                "start": prediction.start,
                "end": prediction.end,
                "label": prediction.label,
                "score": prediction.score,
                "source": prediction.source,
                "text": prediction.text,
            }
        )
        # NOTE:(jm) this covers the Spacy case where
        # no score is emitted. Predictions here could be
        # hit or miss so we throw it into medium
        if prediction.score is None:
            medium_entities.add(prediction.label)
        elif prediction.score >= high_score:
            high_entities.add(prediction.label)
        elif prediction.score >= med_score:
            medium_entities.add(prediction.label)
        else:
            low_entities.add(prediction.label)
        fields_by_entity[prediction.label].add(prediction.field)
    for _, preds in predictions_by_key.items():
        preds.sort(key=lambda p: p["start"])
    entity_map: NEREntityMapPayload = {
        SCORE_HIGH: list(high_entities),
        SCORE_MED: list(medium_entities),
        SCORE_LOW: list(low_entities),
        E2F: {entity: list(fields) for entity, fields in fields_by_entity.items()},
    }
    return predictions_by_key, entity_map


def fragment_from_ner_predictions(
    fragment_name: str,
    predictions: list[NERPrediction],
    record_id: str,
) -> tuple[MetadataFragment, NEREntityMapPayload]:
    """Build a ``MetadataFragment`` and entity map from NER predictions.

    Args:
        fragment_name: Identifier for this annotation pass (e.g., ``"ner"``).
        predictions: List of NER predictions to aggregate.
        gretel_id: Unique identifier for the source record.

    Returns:
        A tuple of (fragment, entity_map).
    """
    epoch = time.time()
    fragment = MetadataFragment(
        record_id=record_id,
        fragment_ts=datetime.fromtimestamp(epoch).isoformat() + "Z",
        fragment_epoch=epoch,
        fragment_name=fragment_name,
    )
    preds_by_field, ent_map = predictions_to_dict(predictions)
    for _field, preds in preds_by_field.items():
        fragment.add_field_data(_field, "labels", preds)

    return fragment, ent_map


def build_ner_metadata(preds: list[NERRawPredictionPayload]) -> NERMetadataPayload:
    """Construct an API-facing metadata payload from raw prediction dicts."""
    ner_preds = [NERPrediction.from_dict(dict(p)) for p in preds]
    fragment, ent_map = fragment_from_ner_predictions(
        "ner",
        ner_preds,
        uuid.uuid4().hex,
    )
    meta = merge_fragments(fragment)
    meta.entities = ent_map
    return meta.as_dict()


def create_ner_api_response(
    records: list[NERRecordPayload],
    predictions: list[list[NERRawPredictionPayload]],
    pure_dict: bool = False,
) -> list[NERApiResponseRow]:
    """Build an API-compatible list of ``{data, model_metadata}`` dicts.

    Args:
        records: Raw record dictionaries.
        predictions: Per-record NER prediction lists (parallel with ``records``).
        pure_dict: If True, round-trip through JSON to eliminate non-dict types.

    Returns:
        List of dicts, each containing ``data`` and ``model_metadata`` keys.
    """
    out: list[NERApiResponseRow] = [
        {"data": record, "model_metadata": build_ner_metadata(prediction)}
        for record, prediction in zip(records, predictions)
    ]
    if pure_dict:
        data_rows = json.loads(json.dumps([row["data"] for row in out]))
        if not isinstance(data_rows, list):
            raise TypeError("expected JSON round-trip to preserve response row list")
        rows: list[NERApiResponseRow] = []
        for data, row in zip(data_rows, out):
            if not isinstance(data, dict):
                raise TypeError("expected JSON round-trip to preserve response data dictionaries")
            record: NERRecordPayload = {str(key): value for key, value in data.items()}
            response_row: NERApiResponseRow = {"data": record, "model_metadata": row["model_metadata"]}
            rows.append(response_row)
        return rows
    return out
