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
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ...pii_replacer.ner.entity import Score
from ...pii_replacer.ner.predictor import NERPrediction


class MetadataError(Exception):
    """Raised when metadata fragments cannot be merged (e.g., mismatched IDs)."""


SCORE_HIGH = "score_high"
SCORE_MED = "score_med"
SCORE_LOW = "score_low"
E2F = "fields_by_entity"


@dataclass
class Metadata:
    """Merged record metadata aggregated from one or more ``MetadataFragment`` objects.

    The ``fields`` dict has the structure::

        field_name -> fragment_name -> metadata_type -> [metadata_items]
    """

    record_id: str

    fields: dict
    """Nested dict of per-field, per-fragment metadata."""

    entities: dict
    """Entity map produced by ``predictions_to_dict``."""

    received_at: str
    """ISO-8601 timestamp of the earliest fragment."""

    def as_dict(self):
        """Serialize to a plain dictionary."""
        return self.__dict__


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

    def __post_init__(self):
        self.fields = defaultdict(lambda: defaultdict(list))

    @property
    def fragment_datetime(self) -> datetime:
        """Fragment creation time as a ``datetime`` object."""
        return datetime.fromtimestamp(self.fragment_epoch)

    def add_field_data(self, field_name: str, metadata_type: str, field_data: dict | list):
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
        elif isinstance(field_data, dict):
            self.fields[field_name][metadata_type].append(field_data)
        else:
            raise TypeError("field_data must be a dict or list, got ", type(field_data))

    def as_dict(self):
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

    # todo(dn): there might be a better way to build up this object
    merged_fragment = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ts = ts or min([f.fragment_datetime for f in fragments]).isoformat() + "Z"
    fragment: MetadataFragment
    for fragment in fragments:
        for field_name, field_data in fragment.fields.items():
            for meta_type, meta_data in field_data.items():
                merged_fragment[field_name][fragment.fragment_name][meta_type].extend(meta_data)
    return Metadata(record_id=record_id, fields=merged_fragment, received_at=ts, entities={})


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
) -> tuple[dict, dict]:
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
    entity_map: dict[str, Any] = {
        SCORE_HIGH: set(),
        SCORE_MED: set(),
        SCORE_LOW: set(),
        E2F: defaultdict(set),
    }
    predictions_by_key = defaultdict(list)
    for prediction in predictions:
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
            entity_map[SCORE_MED].add(prediction.label)
        elif prediction.score >= high_score:
            entity_map[SCORE_HIGH].add(prediction.label)
        elif prediction.score >= med_score:
            entity_map[SCORE_MED].add(prediction.label)
        else:
            entity_map[SCORE_LOW].add(prediction.label)
        entity_map[E2F][prediction.label].add(prediction.field)
    for _, preds in predictions_by_key.items():
        preds.sort(key=lambda p: p["start"])
    for level in (SCORE_HIGH, SCORE_MED, SCORE_LOW):
        entity_map[level] = list(entity_map[level])
    for entity, _set in entity_map[E2F].items():
        entity_map[E2F][entity] = list(_set)
    return predictions_by_key, entity_map


def fragment_from_ner_predictions(
    fragment_name: str,
    predictions: list[NERPrediction],
    record_id: str,
) -> tuple[MetadataFragment, dict]:
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


def build_ner_metadata(preds: list[dict]) -> Metadata:
    """Construct a ``Metadata`` object from raw prediction dicts."""
    ner_preds = [NERPrediction.from_dict(p) for p in preds]
    fragment, ent_map = fragment_from_ner_predictions(
        "ner",
        ner_preds,
        uuid.uuid4().hex,
    )
    meta = merge_fragments(fragment)
    meta.entities = ent_map
    return meta.as_dict()


def create_ner_api_response(records: list[dict], predictions: list[dict], pure_dict: bool = False) -> list[dict]:
    """Build an API-compatible list of ``{data, model_metadata}`` dicts.

    Args:
        records: Raw record dictionaries.
        predictions: Per-record NER prediction lists (parallel with ``records``).
        pure_dict: If True, round-trip through JSON to eliminate non-dict types.

    Returns:
        List of dicts, each containing ``data`` and ``model_metadata`` keys.
    """
    out = [
        {"data": record, "model_metadata": build_ner_metadata(prediction)}
        for record, prediction in zip(records, predictions)
    ]
    if pure_dict:
        return json.loads(json.dumps(out))
    return out
