# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from ...pii_replacer.ner.entity import Score
from ...pii_replacer.ner.predictor import NERPrediction


class MetadataError(Exception):
    pass


SCORE_HIGH = "score_high"
SCORE_MED = "score_med"
SCORE_LOW = "score_low"
E2F = "fields_by_entity"


@dataclass
class Metadata:
    """
    Represents record metadata
    fields have an internal structure of
        field_name:
            fragment_name:
                metadata_type: [meta_data]
    """

    gretel_id: str
    fields: dict
    entities: dict
    received_at: str

    def as_dict(self):
        return self.__dict__


@dataclass
class MetadataFragment:
    gretel_id: str
    gretel_fragment_ts: str
    gretel_fragment_epoch: float
    fragment_name: str

    def __post_init__(self):
        self.fields = defaultdict(lambda: defaultdict(list))

    @property
    def gretel_fragment_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.gretel_fragment_epoch)

    def add_field_data(self, field_name: str, metadata_type: str, field_data: dict | list):
        """
        Adds field data by type to the model_metadata fragment
        Args:
            field_name - the name of the field to add model_metadata for
            metadata_type - the type of model_metadata the data represents
            field_data - object or list of objects to associate with
                the model_metadata fragment.
        """
        if isinstance(field_data, list):
            self.fields[field_name][metadata_type].extend(field_data)
        elif isinstance(field_data, dict):
            self.fields[field_name][metadata_type].append(field_data)
        else:
            raise TypeError("field_data must be a dict or list, got ", type(field_data))

    def as_dict(self):
        return self.__dict__


def merge_fragments(*fragments, ts: str | None = None) -> Metadata:
    """
    Reduces a list of fragments to a single `Metadata` object.
    Args:
        *fragments: a list of MetadataFragments to merge
    Returns:
        a single Metadata object with all fragments merged
    Raises:
        MetadataError if all input fragments don't correspond to the
            same id.
    """
    if len(set([fragment.gretel_id for fragment in fragments])) != 1:
        raise MetadataError("cannot merge fragments from different gretel records")
    else:
        gretel_id = fragments[0].gretel_id

    # todo(dn): there might be a better way to build up this object
    merged_fragment = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ts = ts or min([f.gretel_fragment_datetime for f in fragments]).isoformat() + "Z"
    fragment: MetadataFragment
    for fragment in fragments:
        for field_name, field_data in fragment.fields.items():
            for meta_type, meta_data in field_data.items():
                merged_fragment[field_name][fragment.fragment_name][meta_type].extend(meta_data)
    return Metadata(gretel_id=gretel_id, fields=merged_fragment, received_at=ts, entities={})


def fragment_for_record(gretel_id: str, fragment_name: str) -> MetadataFragment:
    epoch = time.time()
    ts = datetime.fromtimestamp(epoch).isoformat() + "Z"
    return MetadataFragment(
        gretel_id=gretel_id,
        gretel_fragment_epoch=epoch,
        gretel_fragment_ts=ts,
        fragment_name=fragment_name,
    )


def predictions_to_dict(
    predictions: list[NERPrediction],
    *,
    high_score: float = Score.HIGH,
    med_score: float = Score.MED,
) -> tuple[dict, dict]:
    """
    Reduces a list of prediction results into a single dict by field key. Also
    creates an entity mapping to easily find entity data about a record.
    Schema for entity map::
        {
            "score_high": ["ip_address", "..."],
            "score_med": [],
            "score_low": [],
            "fields_by_entity: {
                "ip_address": ["conn_str"]
            }
        }
    Returns:
        A tuple with a dict of field prediction results by field key and a mapping
        entities to their fields and a list of all entities seen in the record.
    """
    entity_map = {
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
    gretel_id: str,
) -> tuple[MetadataFragment, dict]:
    epoch = time.time()
    fragment = MetadataFragment(
        gretel_id=gretel_id,
        gretel_fragment_ts=datetime.fromtimestamp(epoch).isoformat() + "Z",
        gretel_fragment_epoch=epoch,
        fragment_name=fragment_name,
    )
    preds_by_field, ent_map = predictions_to_dict(predictions)
    for _field, preds in preds_by_field.items():
        fragment.add_field_data(_field, "labels", preds)

    return fragment, ent_map


def build_ner_metadata(preds: list[dict]) -> Metadata:
    preds = [NERPrediction.from_dict(p) for p in preds]
    fragment, ent_map = fragment_from_ner_predictions(
        "ner",
        preds,
        uuid.uuid4().hex,
    )
    meta = merge_fragments(fragment)
    meta.entities = ent_map
    return meta.as_dict()


def create_ner_api_response(records: list[dict], predictions: list[dict], pure_dict: bool = False) -> list[dict]:
    out = [
        {"data": record, "model_metadata": build_ner_metadata(prediction)}
        for record, prediction in zip(records, predictions)
    ]
    if pure_dict:
        return json.loads(json.dumps(out))
    return out
