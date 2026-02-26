# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import string
from dataclasses import dataclass
from numbers import Number
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Optional

from ...data_processing.records.base import KVPair
from ...data_processing.records.json_record import JSONRecord
from ...data_processing.records.value_path import ValuePath
from .const import const
from .entity import Entity, Score
from .fasttext import FTEntityMatcher
from .models import (
    ModelManifest,
    ObjectRef,
    Visibility,
    get_cache_manager,
)
from .ner import NERPrediction
from .predictor import Predictor
from .utils import is_string_a_number

spacy = None
Doc = None
Span = None
srsly = None


if TYPE_CHECKING:
    pass
else:
    RegexPredictor = None


MODEL_REGISTRY = {
    "fasttext": FTEntityMatcher,
}


NLP_ENTITY_MAP = {
    "ADDRESS": Entity.LOCATION,
    "BOROUGH": Entity.LOCATION,
    "CITY": Entity.LOCATION,
    "CONTINENT": Entity.LOCATION,
    "COUNTY": Entity.LOCATION,
    "COUNTRY": Entity.LOCATION,
    "DISTRICT": Entity.LOCATION,
    "LOCATION": Entity.LOCATION,
    "MUNICIPALITY": Entity.LOCATION,
    "NATIONALITY": Entity.LOCATION,
    "NEIGHBORHOOD": Entity.LOCATION,
    "PROVINCE": Entity.LOCATION,
    "REGION": Entity.LOCATION,
    "STATE": Entity.LOCATION,
    "SUBURB": Entity.LOCATION,
    "TOWN": Entity.LOCATION,
    "PER": Entity.PERSON_NAME,
    "I-PER": Entity.PERSON_NAME,
    "PERSON": Entity.PERSON_NAME,
    "LOC": Entity.LOCATION,
    "I-LOC": Entity.LOCATION,
    "GPE": Entity.LOCATION,
}


# Valid characters and lengths for detected entities
ENTITY_VALID_CHARS = set(string.ascii_lowercase + string.ascii_uppercase + string.digits + "." + "-" + "/" + " ")
ENTITY_MAX_CHARS = 30
SPACY_DELIM = " is "


def _is_valid_spacy_entity(ent: Span):
    """Removes entities predicted by Spacy ML models that are not contained
    in ENTITY_VALID_CHARACTERS, or that are greater than ENTITY_MAX_CHARS
    in length

    Also remove any entities where the string composition of the
    entity entirely can represent a number
    """
    entity_text = ent.text
    if len(entity_text) > ENTITY_MAX_CHARS:
        return False

    if not set(entity_text) <= ENTITY_VALID_CHARS:
        return False

    if is_string_a_number(entity_text):
        return False

    # if the entity is composed of whitespace or null characters, ignore it.
    null_characters = ("na", "n/a", "n.a.", "n.a")
    if entity_text.isspace() or entity_text.lower() in null_characters:
        return False

    return True


@dataclass
class FieldStr:
    """String optimized field representation for NLP prediction pipelines."""

    field: str
    value_path: ValuePath
    offset: int
    text: str

    @classmethod
    def from_kv_pair(cls, pair: KVPair) -> FieldStr:
        """Returns a string optimized input for NLP predictions.

        For example give a k,v pair

            {"location": "united states"} this function will

        merge the pair into a string

            "location is united states"

        These merged strings produce better prediction results from our NLP pipeline.

        Args:
            pair: ``KVPair`` from a ``JSONRecord`` to merge.

        Returns:
            An instance of ``FieldStr``.
        """
        prefix = " ".join(pair.field_tokens) + SPACY_DELIM if pair.field else ""
        return cls(
            field=pair.field,
            value_path=pair.value_path,
            offset=len(prefix),
            text=prefix + str(pair.value),
        )

    def spacy_doc_to_ner_prediction(
        self, doc: Doc, source: str, validator: Optional[Callable] = None
    ) -> list[NERPrediction]:
        """Given a prediction document, return an NERPrediction.

        This function will apply a set of rules on a Spacy doc and extract predictions
        based on those rules. Certain predictions are filtered out based on score
        and entity type.

        This function is also responsible for reconstructing the input string into it's
        source KVPair. Since Spacy creates spans on texts of different lengths, we account
        for those lengths during reconstruction.

        Args:
            doc: The spacy doc to extract entities from
            source: the model used to create predictions.
        """
        preds = []
        for ent in doc.ents:
            # Don't create predictions for entities that were found inside the tokenized prefix.
            if ent.start_char - self.offset >= 0:
                if validator is None or validator(ent):
                    label = NLP_ENTITY_MAP.get(getattr(ent, "label_"))
                    start = ent.start_char - self.offset
                    end = ent.end_char - self.offset
                    substring_match = end - start + self.offset != len(doc.text)
                    if label:
                        pred = NERPrediction(
                            text=ent.text.strip(),
                            start=start,
                            end=end,
                            field=self.field,
                            value_path=self.value_path,
                            label=label.tag,
                            score=doc._.ent_score(label),
                            source=source,  # todo: get source from model instead of cls
                            substring_match=substring_match,
                        )
                        preds.append(pred)
        return preds


def _flatten_fields(in_data: list[KVPair]) -> list[FieldStr]:
    """Prepare input data for NLP usage.

    Args:
        in_data: tokenized input fields to flatten to string
    """
    return [
        FieldStr.from_kv_pair(field)
        for field in in_data
        if not isinstance(field.value, Number)  # issue #52, don't predict on numbers
    ]


entity_ruler_manifest = ModelManifest(
    model="entityruler",
    version="2",
    sources=[ObjectRef(key="entityruler", file_name="entityruler.pickle")],
    visibility=Visibility.INTERNAL,
)


def _get_spacy_ent_score(_, ent: Entity) -> float:
    """This function gets attached to a Spacy Doc as an extension
    to return the scores we need based on internal Entity labels.
    """
    if ent == Entity.LOCATION:
        return Score.MED
    elif ent == Entity.PERSON_NAME:
        return Score.LOW
    else:
        return None


class SpacyPredictor(Predictor):
    nlp: Any
    timings: dict[str, Number]
    default_name: str = "spacy"

    def __init__(
        self,
        name: str = None,
        model: str = None,
        namespace: Optional[str] = None,
    ):
        if spacy is None:
            raise RuntimeError("spacy is not installed and must be for spacy predictors")

        self.timings = {}

        if name is None and model is None:
            # We load our default Spacy model
            name = "spacy"
            start_time = perf_counter()
            model_bytes = get_cache_manager().resolve(spacy_manifest)["model_data"]  # noqa: F821
            nlp = spacy.blank("en")
            ner_pipe = nlp.create_pipe("ner")
            nlp.add_pipe(ner_pipe)
            nlp.from_bytes(model_bytes)
            self.timings[self.default_name] = perf_counter() - start_time
            self.nlp = nlp
        else:
            raise ValueError("Spacy predictor no longer works")

        super().__init__(name=name, namespace=namespace)

    @classmethod
    def from_manifest(cls, manifest: ModelManifest) -> "SpacyPredictor":
        start_time = perf_counter()
        model_bytes = get_cache_manager().resolve(manifest)["model_data"]
        nlp = spacy.blank("en")
        ner_pipe = nlp.create_pipe("ner")
        nlp.add_pipe(ner_pipe)
        nlp.from_bytes(model_bytes)
        pred_inst = cls(cls.default_name, nlp)
        pred_inst.timings[cls.default_name] = perf_counter() - start_time
        return pred_inst

    def _predict(self, input_text: str) -> Doc:
        doc = self.nlp.make_doc(input_text)
        doc.set_extension("ent_score", method=_get_spacy_ent_score, force=True)
        doc.set_extension(const.NER_SCORE, default=None, force=True)
        return self.nlp(doc.text)

    def evaluate(self, in_data: JSONRecord) -> list[NERPrediction]:
        fields = _flatten_fields(in_data.kv_pairs)

        pred_by_field = []

        for field in fields:
            prediction = self._predict(field.text)
            pred_by_field.append(field.spacy_doc_to_ner_prediction(prediction, self.source, _is_valid_spacy_entity))

        return [pred for field_preds in pred_by_field for pred in field_preds]
