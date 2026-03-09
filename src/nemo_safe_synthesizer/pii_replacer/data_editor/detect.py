# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain, islice
from time import monotonic
from timeit import default_timer as timer
from typing import Callable, Iterable, Iterator, Optional

import json_repair
import pandas as pd
import torch
from gliner import GLiNER
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from pydantic import ConfigDict, TypeAdapter, ValidationError

from ...observability import get_logger
from ..ner import ner_mp
from ..ner.factory import LabelSetPredictorFilter, NERFactory
from ..ner.ner import NERPrediction
from ..ner.pipeline import Pipeline

logger = get_logger(__name__)


class DefaultLLMConfig:
    CONFIG_ID = os.environ.get("NIM_MODEL_ID", "qwen/qwen2.5-coder-32b-instruct")
    SYSTEM_PROMPT = "You are a helpful AI that annotates columns in datasets with their respective types. "
    MAX_OUTPUT_TOKENS = 2048
    TEMPERATURE = 0.2


DEFAULT_ENTITIES: set[str] = {
    "name",
    "first_name",
    "last_name",
    "company",
    "email",
    "phone_number",
    "address",
    "street_address",
    "city",
    "administrative_unit",
    "country",
    "postcode",
    "ssn",
    "credit_card_number",
    "iban",
    "text",
}

UNKNOWN_ENTITY: str = "none"

MAX_COL_STR_LEN = 128

TEMPLATE = """Valid types are: [
{valid_types_str}
]

Return the column type for each of the column names and example values in the question.

 Additional instructions:
* You only return the list of columns and types (e.g. column_name: column_type) in json format and no written explanations.
* The type should exactly match one of the valid types.
* You don't write other helpful information outside of the column names and types.
* The type "last_name" can be two names with a hyphen between them, for example 'Wagner-Gomez'.
* If example values contain only city names use the type "city"
* Use the type "street_address" if the example values only contain a street number and name.
* Use the type "first_name" when the example values contain only a first name.
* The column type "ssn" must include hyphens, for example "427-24-1865"
* The column types "date" and "date_time" must include a date in proper format, not just an integer or float
* Only use the column type "name" if the example values are referring to the name of a person. Names of non-persons get the column type "none".
* Use the column type "age" if the example values are referring to the age of a person.
* Use the column type "sexuality" if the example values are referring to sexual orientation. For example "heterosexual" or "gay"
* Use the column type "gender" if the example values are referring to the sex of a person.  For example "male" or "female"
* Use the column type "date_of_birth" if the example values are referring to birth dates.
* The type "vehicle_identifier" refers to an alphanumeric string that is exactly 17 characters long with no dashes.
* The type "license_plate" refers to an alphanumeric string, sometimes with dashes, that is not longer than 15 characters

Example:

Input:

prenom: João, Susan, Zoe
ciudad: London, Marseille, Nagoya

Output:

{{"prenom": "first_name",
"ciudad": "city"}}

 Input: {prompt_columns}

 Output:
"""


def _format_prompt(
    df: pd.DataFrame,
    entities: set[str],
    num_samples: Optional[int],
) -> Optional[str]:
    types = [
        "certificate_license_number",
        "first_name",
        "date_of_birth",
        "ssn",
        "medical_record_number",
        "password",
        "unique_identifier",
        "phone_number",
        "national_id",
        "swift_bic",
        "company_name",
        "country",
        "license_plate",
        "name",
        "tax_id",
        "employee_id",
        "pin",
        "state",
        "email",
        "date_time",
        "api_key",
        "biometric_identifier",
        "credit_debit_card",
        "coordinate",
        "device_identifier",
        "city",
        "postcode",
        "bank_routing_number",
        "vehicle_identifier",
        "health_plan_beneficiary_number",
        "url",
        "ipv4",
        "last_name",
        "time",
        "cvv",
        "customer_id",
        "date",
        "user_name",
        "street_address",
        "ipv6",
        "account_number",
        "address",
        "age",
        "fax_number",
        "county",
        "gender",
        "sexuality",
        "political_view",
        "race_ethnicity",
        "religious_belief",
        "language",
        "blood_type",
        "mac_address",
        "http_cookie",
        "employment_status",
        "education_level",
        "occupation",
        "region",
        "year",
        "department",
        "class",
        "minutes",
        "day",
        "type",
    ]

    notes = {
        "ssn": " (e.g. social security number)",
        "pin": " (e.g. personal identification number)",
        "cvv": " (e.g. card verification value)",
    }

    user_entities = entities - set(types)
    types.extend(user_entities)
    types.append(UNKNOWN_ENTITY)

    # Not actually valid json with notes, but it's what AS team has found to work.
    valid_types_str = "\n".join(f"{t}{notes.get(t, '')}," for t in types)

    prompt = PromptTemplate.from_template(TEMPLATE)
    column_samples = sample_columns(df, num_samples)
    if not column_samples:
        return None
    prompt_columns = "\n".join([f"{name}: {', '.join(values)}" for name, values in column_samples.items()])

    return prompt.format(
        prompt_columns=prompt_columns,
        valid_types_str=valid_types_str,
        entities=", ".join(entities),
        unknown_entity_type=UNKNOWN_ENTITY,
    )


def classify_columns(
    df: pd.DataFrame,
    entities: set[str],
    num_samples: Optional[int],
    client: Optional[OpenAI],
    on_validation_error: Callable[[], None],
    logger: logging.Logger,
) -> dict[str, Optional[str]]:
    formatted_prompt = _format_prompt(df, entities, num_samples)
    if not formatted_prompt:
        return {}

    llm_start = timer()
    response = client.chat.completions.create(
        model=DefaultLLMConfig.CONFIG_ID,
        messages=[
            {"role": "system", "content": DefaultLLMConfig.SYSTEM_PROMPT},
            {"role": "user", "content": formatted_prompt},
        ],
        temperature=DefaultLLMConfig.TEMPERATURE,
        max_tokens=DefaultLLMConfig.MAX_OUTPUT_TOKENS,
    )
    entities_str = response.choices[0].message.content
    llm_elapsed = timer() - llm_start
    logger.info(
        f"LLM column classification took {llm_elapsed} seconds.",
        extra={
            "ctx": {
                "llm_elapsed": llm_elapsed,
            },
        },
    )

    col_entities = _try_extract_entities(entities_str, on_validation_error)
    return {col: ent if ent in entities else UNKNOWN_ENTITY for col, ent in col_entities.items()}


def sample_columns(df: pd.DataFrame, num_samples: int, random_state: Optional[int] = None) -> dict[str, pd.Series]:
    nonempty_columns = df.dropna(axis="columns", how="all").columns
    col_samples = {}
    for col in nonempty_columns:
        filtered = df[col][df[col].apply(lambda x: len(str(x)) < MAX_COL_STR_LEN)].dropna()
        if filtered.empty:
            continue
        col_samples[col] = (
            filtered.sample(frac=1, random_state=random_state).value_counts().index[:num_samples].astype(str)
        )
    return col_samples


def _try_extract_entities(
    entities_str: str,
    on_validation_error: Callable[[], None],
) -> dict[str, Optional[str]]:
    entity_types = json_repair.loads(entities_str)

    # `json_repair` has a tendency to wrap singleton objects in a list during
    # fuzzy parsing; this will simply attempt to use the 0th element.
    if isinstance(entity_types, list) and len(entity_types):
        entity_types = entity_types[0]

    try:
        # A `TypeAdapter` lets you validate an object against a provided type,
        # obviating the need to create a bespoke pydantic object.
        ta = TypeAdapter(dict[str, Optional[str]], config=ConfigDict(hide_input_in_errors=True))
        return ta.validate_python(entity_types)
    except ValidationError:
        logger.exception("Error decoding classification JSON returned by llm")
        on_validation_error()


class ColumnClassifier(ABC):
    """
    Object used to classify column types. Derived implementation classes
    may hold various backends such as VertexAI, OpenAI, etc.
    """

    @abstractmethod
    def detect_types(self, df: pd.DataFrame, entities: Optional[set[str]]) -> dict[str, Optional[str]]:
        """
        Classify dataframe column types into given entities.
        Return dict mapping from column name to entity type.
        """
        ...


class ColumnClassifierNoop(ColumnClassifier):
    """NOOP column classifier"""

    def detect_types(self, df: pd.DataFrame, entities: Optional[set[str]] = None) -> dict[str, Optional[str]]:
        return {col: UNKNOWN_ENTITY for col in df.columns}


@dataclass
class IAPIClassifierConfig:
    """Fields required to communicate"""

    endpoint: str
    model_key: str
    job_id: str
    num_samples: int


class ColumnClassifierLLM(ColumnClassifier):
    """Classify column types using an LLM"""

    _llm: Optional[OpenAI]
    _num_samples: Optional[int]

    def __init__(self):
        self._llm = None
        self._num_samples = None

    def detect_types(self, df: pd.DataFrame, entities: set[str]) -> dict[str, Optional[str]]:
        """
        Detect column datatypes by sampling column data and asking
        InferenceAPI to classify it.
        """
        if self._llm is None:
            raise Exception("InferenceAPI classifier not initialized. Use get_classifier() method.")

        return classify_columns(
            df=df,
            entities=entities,
            num_samples=self._num_samples,
            client=self._llm,
            on_validation_error=self._on_validation_error,
            logger=logger,
        )

    def _on_validation_error(self) -> None:
        raise RuntimeError(
            "There was an error performing classification: "
            "the classifier LLM failed to return valid JSON. "
            "Please reach out to support if the error recurs."
        )


@dataclass
class ClassifyConfig:
    """Configuration options for column classification and NER"""

    valid_entities: set[str]
    ner_threshold: float
    ner_regexps_enabled: bool
    ner_entities: set[str] | None
    gliner_enabled: bool
    gliner_batch_mode_enabled: bool
    gliner_batch_mode_chunk_length: int
    gliner_batch_mode_batch_size: int
    gliner_model: str


class EntityExtractor(ABC):
    """Object used to extract entity/value pairs from free text."""

    column_report: NerReport
    current_column: str

    @abstractmethod
    def extract_entity_values(self, text: str, entities: Optional[set[str]]) -> list[dict[str, str]]:
        """Return a list of dicts which each contain an entity and its value."""
        ...

    @abstractmethod
    def extract_ner_predictions(self, text: str, entities: Optional[set[str]]) -> list[NERPrediction]:
        """
        Return a list of NERPrediction objects. Includes enough
        information to compose and dedup multiple EntityExtractors
        """
        ...

    @classmethod
    def get_entity_extractor(
        cls,
        clsfy_config: ClassifyConfig,
    ) -> EntityExtractor:
        return cls()

    def __init__(self):
        self.column_report = {}
        self.current_column = "unknown"

    def extract_and_replace_entities(self, redact_fn: RedactFn, text: str, entities: Optional[set[str]] = None) -> str:
        """
        Make NER predictions using subclass instantiations of above abstract methods,
        deduplicate / merge, update report, replace entities using the given
        redact function.
        """
        # Ensure text is a string - Jinja templates may pass non-string types (e.g., float/NaN)
        text = str(text)

        detected = merge_subsume(self.extract_ner_predictions(text, entities))
        if not detected:
            return text

        report = self.column_report.setdefault(self.current_column, {})
        for entity in detected:
            report.setdefault(entity.label, EntityReport(0, set()))
            report[entity.label].count += 1
            report[entity.label].values.add(entity.text)

        return redact_from_entities(text, detected, redact_fn)

    def batch_update_cache(self, texts: list[str], entities: Optional[set[str]] = None):
        pass


class EntityExtractorNoop(EntityExtractor):
    """NOOP extractor"""

    def extract_entity_values(self, text: str, entities: Optional[set[str]]) -> list[dict[str, str]]:
        return []

    def extract_ner_predictions(self, text: str, entities: Optional[set[str]]) -> list[NERPrediction]:
        return []


@dataclass
class EntityReport:
    count: int
    values: set()


NerReport = dict[str, dict[str, EntityReport]]
"""Dictionary of column name to dictionary of entity name to EntityReport."""


class EntityExtractorRegexp(EntityExtractor):
    """Extract entities using regular expressions."""

    _entity_types: set[str]

    def pipeline_from_entities(self, entities: set[str]) -> Callable[[], Pipeline]:
        if not entities:
            entities = self._entity_types
        predictor_filter = LabelSetPredictorFilter(entities)
        factory = NERFactory(regex_only=True)
        ner = factory.create(predictor_filter=predictor_filter)
        return ner.pipeline_factory

    def _detect_entities(self, text: str, entities: set[str]) -> list[dict]:
        # Ensure text is a string - Jinja templates may pass non-string types (e.g., float/NaN)
        text = str(text)
        pipeline_factory = self.pipeline_from_entities(entities)
        predictor = ner_mp.NERParallel(pipeline_factory=pipeline_factory, num_proc=2)
        results = predictor.predict({"text": text})
        return results

    def extract_entity_values(self, text: str, entities: Optional[set[str]] = None) -> list[dict[str, str]]:
        # _detect_entities already converts text to string
        detected = self._detect_entities(text, entities)
        return [{"entity": e.label, "value": e.text} for e in detected]

    def extract_ner_predictions(self, text: str, entities: Optional[set[str]] = None) -> list[NERPrediction]:
        # _detect_entities already converts text to string
        return self._detect_entities(text, entities)

    @classmethod
    def get_entity_extractor(
        cls,
        clsfy_cfg: ClassifyConfig,
    ) -> EntityExtractor:
        entity_types = DEFAULT_ENTITIES
        if clsfy_cfg.ner_entities:
            entity_types = clsfy_cfg.ner_entities
        self = cls()
        self._entity_types = entity_types
        return self


class EntityExtractorGliner(EntityExtractor):
    """Extract entities using GLiNER"""

    _entity_types: set[str]
    _model: Optional[GLiNER]
    _ner_threshold: float
    _chunk_length: int
    _chunk_overlap: int
    _batch_size: int
    _batch_mode_enabled: bool

    # Map (text sha hash, entity set) -> list of detected entities in GLiNER format
    _entity_cache: dict[tuple, list]

    @classmethod
    def get_entity_extractor(
        cls,
        clsfy_cfg: ClassifyConfig,
    ) -> EntityExtractorGliner:
        extractor = cls()
        extractor._model = None

        map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.debug(
            f"Loading NER model from filesystem to {map_location}",
        )

        extractor._model = GLiNER.from_pretrained(
            clsfy_cfg.gliner_model,
            map_location=map_location,
            local_files_only=os.environ.get("LOCAL_FILES_ONLY") in ["true", "True"],
        )
        entity_types = DEFAULT_ENTITIES
        if clsfy_cfg.ner_entities:
            entity_types = clsfy_cfg.ner_entities
        extractor._entity_types = entity_types
        extractor._ner_threshold = 0.3
        if clsfy_cfg.ner_threshold is not None:
            extractor._ner_threshold = clsfy_cfg.ner_threshold
        extractor._batch_mode_enabled = clsfy_cfg.gliner_batch_mode_enabled
        extractor._chunk_length = clsfy_cfg.gliner_batch_mode_chunk_length
        extractor._chunk_overlap = 128
        if extractor._chunk_length <= extractor._chunk_overlap:
            extractor._chunk_overlap = 0
        extractor._entity_cache = {}
        extractor._batch_size = clsfy_cfg.gliner_batch_mode_batch_size
        return extractor

    def _detect_entities_chunked(
        self,
        text: str,
        entity_labels: Optional[set[str]],
    ) -> list[dict]:
        """
        Detect entities from free text using a GLiNER model, chunking text to avoid overloading model context.

        Returns a list of dicts in GLiNER return value format, including entity name, value, start/end indices.
        """
        last_log = monotonic()
        if entity_labels is None:
            entity_labels = self._entity_types

        if entity_labels is None:
            return []
        entities_key = tuple(sorted(entity_labels))
        start = 0
        entities = []
        # Occasionally text interpreted as type other than string by jinja
        text = str(text)
        nchunks = 0
        n_cache_miss = 0
        while start < len(text):
            if monotonic() - last_log > 30:
                pct_complete = float(len(text) - start) / float(len(text)) * 100.0
                logger.info(f"""Current NER cell {pct_complete:.1f}% complete.""")
                last_log = monotonic()
            chunk = text[start : start + self._chunk_length]
            nchunks += 1
            temp_entities = self._entity_cache.get((hash(chunk), entities_key))
            if temp_entities is None:
                n_cache_miss += 1
                temp_entities = self._model.predict_entities(
                    chunk,
                    entity_labels,
                    threshold=self._ner_threshold,
                    flat_ner=False,
                )
            for idx in range(len(temp_entities)):
                temp_entities[idx]["start"] += start
                temp_entities[idx]["end"] += start
            entities.extend(temp_entities)
            start += self._chunk_length - self._chunk_overlap

        if monotonic() - last_log > 30:
            logger.info(
                f"""NER {nchunks - n_cache_miss} of {nchunks} chunks cached.""",
                extra={
                    "ctx": {"nchunks": nchunks, "misses": n_cache_miss},
                },
            )
            last_log = monotonic()
        entities_to_delete = []
        for idx, ent in enumerate(entities):
            has_superset = any(
                [
                    i != idx and i not in entities_to_delete and e["start"] <= ent["start"] and e["end"] >= ent["end"]
                    for i, e in enumerate(entities)
                ]
            )
            if has_superset:
                entities_to_delete.append(idx)
        for idx in sorted(entities_to_delete, reverse=True):
            del entities[idx]

        report = self.column_report.setdefault(self.current_column, {})
        for entity in entities:
            report.setdefault(entity["label"], EntityReport(0, set()))
            report[entity["label"]].count += 1
            report[entity["label"]].values.add(entity["text"])

        return entities

    def extract_entity_values(self, text: str, entities: Optional[set[str]] = None) -> list[dict[str, str]]:
        detected = self._detect_entities_chunked(text, entities)
        return [{"entity": e["label"], "value": e["text"]} for e in detected]

    def extract_ner_predictions(self, text: str, entities: Optional[set[str]]) -> list[NERPrediction]:
        return [
            NERPrediction(e["text"], e["start"], e["end"], e["label"], "GLiNER", 9.0)
            for e in self._detect_entities_chunked(text, entities)
        ]

    def batch_update_cache(self, texts: list[str], entity_labels: Optional[set[str]] = None):
        if not self._batch_mode_enabled:
            return

        logger.info(
            "",
            extra={
                "ctx": {
                    "render_table": True,
                    "tabular_data": {"chunk_length": self._chunk_length, "batch_size": self._batch_size},
                    "title": "GLiNER batch preprocessing",
                }
            },
        )

        last_log = monotonic()
        chunks = []
        if entity_labels is None:
            entity_labels = self._entity_types
        if entity_labels is None:
            return
        entities_key = tuple(sorted(entity_labels))
        for text in texts:
            text = str(text)
            start = 0
            while start < len(text):
                chunks.append(text[start : start + self._chunk_length])
                start += self._chunk_length - self._chunk_overlap
        batch_n_total = int(len(chunks) / self._batch_size) + 1
        for batch_n, batch in enumerate(
            [chunks[t : t + self._batch_size] for t in range(0, len(chunks), self._batch_size)]
        ):
            entities_lists = self._model.batch_predict_entities(
                batch,
                entity_labels,
                threshold=self._ner_threshold,
                flat_ner=False,
            )
            if monotonic() - last_log > 30:
                logger.info(
                    f"NER batch #{batch_n + 1} of {batch_n_total} complete.",
                    extra={
                        "ctx": {
                            "batch_number": batch_n + 1,
                            "batch_total": batch_n_total,
                        },
                    },
                )
                last_log = monotonic()
            for idx, entities in enumerate(entities_lists):
                self._entity_cache[(hash(batch[idx]), entities_key)] = entities


class EntityExtractorMulti(EntityExtractor):
    """EntityExtractor which encapsulates multiple extractors."""

    extractors: list[EntityExtractor]

    def extract_entity_values(self, text: str, entities: Optional[set[str]] = None) -> list[dict[str, str]]:
        """Return a list of dicts which each contain an entity and its value."""
        retval = []
        for extractor in self.extractors:
            retval += extractor.extract_entity_values(text, entities)
        return retval

    def extract_ner_predictions(self, text: str, entities: Optional[set[str]] = None) -> list[NERPrediction]:
        """Get entities across all providers and merge"""
        predictions = []
        for extractor in self.extractors:
            predictions += extractor.extract_ner_predictions(text, entities)
        return predictions

    @classmethod
    def get_entity_extractor(cls, clsfy_cfg: ClassifyConfig) -> EntityExtractorMulti:
        self = cls()
        self.extractors = []
        return self

    def batch_update_cache(self, texts, entities: Optional[set[str]] = None):
        for extractor in self.extractors:
            extractor.batch_update_cache(texts, entities)

    def add_entity_extractor(self, extractor: EntityExtractor):
        """Add an extractor to the list."""
        self.extractors.append(extractor)


RedactFn = Callable[[NERPrediction], str]


def redact_from_entities(text: str, detected: list[NERPrediction], redact_fn: RedactFn) -> str:
    return "".join(chain(*traverse_redact(text, detected, redact_fn)))


def traverse_redact(text: str, entities: list[NERPrediction], redact_fn: RedactFn) -> Iterator[Iterable[str]]:
    """
    Take a text string and a list of entities with start/end indicies
    within the string and yield iterators to replace the entities in
    the string with their labels via chain()
    """
    prev = 0
    for entity in sorted(entities, key=lambda e: e.start):
        # Yes this is an iterator which yields iterators. It allows a
        # single pass, and single copy, across text, which may be very large.
        yield islice(text, prev, entity.start)
        yield redact_fn(entity)
        prev = entity.end
    yield islice(text, prev, None)


def find_best(entities: list[NERPrediction]) -> NERPrediction:
    """
    Given a list of ner predictions, determine which one is "best".
    For now, just pick the one with the largest span.
    """
    span_max = 0
    best = entities[0]
    for entity in entities:
        span = entity.end - entity.start
        if span > span_max:
            best = entity
            span_max = span
    return best


def merge_subsume(entities: list[NERPrediction]) -> list[NERPrediction]:
    """
    Find overlaping ner prediction spans and replace with "best" prediction,
    as determined by find_best()
    """
    result = []
    entities = sorted(entities, key=lambda e: (e.start, e.end))
    while entities:
        entity = entities.pop(0)
        if entity.end <= entity.start:
            continue
        if not entities:
            result.append(entity)
            break

        peek = entities[0]
        if entity.end <= peek.start:
            # No overlap
            result.append(entity)
            continue

        # Note - sorting prevents:
        # entity.start > peek.start
        # entity.end > peek.end when starts are equal

        # subsume greedily
        start = entity.start
        end = max(entity.end, peek.end)
        candidates = [entity]

        while entities and peek.start < end:
            candidates.append(entities.pop(0))
            end = max(end, peek.end)
            if entities:
                peek = entities[0]
        best = find_best(candidates)
        result.append(NERPrediction(best.text, start, end, best.label, best.source, best.score))
    return result
