# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from numbers import Number
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Type, Union

import nemo_safe_synthesizer.pii_replacer.ner.person_name as person_name
from nemo_safe_synthesizer.observability import get_logger
from nemo_safe_synthesizer.pii_replacer.ner.custom import get_predictors_from_yaml
from nemo_safe_synthesizer.pii_replacer.ner.datetime import BirthDateTime, DateTime
from nemo_safe_synthesizer.pii_replacer.ner.ner import NER
from nemo_safe_synthesizer.pii_replacer.ner.predictor import Predictor
from nemo_safe_synthesizer.pii_replacer.ner.regex import Pattern, RegexPredictor
from nemo_safe_synthesizer.pii_replacer.ner.regexes import rules

logger = get_logger(__name__)


CUSTOM_CONFIG = "config.yml"
"""User defined custom regex predictors and patterns"""


class PredictionSource(Enum):
    """This enum stores default source tags for NLP and other more
    complex predictors that have associated "models" that need
    to be downloaded or are not automatically loaded based
    on a sub-package structure like regexes.
    """

    PERSON_NAME = person_name.PersonNamePredictor
    DATETIME = DateTime
    BIRTH_DATE = BirthDateTime

    @property
    def name(self):
        return f"{Predictor.default_namespace}/{self.value.default_name}"  # pylint: disable=no-member

    @property
    def cls(self):
        return self.value


class Pipeline:
    predictors: List[Predictor]
    load_timings: Dict[str, Number]

    def __init__(self, predictors: List[Predictor] = None):
        """A lightweight container class for managing prediction pipelines."""
        self.predictors = predictors or []
        self.load_timings = {}

    def _next_udf_name(self):
        return f"user_defined_predictor_{len(self.predictors) + 1}"

    def add_predictors(self, predictors: Union[Predictor, List[Predictor]]):
        if isinstance(predictors, Predictor):
            self.predictors.append(predictors)
        if isinstance(predictors, list):
            self.predictors.extend(predictors)
        return self

    def add_pattern(self, pattern: Pattern, name: str = None):
        name = name or self._next_udf_name()
        predictor = RegexPredictor.from_pattern(pattern, name=name)
        self.add_predictors(predictor)

    def add_regex(self, regex: str, name: str = None, namespace: str = None):
        name = name or self._next_udf_name()
        predictor = RegexPredictor.from_regex(regex, name=name)
        self.add_predictors(predictor)

    def merge(self, other_pipeline: Pipeline) -> Pipeline:
        self.add_predictors(list(other_pipeline.iter_predictors()))
        return self

    def iter_predictors(self) -> Iterator[Predictor]:
        return iter(self.predictors)

    def get_predictor(self, source: str) -> Predictor:
        """Returns the first predictor by source name in the pipeline

        Args:
            source: source search token

        Returns:
            the first found predictor
        """
        try:
            return next(p for p in self.predictors if p.source == source)
        except StopIteration:
            return None

    def add_predictors_from_yaml(self, file_path: str = CUSTOM_CONFIG):
        """Look for a custom config file with regex predictor data and
        load them into the pipeline
        """
        _path = Path(file_path)
        if not _path.is_file():
            logger.info("Custom Predictors: Not Found, skipping")
            return
        logger.info("Custom Predictors: loading from %s", file_path)
        self.add_predictors(get_predictors_from_yaml(_path))

    @classmethod
    def from_class_refs(cls, predictors: Sequence[Type[Predictor]]) -> Pipeline:
        klasses = [p() for p in predictors]
        return cls(klasses)

    @property
    def predictor_list(self) -> List[str]:
        """Return the list of all namespaced predictors
        currently loaded on the pipeline
        """
        return [p.source for p in self.predictors]


def regex_pipeline() -> Pipeline:
    """Returns a pipeline with regex predictors"""
    return Pipeline.from_class_refs(rules)


def default_pipeline() -> Pipeline:
    """Returns a pipeline with  the following predictors:
    - All regexes
    - DateTime
    - BirthDateTime
    - PersonName
    - Locations (ER + FT)
    """
    pipe = regex_pipeline().add_predictors(
        [
            DateTime(),
            BirthDateTime(),
            person_name.PersonNamePredictor(),
        ]
    )
    return pipe


fast_pipeline = default_pipeline


def create_default_ner(full: bool = False) -> NER:
    """Helper function that creates a NER
    instance already configured with the default pipeline
    """
    pipe = default_pipeline()
    return NER(pipeline=pipe)


def from_source_string_list(*, include: List[str] = None, exclude: List[str] = None) -> Pipeline:
    if include and exclude:
        raise ValueError("cannot include and exclude")

    pipeline = Pipeline()
    for source in PredictionSource:
        if include:
            if source.name in include:
                pipeline.add_predictors(source.cls())
            continue

        if exclude and source.name in exclude:
            continue

        pipeline.add_predictors(source.cls())

    # create as separate temp pipeline that hold all regexes, then
    # extract only the regexes we are keeping based on the include list
    _regex_pipeline = regex_pipeline()
    for regex_pred in _regex_pipeline.predictors:
        if include:
            if regex_pred.source in include:
                pipeline.add_predictors(regex_pred)
            continue

        if exclude and regex_pred.source in exclude:
            continue

        pipeline.add_predictors(regex_pred)
    return pipeline


def all_built_in_predictor_sources():
    tmp = [source.name for source in PredictionSource]
    _regex_pipeline = regex_pipeline()
    regex_sources = [pred.source for pred in _regex_pipeline.predictors]
    return tmp + regex_sources
