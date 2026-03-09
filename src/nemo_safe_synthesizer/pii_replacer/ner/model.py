# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""This module provides an interface to Nemo Safe Synthesizer Pii Replacer NER functionality."""

import re
from typing import Pattern

from ...data_processing.records.fragment import create_ner_api_response
from ...pii_replacer.ner.entity import Score
from ...pii_replacer.ner.ner import ner
from ...pii_replacer.ner.pipeline import pipeline
from ...pii_replacer.ner.regex import regex

InputData = str | dict | list[str] | list[dict]

INPUT_ERR = "Input data must be a string, dict, or a list of either"

_source_validator = re.compile(r"[A-Za-z_]{2,15}$")


def _parse_custom_source(source: str) -> tuple[str, str]:
    """Return a namespace, name str tuple"""
    parts = source.split("/")
    if len(parts) != 2:
        raise ValueError("source string must be in: foo/bar format")
    for part in parts:
        if not _source_validator.match(part):
            raise ValueError("parts of source strings must contain letters, underscores and be between 3 and 16 chars")

    return parts[0], parts[1]


class Model:
    """A representation of a singular NER model. This class combines
    several NER techniques into a simple interface
    """

    def __init__(self, *args, exclude: list[str] | None = None):
        if args and exclude:
            raise ValueError("Cannot include and exclude predictors")

        _pipeline = pipeline.from_source_string_list(include=args, exclude=exclude)
        self._ner = ner.NER(pipeline=_pipeline)

    @property
    def predictors(self) -> list[str]:
        return [pred.source for pred in self._ner.pipeline.predictors]

    def predict(self, input_data: InputData, *, timings_only=False) -> list[dict] | dict:
        if isinstance(input_data, (str, dict)):
            input_data = [input_data]

        if not isinstance(input_data, list):
            raise ValueError(INPUT_ERR)

        if not isinstance(input_data[0], (str, dict)):
            raise ValueError(INPUT_ERR)

        _target_type = type(input_data[0])

        for _target in input_data:
            if not isinstance(_target, _target_type):
                raise ValueError(INPUT_ERR)

        predictions = self._ner.predict(input_data, timings_only=timings_only, dict_result=True)

        if timings_only:
            return predictions.to_dict()
        if _target_type is str:
            return predictions

        return create_ner_api_response(input_data, predictions, pure_dict=True)

    def add_regex(self, source: str, pattern: str | Pattern, score: Score | None = None):
        namespace, name = _parse_custom_source(source)
        if score is None:
            score = Score.HIGH
        pattern = regex.Pattern(pattern=pattern, raw_score=score.value)
        predictor = regex.RegexPredictor(name=name, namespace=namespace, patterns=[pattern])
        self._ner.pipeline.add_predictors(predictor)


def create_empty() -> Model:
    return Model("__empty__")


def list_predictors() -> list[str]:
    return pipeline.all_built_in_predictor_sources()
