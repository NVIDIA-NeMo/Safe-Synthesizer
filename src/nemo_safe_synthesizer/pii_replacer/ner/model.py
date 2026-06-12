# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""This module provides an interface to Nemo Safe Synthesizer Pii Replacer NER functionality."""

from __future__ import annotations

import re
from typing import Any, Literal, TypeAlias, TypedDict, cast, overload

from ...data_processing.records.fragment import (
    NERApiResponseRow,
    NERRawPredictionPayload,
    create_ner_api_response,
)
from ...pii_replacer.ner import ner, pipeline, regex
from ...pii_replacer.ner.entity import Score

INPUT_ERR = "Input data must be a string, dict, or a list of either"

_source_validator = re.compile(r"[A-Za-z_]{2,15}$")


class NERPredictorTimingPayload(TypedDict):
    total_time_ms: float
    total_time_ms_avg: float


class NERTimingsPayload(TypedDict):
    records: int
    total_predictions: int
    total_time_ms: float
    total_time_ms_avg: float
    time_per_prediction_ms: float
    predictors: dict[str, NERPredictorTimingPayload]


NERPredictionRows: TypeAlias = list[list[NERRawPredictionPayload]]
NERModelPredictionResponse: TypeAlias = NERPredictionRows | list[NERApiResponseRow]


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

    def __init__(self, *args: str, exclude: list[str] | None = None):
        if args and exclude:
            raise ValueError("Cannot include and exclude predictors")

        include = list(args) if args else None
        if include is not None:
            _pipeline = pipeline.from_source_string_list(include=include)
        elif exclude is not None:
            _pipeline = pipeline.from_source_string_list(exclude=exclude)
        else:
            _pipeline = pipeline.from_source_string_list()
        self._ner = ner.NER(pipeline=_pipeline)

    @property
    def predictors(self) -> list[str]:
        return [pred.source for pred in self._ner.pipeline.predictors]

    @overload
    def predict(
        self,
        input_data: str | dict[str, Any] | list[str] | list[dict[str, Any]],
        *,
        timings_only: Literal[True],
    ) -> NERTimingsPayload: ...

    @overload
    def predict(
        self,
        input_data: str,
        *,
        timings_only: Literal[False] = False,
    ) -> NERPredictionRows: ...

    @overload
    def predict(
        self,
        input_data: list[str],
        *,
        timings_only: Literal[False] = False,
    ) -> list[list[NERRawPredictionPayload]]: ...

    @overload
    def predict(
        self,
        input_data: dict[str, Any] | list[dict[str, Any]],
        *,
        timings_only: Literal[False] = False,
    ) -> list[NERApiResponseRow]: ...

    @overload
    def predict(
        self,
        input_data: str | dict[str, Any] | list[str] | list[dict[str, Any]],
        *,
        timings_only: bool = False,
    ) -> NERModelPredictionResponse | NERTimingsPayload: ...

    def predict(
        self,
        input_data: str | dict[str, Any] | list[str] | list[dict[str, Any]],
        *,
        timings_only: bool = False,
    ) -> NERModelPredictionResponse | NERTimingsPayload:
        if isinstance(input_data, str):
            input_rows: list[str] | list[dict[str, Any]] = [input_data]
        elif isinstance(input_data, dict):
            input_rows = [input_data]
        else:
            input_rows = input_data

        if not isinstance(input_rows, list):
            raise ValueError(INPUT_ERR)

        if not isinstance(input_rows[0], (str, dict)):
            raise ValueError(INPUT_ERR)

        _target_type = type(input_rows[0])

        for _target in input_rows:
            if not isinstance(_target, _target_type):
                raise ValueError(INPUT_ERR)

        predictions = self._ner.predict(input_rows, timings_only=timings_only, dict_result=True)

        if timings_only:
            return cast(NERTimingsPayload, cast(Any, predictions).to_dict())
        if _target_type is str:
            return cast(NERPredictionRows, predictions)

        return create_ner_api_response(
            cast(list[dict[str, Any]], input_rows),
            cast(list[list[NERRawPredictionPayload]], predictions),
            pure_dict=True,
        )

    def add_regex(self, source: str, pattern: str | re.Pattern, score: float | None = None):
        namespace, name = _parse_custom_source(source)
        if score is None:
            score = Score.HIGH
        regex_pattern = regex.Pattern(pattern=pattern, raw_score=score)
        predictor = regex.RegexPredictor(name=name, namespace=namespace, patterns=[regex_pattern])
        self._ner.pipeline.add_predictors(predictor)


def create_empty() -> Model:
    return Model("__empty__")


def list_predictors() -> list[str]:
    return pipeline.all_built_in_predictor_sources()
