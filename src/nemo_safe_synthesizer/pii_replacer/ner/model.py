# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""This module provides an interface to Nemo Safe Synthesizer Pii Replacer NER functionality."""

from __future__ import annotations

import re
from collections.abc import Mapping
from numbers import Real
from typing import Any, Literal, TypeAlias, TypedDict, overload

from typing_extensions import TypeIs

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
NERInputRecord: TypeAlias = dict[str, Any]
NERInputRows: TypeAlias = list[str] | list[NERInputRecord]
RawPredictionRows: TypeAlias = list[list[object]]


def _is_string_rows(rows: object) -> TypeIs[list[str]]:
    return isinstance(rows, list) and bool(rows) and all(isinstance(row, str) for row in rows)


def _is_record_rows(rows: object) -> TypeIs[list[NERInputRecord]]:
    return isinstance(rows, list) and bool(rows) and all(isinstance(row, dict) for row in rows)


def _is_raw_prediction_rows(value: object) -> TypeIs[RawPredictionRows]:
    return isinstance(value, list) and all(isinstance(row, list) for row in value)


def _required_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"NER prediction field {field_name!r} must be a string")
    return value


def _required_int(value: object, field_name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"NER prediction field {field_name!r} must be an integer")
    return value


def _optional_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"NER prediction field {field_name!r} must be a float or None")
    return float(value)


def _required_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"NER timing field {field_name!r} must be a float")
    return float(value)


def _optional_str(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"NER prediction field {field_name!r} must be a string or None")
    return value


def _optional_value_path(value: object, field_name: str) -> tuple[str | int, ...] | list[str | int] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"NER prediction field {field_name!r} must be a string/integer path or None")
    path_parts: list[str | int] = []
    for part in value:
        if not isinstance(part, (str, int)):
            raise TypeError(f"NER prediction field {field_name!r} must be a string/integer path or None")
        path_parts.append(part)
    if isinstance(value, tuple):
        return tuple(path_parts)
    return path_parts


def _optional_bool(value: object, field_name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"NER prediction field {field_name!r} must be a boolean or None")
    return value


def _raw_prediction_payload(value: object) -> NERRawPredictionPayload:
    match value:
        case Mapping() as prediction:
            payload: NERRawPredictionPayload = {
                "text": _required_str(prediction.get("text"), "text"),
                "start": _required_int(prediction.get("start"), "start"),
                "end": _required_int(prediction.get("end"), "end"),
                "label": _required_str(prediction.get("label"), "label"),
                "source": _required_str(prediction.get("source"), "source"),
                "score": _optional_float(prediction.get("score"), "score"),
            }
            match prediction:
                case {"field": field}:
                    payload["field"] = _optional_str(field, "field")
            match prediction:
                case {"value_path": value_path}:
                    payload["value_path"] = _optional_value_path(value_path, "value_path")
            match prediction:
                case {"substring_match": substring_match}:
                    payload["substring_match"] = _optional_bool(substring_match, "substring_match")
            return payload
        case _:
            raise TypeError("NER prediction rows must contain dictionaries")


def _prediction_rows(value: object) -> NERPredictionRows:
    match value:
        case list() as rows if _is_raw_prediction_rows(rows):
            return [[_raw_prediction_payload(prediction) for prediction in row] for row in rows]
        case _:
            raise TypeError("NER predictions must be a list of prediction rows")


def _timings_payload_from_mapping(value: object) -> NERTimingsPayload:
    match value:
        case Mapping() as timing_data:
            match timing_data.get("predictors"):
                case Mapping() as predictors:
                    predictor_timings: dict[str, NERPredictorTimingPayload] = {}
                    for predictor, timing in predictors.items():
                        match predictor, timing:
                            case str() as predictor_name, Mapping() as predictor_timing:
                                predictor_timings[predictor_name] = {
                                    "total_time_ms": _required_float(
                                        predictor_timing.get("total_time_ms"), "total_time_ms"
                                    ),
                                    "total_time_ms_avg": _required_float(
                                        predictor_timing.get("total_time_ms_avg"), "total_time_ms_avg"
                                    ),
                                }
                            case _, Mapping():
                                raise TypeError("NER timing predictor names must be strings")
                            case str(), _:
                                raise TypeError("NER predictor timings must be dictionaries")
                            case _:
                                raise TypeError("NER timing predictor names must be strings")
                case _:
                    raise TypeError("NER timings field 'predictors' must be a dictionary")
        case _:
            raise TypeError("NER timings must be a dictionary")

    return {
        "records": _required_int(timing_data.get("records"), "records"),
        "total_predictions": _required_int(timing_data.get("total_predictions"), "total_predictions"),
        "total_time_ms": _required_float(timing_data.get("total_time_ms"), "total_time_ms"),
        "total_time_ms_avg": _required_float(timing_data.get("total_time_ms_avg"), "total_time_ms_avg"),
        "time_per_prediction_ms": _required_float(timing_data.get("time_per_prediction_ms"), "time_per_prediction_ms"),
        "predictors": predictor_timings,
    }


def _timings_payload(value: object) -> NERTimingsPayload:
    match getattr(value, "to_dict", None):
        case to_dict if callable(to_dict):
            return _timings_payload_from_mapping(to_dict())
        case _:
            raise TypeError("timings_only=True must return NER timings")


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
        active_pipeline = self._ner.pipeline
        if active_pipeline is None:
            raise RuntimeError("NER pipeline is not configured")
        return [pred.source for pred in active_pipeline.predictors]

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
        match input_data:
            case str() as text:
                input_rows: NERInputRows = [text]
            case dict() as record:
                input_rows = [record]
            case list() as rows if _is_string_rows(rows):
                input_rows = rows
            case list() as rows if _is_record_rows(rows):
                input_rows = rows
            case _:
                raise ValueError(INPUT_ERR)

        predictions = self._ner.predict(input_rows, timings_only=timings_only, dict_result=True)

        if timings_only:
            return _timings_payload(predictions)

        prediction_rows = _prediction_rows(predictions)
        match input_rows:
            case list() as rows if _is_string_rows(rows):
                return prediction_rows
            case list() as rows if _is_record_rows(rows):
                return create_ner_api_response(
                    rows,
                    prediction_rows,
                    pure_dict=True,
                )
            case _:
                raise ValueError(INPUT_ERR)

    def add_regex(self, source: str, pattern: str | re.Pattern, score: float | None = None):
        namespace, name = _parse_custom_source(source)
        if score is None:
            score = Score.HIGH
        regex_pattern = regex.Pattern(pattern=pattern, raw_score=score)
        predictor = regex.RegexPredictor(name=name, namespace=namespace, patterns=[regex_pattern])
        active_pipeline = self._ner.pipeline
        if active_pipeline is None:
            raise RuntimeError("NER pipeline is not configured")
        active_pipeline.add_predictors(predictor)


def create_empty() -> Model:
    return Model("__empty__")


def list_predictors() -> list[str]:
    return pipeline.all_built_in_predictor_sources()
