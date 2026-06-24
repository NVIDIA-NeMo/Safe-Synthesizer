# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from nemo_safe_synthesizer.data_processing.records.fragment import NERRawPredictionPayload
from nemo_safe_synthesizer.data_processing.records.json_record import JSONRecord
from nemo_safe_synthesizer.pii_replacer.ner.model import Model
from nemo_safe_synthesizer.pii_replacer.ner.utils import input_to_json_records


def _prediction() -> NERRawPredictionPayload:
    return {
        "text": "alice@example.com",
        "start": 0,
        "end": 17,
        "label": "email",
        "source": "safe-synthesizer/email",
        "score": 0.99,
        "field": "contact",
        "value_path": ("contact",),
        "substring_match": None,
    }


class _Timings:
    def to_dict(self) -> dict[str, Any]:
        return {
            "records": 1,
            "total_predictions": 1,
            "total_time_ms": 2.0,
            "total_time_ms_avg": 2.0,
            "time_per_prediction_ms": 2.0,
            "predictors": {},
        }


class _NERStub:
    def __init__(self, result: list[list[NERRawPredictionPayload]]) -> None:
        self.result = result
        self.calls: list[tuple[object, dict[str, object]]] = []

    def predict(self, input_data: object, **kwargs: object) -> object:
        self.calls.append((input_data, kwargs))
        if kwargs["timings_only"]:
            return _Timings()
        return self.result


def _model_with_stub(result: list[list[NERRawPredictionPayload]]) -> tuple[Model, _NERStub]:
    model = Model.__new__(Model)
    stub = _NERStub(result)
    model._ner = stub
    return model, stub


def test_predict_dict_input_returns_api_response_list():
    model, stub = _model_with_stub([[_prediction()]])

    response = model.predict({"contact": "alice@example.com"})

    assert response[0]["data"] == {"contact": "alice@example.com"}
    assert response[0]["model_metadata"]["fields"]["contact"]["ner"]["labels"][0]["label"] == "email"
    assert stub.calls == [
        (
            [{"contact": "alice@example.com"}],
            {"timings_only": False, "dict_result": True},
        )
    ]


def test_predict_string_input_returns_prediction_rows():
    prediction = _prediction()
    model, stub = _model_with_stub([[prediction]])

    response = model.predict("alice@example.com")

    assert response == [[prediction]]
    assert stub.calls == [
        (
            ["alice@example.com"],
            {"timings_only": False, "dict_result": True},
        )
    ]


def test_predict_timings_only_returns_timing_payload_for_dict_input():
    model, _stub = _model_with_stub([[_prediction()]])

    response = model.predict({"contact": "alice@example.com"}, timings_only=True)

    assert response == {
        "records": 1,
        "total_predictions": 1,
        "total_time_ms": 2.0,
        "total_time_ms_avg": 2.0,
        "time_per_prediction_ms": 2.0,
        "predictors": {},
    }


def test_input_to_json_records_preserves_json_records_and_wraps_raw_records():
    existing = JSONRecord({"contact": "alice@example.com"})

    records = input_to_json_records([existing, {"name": "Alice"}, "raw text"])

    assert records[0] is existing
    assert [record.original for record in records[1:]] == [{"name": "Alice"}, "raw text"]


def test_input_to_json_records_rejects_unsupported_list_items():
    bad_input: Any = [1]

    with pytest.raises(TypeError, match="Input data not supported"):
        input_to_json_records(bad_input)
