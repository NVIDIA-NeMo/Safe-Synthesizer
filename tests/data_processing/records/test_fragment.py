# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_safe_synthesizer.data_processing.records.fragment import (
    E2F,
    SCORE_HIGH,
    SCORE_LOW,
    SCORE_MED,
    NERRawPredictionPayload,
    build_ner_metadata,
    create_ner_api_response,
)


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


def test_build_ner_metadata_returns_payload_dict():
    metadata = build_ner_metadata([_prediction()])

    assert isinstance(metadata, dict)
    assert metadata["record_id"]
    assert metadata["received_at"].endswith("Z")
    assert metadata["fields"]["contact"]["ner"]["labels"] == [
        {
            "start": 0,
            "end": 17,
            "label": "email",
            "score": 0.99,
            "source": "safe-synthesizer/email",
            "text": "alice@example.com",
        }
    ]
    assert metadata["entities"][SCORE_HIGH] == ["email"]
    assert metadata["entities"][SCORE_MED] == []
    assert metadata["entities"][SCORE_LOW] == []
    assert metadata["entities"][E2F] == {"email": ["contact"]}


def test_create_ner_api_response_pure_dict_preserves_shape_and_normalizes_metadata():
    response = create_ner_api_response(
        [{"contact": "alice@example.com"}],
        [[_prediction()]],
        pure_dict=True,
    )

    assert response[0]["data"] == {"contact": "alice@example.com"}

    metadata = response[0]["model_metadata"]
    assert type(metadata["fields"]) is dict
    assert type(metadata["fields"]["contact"]) is dict
    assert type(metadata["fields"]["contact"]["ner"]) is dict
    assert type(metadata["entities"][E2F]) is dict
    assert metadata["fields"]["contact"]["ner"]["labels"][0]["label"] == "email"
