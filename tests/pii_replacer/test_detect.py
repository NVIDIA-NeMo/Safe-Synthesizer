# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

# Skip all tests in this module if torch is not available
pytest.importorskip("torch", reason="torch is required for these tests (install with: uv sync --extra cpu)")

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from pandas.testing import assert_index_equal
from pydantic import ValidationError

from nemo_safe_synthesizer.pii_replacer.data_editor.detect import (
    DEFAULT_ENTITIES,
    ClassifyConfig,
    ColumnClassifierLLM,
    EntityExtractorGliner,
    merge_subsume,
    redact_from_entities,
    sample_columns,
)
from nemo_safe_synthesizer.pii_replacer.data_editor.environment import redact_entities_fn
from nemo_safe_synthesizer.pii_replacer.ner.ner import NERPrediction


def test_gliner_batch_predict_config():
    # Test batch_update_cache is short-circuited iff batch mode disabled.
    cfg = ClassifyConfig(
        valid_entities=["name"],
        ner_threshold=0.8,
        ner_regexps_enabled=False,
        ner_entities=None,
        gliner_enabled=True,
        gliner_batch_mode_enabled=False,
        gliner_batch_mode_chunk_length=10,
        gliner_batch_mode_batch_size=20,
        gliner_model="nvidia/gliner-PII",
    )

    with patch("nemo_safe_synthesizer.pii_replacer.data_editor.detect.GLiNER", MagicMock()):
        entity_extractor = EntityExtractorGliner.get_entity_extractor(cfg)
        entity_extractor.batch_update_cache(["abc"], None)
        entity_extractor._model.batch_predict_entities.assert_not_called()

    cfg = ClassifyConfig(
        valid_entities=set(["name"]),
        ner_threshold=0.8,
        ner_regexps_enabled=False,
        ner_entities=None,
        gliner_enabled=True,
        gliner_batch_mode_enabled=True,
        gliner_batch_mode_chunk_length=10,
        gliner_batch_mode_batch_size=20,
        gliner_model="nvidia/gliner-PII",
    )

    with patch("nemo_safe_synthesizer.pii_replacer.data_editor.detect.GLiNER", MagicMock()):
        entity_extractor = EntityExtractorGliner.get_entity_extractor(cfg)
        entity_extractor.batch_update_cache(["abc"], None)
        entity_extractor._model.batch_predict_entities.assert_called()


def test_gliner_pii_detection_recall():
    # Tests GLiNER’s PII detection on a short text, ensuring it finds a reasonable number of entities without over- or under-detecting.

    GLINER_PII_TEST_MIN_ENTITIES = 4
    GLINER_PII_TEST_MAX_ENTITIES = 10
    entity_labels = {"first_name", "last_name", "ssn", "age", "date_time", "phone_number", "address", "city"}

    cfg = ClassifyConfig(
        valid_entities=entity_labels,
        ner_threshold=0.3,
        ner_regexps_enabled=False,
        ner_entities=None,
        gliner_enabled=True,
        gliner_batch_mode_enabled=False,
        gliner_batch_mode_chunk_length=512,
        gliner_batch_mode_batch_size=8,
        gliner_model="nvidia/gliner-PII",
    )
    extractor = EntityExtractorGliner.get_entity_extractor(cfg)

    # Short text with clear PII.
    text_with_pii = "Daniel Martinez, born September 3, 1988, age 38, visited the clinic for a follow-up regarding hypertension. His patient ID is PT30984. He lives at 912 Cedar Avenue, Vernon, CA 90058, and can be reached at 323-555-6724 for appointment reminders. During the visit, the physician reviewed his recent blood pressure readings and confirmed he has been taking his prescribed Lisinopril daily. A follow-up appointment was scheduled in two months to monitor his response to the treatment plan."

    all_predictions = []
    preds = extractor.extract_ner_predictions(text_with_pii, entity_labels)
    all_predictions.extend(preds)

    total = len(all_predictions)
    assert total >= GLINER_PII_TEST_MIN_ENTITIES, (
        f"Expected at least {GLINER_PII_TEST_MIN_ENTITIES} entities, got {total}. GLiNER may have missed expected PII."
    )
    assert total <= GLINER_PII_TEST_MAX_ENTITIES, (
        f"Expected at most {GLINER_PII_TEST_MAX_ENTITIES} entities, got {total}. "
        "GLiNER may be over-labeling text as PII."
    )


def test_column_sample_sizes():
    df = pd.DataFrame(
        {
            "ColA": [1, 2, 3, 4, 5],
            "ColB": [1, 2, np.nan, 4, np.nan],
            "EmptyCol": [np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    cols = sample_columns(df, 1)
    assert set(cols.keys()) == set(["ColA", "ColB"])
    cols = sample_columns(df, 2)
    assert set(cols.keys()) == set(["ColA", "ColB"])
    cols = sample_columns(df, 4)
    assert set(cols.keys()) == set(["ColA", "ColB"])


def test_column_sample_values():
    random_seed = 1
    df = pd.DataFrame(
        {
            "ColA": [1, 2, 3, 4, 5, 6, 7],
            "ColB": [1.0, 5.0, np.nan, 4.0, np.nan, 4.0, 5.0],
            "ColC": [1, 2, 3, 4, 4, 5, 5],
            "ColD": ["a", "b", np.nan, np.nan, "a", "c", "b"],
            "EmptyCol": [np.nan] * 7,
        }
    )
    expected = {
        "ColA": pd.Index(["7", "3", "2"]),
        "ColB": pd.Index(["4.0", "5.0", "1.0"]),
        "ColC": pd.Index(["5", "4", "3"]),
        "ColD": pd.Index(["a", "b", "c"]),
    }

    cols = sample_columns(df, 3, random_state=random_seed)
    assert cols.keys() == expected.keys()
    for name, expected_col in expected.items():
        assert_index_equal(expected_col, cols[name], check_names=False)


def test_column_sample_size_limit():
    random_seed = 1
    df = pd.DataFrame(
        {
            "ColA": [
                "This is a very long string, and in fact it is so long that it needs to be filtered from the column before samples are taken for column classification.",
                "Smaller string",
                "ok string",
            ],
            "ColB": [1, 2, 3],
        }
    )
    expected = {
        "ColA": pd.Index(["Smaller string", "ok string"]),
        "ColB": pd.Index(["1", "3", "2"]),
    }

    cols = sample_columns(df, 3, random_state=random_seed)
    assert cols.keys() == expected.keys()
    for name, expected_col in expected.items():
        assert_index_equal(expected_col, cols[name], check_names=False)


def test_column_empty_after_filtered():
    random_seed = 1
    df = pd.DataFrame(
        {
            "ColA": [
                "This is a very long string, and in fact it is so long that it needs to be filtered from the column before samples are taken for column classification.",
                "This is also a very long string, and in fact it is so long that it needs to be filtered from the column before samples are taken for column classification.",
                "This is also, also a very long string, and in fact it is so long that it needs to be filtered from the column before samples are taken for column classification.",
            ],
            "ColB": [1, 2, 3],
            "ColC": [np.nan, np.nan, np.nan],
            "ColD": [
                "This is a very long string, and in fact it is so long that it needs to be filtered from the column before samples are taken for column classification.",
                "This is also a very long string, and in fact it is so long that it needs to be filtered from the column before samples are taken for column classification.",
                np.nan,
            ],
        }
    )
    expected = {
        "ColB": pd.Index(["1", "3", "2"]),
    }

    cols = sample_columns(df, 3, random_state=random_seed)
    assert cols.keys() == expected.keys()
    for name, expected_col in expected.items():
        assert_index_equal(expected_col, cols[name], check_names=False)


def test_no_columns_after_filter():
    random_seed = 1
    df = pd.DataFrame(
        {
            "ColA": [
                "This is a very long string, and in fact it is so long that it needs to be filtered from the column before samples are taken for column classification.",
                "This is also a very long string, and in fact it is so long that it needs to be filtered from the column before samples are taken for column classification.",
                "This is also, also a very long string, and in fact it is so long that it needs to be filtered from the column before samples are taken for column classification.",
            ],
            "ColC": [np.nan, np.nan, np.nan],
            "ColD": [
                "This is a very long string, and in fact it is so long that it needs to be filtered from the column before samples are taken for column classification.",
                "This is also a very long string, and in fact it is so long that it needs to be filtered from the column before samples are taken for column classification.",
                np.nan,
            ],
        }
    )

    cols = sample_columns(df, 3, random_state=random_seed)
    assert not cols


def test_detect_types_llm_bad_json():
    llm_classifier = ColumnClassifierLLM()

    open_ai_mock = MagicMock()
    llm_classifier._llm = open_ai_mock

    df = pd.DataFrame(
        {
            "ColA": [
                "small string",
                "ok string",
                "fine string",
            ],
            "ColB": [1, 2, 3],
        }
    )

    def attach_mock_response(content: str):
        open_ai_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=content))]
        )

    # LLMs sometimes append extra commentary after the output, but the
    # `json_repair` package should be able to handle that just fine.
    attach_mock_response("""\
    {"ColA": "none", "ColB": "none"}

    Note: The "none" type is used for columns that do not fit into any of the specified categories.
    """)

    assert llm_classifier.detect_types(df, DEFAULT_ENTITIES) == {
        "ColA": "none",
        "ColB": "none",
    }

    # a list of dictionaries should properly take the 0th element

    attach_mock_response("""[{"ColA": "none", "ColB": "none"}]""")
    assert llm_classifier.detect_types(df, DEFAULT_ENTITIES) == {
        "ColA": "none",
        "ColB": "none",
    }

    # if the structure of the LLM response doesn't match expectations
    attach_mock_response("""[["pii", "pii"], "b", "c"]""")
    with pytest.raises(RuntimeError) as exc:
        llm_classifier.detect_types(df, DEFAULT_ENTITIES)
    assert "LLM failed" in exc.value.args[0]

    # ensure none of the (private) payload makes it into the error
    validation_err = exc.value.__context__
    assert isinstance(validation_err, ValidationError)
    assert "pii" not in str(exc.value.__context__)

    # json_repair will return '' when it can't parse the json at all, which
    # will cause us to fail via not-properly-structured json
    attach_mock_response("""\
    |"a", "b"}, "c")

    Note: The "none" type is used for columns that do not fit into any of the specified categories.
    """)
    with pytest.raises(RuntimeError) as exc:
        llm_classifier.detect_types(df, DEFAULT_ENTITIES)
    assert "LLM failed" in exc.value.args[0]


def test_redact_from_entities():
    entities = [
        NERPrediction("", start=3, end=5, label="name", source="test", score=9.0),
        NERPrediction("", start=10, end=15, label="address", source="test", score=9.0),
    ]
    text = "012345678901234566780123456789"
    redacted = redact_from_entities(text, entities, redact_entities_fn)
    assert redacted == "012<name>56789<address>566780123456789"


def test_redact_from_entities_boundaries():
    entities = [
        NERPrediction("", start=0, end=1, label="name", source="test", score=9.0),
        # GLiNER will set 'end' to len(text) for an entity
        # whose final char ends at the final char of the string.
        # This is an invalid index, but works correctly with slicing.
        NERPrediction("", start=29, end=30, label="address", source="test", score=9.0),
    ]
    text = "012345678901234566780123456789"
    redacted = redact_from_entities(text, entities, redact_entities_fn)
    assert redacted == "<name>1234567890123456678012345678<address>"


def test_redact_from_entities_overlap():
    """
    This shouldn't happen, but if it does, system shouldn't
    crash.
    """
    entities = [
        NERPrediction("", start=3, end=10, label="name", source="test", score=9.0),
        NERPrediction("", start=5, end=8, label="address", source="test", score=9.0),
    ]
    text = "012345678901234566780123456789"
    # Don't crash
    redact_from_entities(text, entities, redact_entities_fn)


def test_redact_from_entities_key_misordered():
    entities = [
        NERPrediction("", start=10, end=15, label="address", source="test", score=9.0),
        NERPrediction("", start=3, end=5, label="name", source="test", score=9.0),
    ]
    text = "012345678901234566780123456789"
    redacted = redact_from_entities(text, entities, redact_entities_fn)
    assert redacted == "012<name>56789<address>566780123456789"


def test_redact_from_entities_key_adjacent_entities():
    entities = [
        NERPrediction("", start=0, end=1, label="name", source="test", score=9.0),
        NERPrediction("", start=1, end=3, label="address", source="test", score=9.0),
        NERPrediction("", start=3, end=30, label="email", source="test", score=9.0),
    ]
    text = "012345678901234566780123456789"
    redacted = redact_from_entities(text, entities, redact_entities_fn)
    assert redacted == "<name><address><email>"


def test_redact_from_entities_key_almost_adjacent_entities():
    entities = [
        NERPrediction("", start=1, end=2, label="name", source="test", score=9.0),
        NERPrediction("", start=3, end=4, label="address", source="test", score=9.0),
        NERPrediction("", start=5, end=29, label="email", source="test", score=9.0),
    ]
    text = "012345678901234566780123456789"
    redacted = redact_from_entities(text, entities, redact_entities_fn)
    assert redacted == "0<name>2<address>4<email>9"


@pytest.mark.parametrize(
    "predictions,expected",
    [
        ([], []),
        ([(3, 1, "A")], []),
        ([(1, 3, "")], [(1, 3, "")]),
        ([(1, 3, "A"), (4, 5, "B")], [(1, 3, "A"), (4, 5, "B")]),
        ([(1, 10, "A"), (4, 5, "B")], [(1, 10, "A")]),
        ([(1, 10, "A"), (4, 22, "B")], [(1, 22, "B")]),
        ([(0, 1, "A"), (0, 1, "B")], [(0, 1, "A")]),
        ([(0, 1, "A"), (1, 2, "B")], [(0, 1, "A"), (1, 2, "B")]),
        (
            [
                (1, 3, "A"),
                (4, 10, "B"),
                (5, 9, "C"),
                (6, 7, "D"),
                (8, 11, "E"),
                (11, 12, "F"),
            ],
            [(1, 3, "A"), (4, 11, "B"), (11, 12, "F")],
        ),
        (
            [
                (1, 3, "A"),
                (4, 10, "B"),
                (5, 9, "C"),
                (6, 7, "D"),
                (8, 22, "E"),
                (11, 12, "F"),
            ],
            [(1, 3, "A"), (4, 22, "E")],
        ),
    ],
)
def test_merge_subsume(predictions: list[tuple], expected: list[tuple]):
    def to_ner_prediction(params: tuple):
        return NERPrediction("na", params[0], params[1], params[2], "na", "na")

    predictions = [to_ner_prediction(v) for v in predictions]
    expected = [to_ner_prediction(v) for v in expected]
    assert merge_subsume(predictions) == expected
