# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import random
import re

import pandas as pd
import pytest

from nemo_safe_synthesizer.evaluation.components.pii_replay import (
    PIIReplay,
    _EmbeddedValueIndex,
    _synthetic_replay_counts,
)
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets
from nemo_safe_synthesizer.pii_replacer.transform_result import ColumnStatistics

logger = logging.getLogger(__name__)


@pytest.mark.slow
def test_pii_replay(fixture_training_df_5k, fixture_synthetic_df_5k, fixture_test_df, fixture_column_statistics):
    """PII analysis on 5k rows - computationally expensive."""
    evaluation_datasets = EvaluationDatasets.from_dataframes(
        fixture_training_df_5k, fixture_synthetic_df_5k, fixture_test_df, fixture_column_statistics
    )
    pii_replay = PIIReplay.from_evaluation_datasets(evaluation_datasets)

    assert len(pii_replay.pii_replay_data) == 3

    assert pii_replay.pii_replay_data[0].column_name == "small_cat"
    assert pii_replay.pii_replay_data[0].pii_type == "some_cats"
    assert pii_replay.pii_replay_data[0].total_training_data == 1988
    assert pii_replay.pii_replay_data[0].unique_training_data == 2
    assert pii_replay.pii_replay_data[0].total_synthetic_data == 2044
    assert pii_replay.pii_replay_data[0].unique_synthetic_data == 2
    assert pii_replay.pii_replay_data[0].unique_synthetic_data_percentage == 100.0


def test_pii_replay_column_name_with_apostrophe():
    """Regression: a classified column whose name contains an apostrophe must not crash.

    PII replay previously filtered the synthetic column via ``DataFrame.query`` with a
    backtick-quoted column name. Under Python 3.12+ the query parser routes the backtick
    contents through CPython's tokenizer, so a name like ``judge's score`` raised
    ``SyntaxError: Failed to parse backticks`` (unterminated string literal).
    """
    col = "judge's score"
    training = pd.DataFrame({col: ["foo", "bar", "foo", "baz", "bar"]})
    synthetic = pd.DataFrame({col: ["foo", "foo", "bar", "qux", "qux"]})
    column_statistics = {
        col: ColumnStatistics(
            assigned_type="text",
            assigned_entity="name",
            detected_entity_counts={"name": 4},
            detected_entity_values={"name": {"foo", "bar"}},
            is_transformed=True,
            transform_functions={"fake"},
        )
    }

    evaluation_datasets = EvaluationDatasets.from_dataframes(
        training, synthetic, column_statistics=column_statistics, enable_sampling=False
    )
    pii_replay = PIIReplay.from_evaluation_datasets(evaluation_datasets)

    assert len(pii_replay.pii_replay_data) == 1
    datum = pii_replay.pii_replay_data[0]
    assert datum.column_name == col
    assert datum.pii_type == "name"
    assert datum.total_training_data == 4
    assert datum.unique_training_data == 2
    # synthetic values in {"foo", "bar"}: foo, foo, bar -> 3 rows, 2 unique
    assert datum.total_synthetic_data == 3
    assert datum.unique_synthetic_data == 2
    assert datum.unique_synthetic_data_percentage == 100.0


def test_pii_replay_percentage_rounds_up_to_one_decimal_place():
    training = pd.DataFrame({"name": ["Ada", "Grace", "Katherine"]})
    synthetic = pd.DataFrame({"name": ["Ada", "Margaret"]})
    column_statistics = {
        "name": ColumnStatistics(
            assigned_type="name",
            assigned_entity="name",
            detected_entity_counts={"name": 3},
            detected_entity_values={"name": {"Ada", "Grace", "Katherine"}},
            is_transformed=True,
            transform_functions={"fake"},
        )
    }
    evaluation_datasets = EvaluationDatasets.from_dataframes(
        training, synthetic, column_statistics=column_statistics, enable_sampling=False
    )

    pii_replay = PIIReplay.from_evaluation_datasets(evaluation_datasets)

    assert pii_replay.pii_replay_data[0].unique_synthetic_data_percentage == 33.4


def test_pii_replay_finds_detected_values_embedded_in_free_text():
    training = pd.DataFrame({"notes": ["Ada met Grace", "Call Ada"]})
    synthetic = pd.DataFrame({"notes": ["Ada appeared inside a sentence", "Grace and Ada", "No replay here", None]})
    column_statistics = {
        "notes": ColumnStatistics(
            assigned_type="text",
            assigned_entity="free_text",
            detected_entity_counts={"first_name": 3},
            detected_entity_values={"first_name": {"Ada", "Grace"}},
            is_transformed=True,
            transform_functions={"gliner"},
        )
    }
    evaluation_datasets = EvaluationDatasets.from_dataframes(
        training,
        synthetic,
        column_statistics=column_statistics,
        enable_sampling=False,
    )

    datum = PIIReplay.from_evaluation_datasets(evaluation_datasets).pii_replay_data[0]

    assert datum.total_synthetic_data == 2
    assert datum.unique_synthetic_data == 2
    assert datum.unique_synthetic_data_percentage == 100.0


def test_pii_replay_ignores_detected_values_inside_longer_words():
    training = pd.DataFrame({"notes": ["Ada met Ann"]})
    synthetic = pd.DataFrame({"notes": ["Flights to Canada", "Annual review", "Ada's file"]})
    column_statistics = {
        "notes": ColumnStatistics(
            assigned_type="text",
            assigned_entity="free_text",
            detected_entity_counts={"first_name": 2},
            detected_entity_values={"first_name": {"Ada", "Ann"}},
            is_transformed=True,
            transform_functions={"gliner"},
        )
    }
    evaluation_datasets = EvaluationDatasets.from_dataframes(
        training,
        synthetic,
        column_statistics=column_statistics,
        enable_sampling=False,
    )

    datum = PIIReplay.from_evaluation_datasets(evaluation_datasets).pii_replay_data[0]

    assert datum.total_synthetic_data == 1
    assert datum.unique_synthetic_data == 1
    assert datum.unique_synthetic_data_percentage == 50.0


_REFERENCE_WORD_CHARACTER = re.compile(r"\w")


def _reference_replayed_values(text: str, values: set[str]) -> set[str]:
    """The original one-regex-per-value search that ``_EmbeddedValueIndex`` must match exactly."""

    def pattern(value: str) -> re.Pattern[str]:
        start = r"(?<!\w)" if _REFERENCE_WORD_CHARACTER.match(value[0]) else ""
        end = r"(?!\w)" if _REFERENCE_WORD_CHARACTER.match(value[-1]) else ""
        return re.compile(f"{start}{re.escape(value)}{end}")

    return {value for value in values if pattern(value).search(text)}


def _random_text(rng: random.Random, max_length: int) -> str:
    # Word characters (ASCII, accented, digit, underscore) and non-word characters (space, punctuation, newline).
    alphabet = ["a", "b", "A", "é", "1", "_", " ", " ", "'", "-", ".", "\n", "@"]
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(1, max_length)))


@pytest.mark.parametrize("seed", range(20))
def test_embedded_value_index_matches_the_regex_search(seed: int):
    rng = random.Random(seed)
    texts = [_random_text(rng, 40) for _ in range(30)]
    values = {_random_text(rng, 4) for _ in range(40)}
    # Add substrings of the texts so many values do occur, including ones that start or end mid-word.
    for _ in range(40):
        text = rng.choice(texts)
        start = rng.randrange(len(text))
        values.add(text[start : rng.randint(start + 1, len(text))])
    index = _EmbeddedValueIndex(values)

    for text in texts:
        assert index.find(text) == _reference_replayed_values(text, values), (text, values)
    expected_rows = sum(1 for text in texts if _reference_replayed_values(text, values))
    expected_values = set().union(*(_reference_replayed_values(text, values) for text in texts))
    assert _synthetic_replay_counts(pd.Series(texts), values, substring_match=True) == (
        expected_rows,
        len(expected_values),
    )


@pytest.mark.parametrize(
    ("text", "value", "matches"),
    [
        ("Flights to Canada", "Ada", False),
        ("Ada's file", "Ada", True),
        ("x-7 left", "-7", True),
        ("x-7y", "-7", False),
        ("call (555)", "(555)", True),
        ("a.b.", ".b.", True),
        ("ab.", "b.", False),
    ],
)
def test_embedded_value_index_respects_word_boundaries(text: str, value: str, matches: bool):
    assert (value in _EmbeddedValueIndex([value]).find(text)) is matches
