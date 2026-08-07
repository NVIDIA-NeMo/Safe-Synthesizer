# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for time-series prompt construction."""

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.data_processing.record_utils import ParsedRecord
from nemo_safe_synthesizer.defaults import PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.generation.timeseries_prompting import (
    build_partial_record_prefix,
    build_rolling_record_prefill,
    build_training_compatible_prompt_token_ids,
)
from nemo_safe_synthesizer.llm.metadata import LLMPromptConfig


def test_builds_grouped_prefix_in_training_dialect():
    schema = {
        "properties": {
            "group_id": {"type": "integer"},
            "timestamp": {"type": "string"},
            "value": {"type": "number"},
        }
    }

    prefix = build_partial_record_prefix(
        columns=["group_id", "timestamp", "value"],
        schema=schema,
        group_column="group_id",
        group_id="7",
        timestamp_column="timestamp",
        start_timestamp="08/05/2026",
    )

    assert prefix == '{"group_id":7,"timestamp":"08\\/05\\/2026","'


def test_builds_ungrouped_prefix_without_pseudo_group():
    schema = {
        "properties": {
            "elapsed_seconds": {"type": "integer"},
            "value": {"type": "number"},
        }
    }

    prefix = build_partial_record_prefix(
        columns=["elapsed_seconds", "value"],
        schema=schema,
        group_column=PSEUDO_GROUP_COLUMN,
        group_id="0",
        timestamp_column="elapsed_seconds",
        start_timestamp="0",
    )

    assert prefix == '{"elapsed_seconds":0,"'


def test_supports_empty_timestamp_column_name():
    schema = {"properties": {"": {"type": "string"}, "value": {"type": "integer"}}}

    prefix = build_partial_record_prefix(
        columns=["", "value"],
        schema=schema,
        group_column=PSEUDO_GROUP_COLUMN,
        group_id="0",
        timestamp_column="",
        start_timestamp="2026-08-05",
    )

    assert prefix == '{"":"2026-08-05","'


def test_builds_rolling_prefill_from_exact_record_text():
    records = [
        ParsedRecord(text='{"value":1.0,"date":"08\\/05\\/2026"}', parsed={"value": 1.0}),
        ParsedRecord(text='{"value":2,"date":"08\\/06\\/2026"}', parsed={"value": 2}),
    ]

    assert build_rolling_record_prefill(records) == (
        '{"value":1.0,"date":"08\\/05\\/2026"}\n{"value":2,"date":"08\\/06\\/2026"}\n'
    )


def test_builds_empty_rolling_prefill():
    assert build_rolling_record_prefill([]) == ""


class _CharacterTokenizer:
    """Tokenizer stand-in with transparent token boundaries."""

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


@pytest.mark.parametrize(
    ("add_prompt_bos", "add_prompt_eos", "expected_special_tokens"),
    [
        pytest.param(True, True, [101, 102, 101], id="prompt-bos-and-eos"),
        pytest.param(True, False, [101, 101], id="prompt-bos"),
        pytest.param(False, True, [102, 101], id="prompt-eos"),
        pytest.param(False, False, [101], id="sequence-bos-only"),
    ],
)
def test_builds_training_compatible_special_token_boundary(
    add_prompt_bos,
    add_prompt_eos,
    expected_special_tokens,
):
    prompt_config = LLMPromptConfig(
        template="P{instruction}|{schema}|{prefill}",
        add_bos_token_to_prompt=add_prompt_bos,
        add_eos_token_to_prompt=add_prompt_eos,
        bos_token="<bos>",
        bos_token_id=101,
        eos_token="<eos>",
        eos_token_id=102,
    )

    token_ids = build_training_compatible_prompt_token_ids(
        tokenizer=_CharacterTokenizer(),
        prompt_config=prompt_config,
        instruction="I",
        schema_fragment="S",
        prefill='{"t":1,',
    )

    prompt_ids = [ord(character) for character in "PI|S|"]
    prefill_ids = [ord(character) for character in '{"t":1,']
    expected_ids = prompt_ids
    if add_prompt_bos:
        expected_ids = [101, *expected_ids]
    if add_prompt_eos:
        expected_ids.append(102)
    expected_ids.extend([101, *prefill_ids])

    assert token_ids == expected_ids
    assert [token for token in token_ids if token in {101, 102}] == expected_special_tokens


def test_encodes_rolling_records_as_training_segments():
    class RecordingTokenizer(_CharacterTokenizer):
        def __init__(self):
            self.encoded_text: list[str] = []

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            self.encoded_text.append(text)
            return super().encode(text, add_special_tokens=add_special_tokens)

    tokenizer = RecordingTokenizer()
    prompt_config = LLMPromptConfig(
        template="{instruction}{schema}{prefill}",
        add_bos_token_to_prompt=False,
        add_eos_token_to_prompt=False,
        bos_token="<bos>",
        bos_token_id=101,
        eos_token="<eos>",
        eos_token_id=102,
    )
    records = ['{"t":1}\n', '{"t":2}\n']

    build_training_compatible_prompt_token_ids(
        tokenizer=tokenizer,
        prompt_config=prompt_config,
        instruction="I",
        schema_fragment="S",
        prefill=records,
    )

    assert tokenizer.encoded_text == ["IS", *records]
