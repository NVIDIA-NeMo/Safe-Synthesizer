# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared prompt token assembly."""

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.data_processing.prompt_tokens import (
    encode_prompt_token_ids,
    wrap_sequence_token_ids,
)
from nemo_safe_synthesizer.llm.metadata import LLMPromptConfig


class _CharacterTokenizer:
    """Tokenizer stand-in with transparent token boundaries."""

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


@pytest.fixture
def fixture_prompt_config() -> LLMPromptConfig:
    """Provide prompt metadata with distinct BOS and EOS token IDs."""
    return LLMPromptConfig(
        template="{instruction}{schema}{prefill}",
        add_bos_token_to_prompt=False,
        add_eos_token_to_prompt=False,
        bos_token="<bos>",
        bos_token_id=101,
        eos_token="<eos>",
        eos_token_id=102,
    )


@pytest.mark.parametrize(
    ("add_bos", "add_eos", "expected"),
    [
        pytest.param(True, True, [101, ord("P"), 102], id="bos-and-eos"),
        pytest.param(True, False, [101, ord("P")], id="bos"),
        pytest.param(False, True, [ord("P"), 102], id="eos"),
        pytest.param(False, False, [ord("P")], id="no-boundaries"),
    ],
)
def test_encode_prompt_token_ids_applies_configured_boundaries(
    fixture_prompt_config: LLMPromptConfig,
    add_bos: bool,
    add_eos: bool,
    expected: list[int],
):
    fixture_prompt_config.add_bos_token_to_prompt = add_bos
    fixture_prompt_config.add_eos_token_to_prompt = add_eos

    token_ids = encode_prompt_token_ids(
        "P",
        tokenizer=_CharacterTokenizer(),
        prompt_config=fixture_prompt_config,
    )

    assert token_ids == expected


@pytest.mark.parametrize(
    ("include_eos", "expected"),
    [
        pytest.param(True, [101, 11, 12, 102], id="closed-training-sequence"),
        pytest.param(False, [101, 11, 12], id="open-generation-sequence"),
    ],
)
def test_wrap_sequence_token_ids_returns_new_bounded_list(
    fixture_prompt_config: LLMPromptConfig,
    include_eos: bool,
    expected: list[int],
):
    content_ids = [11, 12]

    sequence_ids = wrap_sequence_token_ids(
        content_ids,
        prompt_config=fixture_prompt_config,
        include_eos=include_eos,
    )

    assert sequence_ids == expected
    assert content_ids == [11, 12]
