# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prompt and sequence token-boundary assembly."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..llm.metadata import LLMPromptConfig

__all__ = [
    "EncodeOnlyTokenizer",
    "encode_prompt_token_ids",
    "wrap_sequence_token_ids",
]


class EncodeOnlyTokenizer(Protocol):
    """Tokenizer interface required for prompt construction."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Encode text while explicitly controlling tokenizer special tokens."""


def encode_prompt_token_ids(
    prompt: str,
    *,
    tokenizer: EncodeOnlyTokenizer,
    prompt_config: LLMPromptConfig,
) -> list[int]:
    """Encode a prompt with its configured BOS and EOS boundaries."""
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    if prompt_config.add_bos_token_to_prompt:
        prompt_ids.insert(0, prompt_config.bos_token_id)
    if prompt_config.add_eos_token_to_prompt:
        prompt_ids.append(prompt_config.eos_token_id)
    return prompt_ids


def wrap_sequence_token_ids(
    token_ids: Sequence[int],
    *,
    prompt_config: LLMPromptConfig,
    include_eos: bool,
) -> list[int]:
    """Return sequence IDs with a leading BOS and an optional trailing EOS."""
    sequence_ids = [prompt_config.bos_token_id, *token_ids]
    if include_eos:
        sequence_ids.append(prompt_config.eos_token_id)
    return sequence_ids
