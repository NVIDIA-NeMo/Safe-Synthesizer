# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the capability-based vLLM tokenizer probe adapter."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from nemo_safe_synthesizer.errors import GenerationError
from nemo_safe_synthesizer.generation.vllm_tokenizer import VllmTokenizerProbe


class _Tokenizer:
    vocab_size = 3
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2
    unk_token_id = None
    sep_token_id = None
    cls_token_id = None
    mask_token_id = None
    additional_special_tokens_ids = [1, 2]

    def __len__(self) -> int:
        return 5

    def get_added_vocab(self):
        return {"<b>": 4, "<a>": 3}

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(self, input_ids):
        return "".join(chr(token_id) for token_id in input_ids)


class _FailingTokenizer:
    @property
    def vocab_size(self) -> int:
        raise RuntimeError("provider failure")

    @property
    def bos_token_id(self) -> int:
        raise RuntimeError("provider failure")

    def __len__(self) -> int:
        raise RuntimeError("provider failure")

    def get_added_vocab(self):
        raise RuntimeError("provider failure")

    def encode(self, text, *, add_special_tokens):
        raise RuntimeError("provider failure")

    def decode(self, input_ids):
        raise RuntimeError("provider failure")


def test_vllm_tokenizer_probe_normalizes_public_capabilities() -> None:
    probe = VllmTokenizerProbe(_Tokenizer())

    assert probe.vocab_size == 3
    assert probe.total_size == 5
    assert probe.added_vocabulary == (("<a>", 3), ("<b>", 4))
    assert probe.special_token_ids[-1] == ("additional_special_tokens_ids", (1, 2))
    assert probe.encode_no_special("ab") == (97, 98)
    assert probe.decode((97, 98)) == "ab"


def test_vllm_tokenizer_probe_rejects_missing_capabilities() -> None:
    with pytest.raises(GenerationError, match="total vocabulary size"):
        _ = VllmTokenizerProbe(object()).total_size


@pytest.mark.parametrize(
    ("read_capability", "message"),
    [
        (lambda probe: probe.vocab_size, "vocab_size"),
        (lambda probe: probe.total_size, "total vocabulary size"),
        (lambda probe: probe.added_vocabulary, "added vocabulary"),
        (lambda probe: probe.special_token_ids, "special token IDs"),
        (lambda probe: probe.encode_no_special("probe"), "cannot encode"),
        (lambda probe: probe.decode((1,)), "cannot decode"),
    ],
)
def test_vllm_tokenizer_probe_normalizes_unexpected_provider_failures(
    read_capability: Callable[[VllmTokenizerProbe], object],
    message: str,
) -> None:
    probe = VllmTokenizerProbe(_FailingTokenizer())

    with pytest.raises(GenerationError, match=message) as exc_info:
        read_capability(probe)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
