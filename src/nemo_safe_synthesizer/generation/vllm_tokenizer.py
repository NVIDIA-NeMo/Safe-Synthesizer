# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capability adapter for the tokenizer returned by vLLM."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from ..errors import GenerationError

_SPECIAL_TOKEN_FIELDS = (
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "unk_token_id",
    "sep_token_id",
    "cls_token_id",
    "mask_token_id",
    "additional_special_tokens_ids",
)


class VllmTokenizerProbe:
    """Adapt vLLM's public tokenizer capabilities to the NSS parity protocol."""

    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        return self._integer_attribute("vocab_size")

    @property
    def total_size(self) -> int:
        try:
            size = len(cast(Any, self._tokenizer))
        except Exception as exc:
            raise GenerationError("The vLLM tokenizer does not expose a total vocabulary size.") from exc
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise GenerationError("The vLLM tokenizer exposed an invalid total vocabulary size.")
        return size

    @property
    def added_vocabulary(self) -> Sequence[tuple[str, int]]:
        try:
            method = getattr(self._tokenizer, "get_added_vocab", None)
            if not callable(method):
                raise GenerationError("The vLLM tokenizer does not expose added vocabulary.")
            vocabulary = method()
            return tuple(sorted((str(token), self._integer(token_id)) for token, token_id in vocabulary.items()))
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError("The vLLM tokenizer exposed invalid added vocabulary.") from exc

    @property
    def special_token_ids(self) -> Sequence[tuple[str, int | tuple[int, ...] | None]]:
        try:
            return tuple(
                (name, self._special_value(getattr(self._tokenizer, name, None))) for name in _SPECIAL_TOKEN_FIELDS
            )
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError("The vLLM tokenizer exposed invalid special token IDs.") from exc

    def encode_no_special(self, text: str) -> Sequence[int]:
        try:
            ids = cast(Any, self._tokenizer).encode(text, add_special_tokens=False)
            return tuple(self._integer(token_id) for token_id in ids)
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError("The vLLM tokenizer cannot encode NSS parity probes without special tokens.") from exc

    def decode(self, input_ids: Sequence[int]) -> str:
        try:
            decoded = cast(Any, self._tokenizer).decode(list(input_ids))
        except Exception as exc:
            raise GenerationError("The vLLM tokenizer cannot decode NSS parity probes.") from exc
        if not isinstance(decoded, str):
            raise GenerationError("The vLLM tokenizer returned a non-text decode result.")
        return decoded

    def _integer_attribute(self, name: str) -> int:
        try:
            return self._integer(getattr(self._tokenizer, name, None))
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(f"The vLLM tokenizer cannot expose {name}.") from exc

    @staticmethod
    def _integer(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GenerationError("The vLLM tokenizer exposed an invalid vocabulary ID.")
        return value

    @classmethod
    def _special_value(cls, value: object) -> int | tuple[int, ...] | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return tuple(cls._integer(item) for item in value)
        return cls._integer(value)
