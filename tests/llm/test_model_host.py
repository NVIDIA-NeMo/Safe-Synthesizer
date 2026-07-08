# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the local model host contract."""

from unittest.mock import MagicMock

from nemo_safe_synthesizer.generation.vllm_backend import VllmBackend
from nemo_safe_synthesizer.llm.model_host import ModelHost


def test_vllm_backend_is_a_typed_local_model_host() -> None:
    """The local generation backend exposes its model through the shared host boundary."""
    assert issubclass(VllmBackend, ModelHost)


def test_vllm_backend_model_delegates_to_loaded_engine() -> None:
    """The shared model property returns the engine already owned by the backend."""
    engine = object()
    backend = object.__new__(VllmBackend)
    backend.llm = engine  # type: ignore[assignment]

    assert backend.model is engine


def test_vllm_backend_model_is_none_before_initialization() -> None:
    """Callers can inspect whether the host has loaded its model yet."""
    backend = object.__new__(VllmBackend)
    backend.llm = None

    assert backend.model is None


def test_vllm_backend_tokenizer_delegates_to_loaded_engine() -> None:
    """The host exposes the tokenizer owned by its loaded engine."""
    tokenizer = object()
    backend = object.__new__(VllmBackend)
    backend.llm = MagicMock()
    backend.llm.get_tokenizer.return_value = tokenizer

    assert backend.tokenizer is tokenizer


def test_vllm_backend_tokenizer_is_none_before_initialization() -> None:
    """The tokenizer follows the hosted model's lifecycle."""
    backend = object.__new__(VllmBackend)
    backend.llm = None

    assert backend.tokenizer is None
