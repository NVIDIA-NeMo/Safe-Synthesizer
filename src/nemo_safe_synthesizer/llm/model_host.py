# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed lifecycle contract for components that own a local model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

ModelT = TypeVar("ModelT")
TokenizerT = TypeVar("TokenizerT")


class ModelHost(ABC, Generic[ModelT, TokenizerT]):
    """Own a local language model and tokenizer through teardown.

    This contract deliberately stops at model ownership. Tasks such as
    synthetic-record generation and column classification retain their own
    prompt construction, batching, and response parsing.
    """

    @property
    @abstractmethod
    def model(self) -> ModelT | None:
        """Return the hosted model, or ``None`` before initialization."""

    @property
    @abstractmethod
    def tokenizer(self) -> TokenizerT | None:
        """Return the hosted tokenizer, or ``None`` before initialization."""

    @abstractmethod
    def initialize(self) -> None:
        """Load the model and any resources required to use it."""

    @abstractmethod
    def teardown(self) -> None:
        """Release the model and its resources; this must be idempotent."""
