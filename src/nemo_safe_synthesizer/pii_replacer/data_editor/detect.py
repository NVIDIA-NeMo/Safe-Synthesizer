# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal entity extractor stubs for the transforms engine.

The heavyweight NER-based entity extractor was removed when tabular PII
replacement moved to :class:`~nemo_safe_synthesizer.pii_replacer.replacer.TabularPiiReplacer`.
The transforms ``Environment`` (still used by data-processing column updates and
by the structure-preserving ``fake.bothify`` templates) only needs an object
that satisfies the extractor interface, so these stubs implement every method
that :mod:`.environment` and :mod:`.edit` reference as graceful no-ops: no
entities are ever detected, and text passes through unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

DEFAULT_ENTITIES: list[str] = []


class EntityExtractor:
    """Interface expected by the transforms ``Environment``/``Editor``."""

    current_column: str | None = None

    def extract_entity_values(self, text: Any) -> list[Any]:
        raise NotImplementedError

    def extract_and_replace_entities(
        self, replace_fn: Callable[..., Any], text: Any, entities: Any = None
    ) -> Any:
        raise NotImplementedError

    def batch_update_cache(self, texts: Iterable[str], entities: Any = None) -> None:
        raise NotImplementedError


class EntityExtractorNoop(EntityExtractor):
    """No-op extractor: detects nothing and leaves text untouched.

    Used whenever no NER backend is supplied (e.g. data-processing column
    updates and ``fake.bothify`` structure-preserving transforms). NER-based
    template filters (``detect_entities``, ``redact_entities`` ...) degrade to
    "no entities found" rather than raising.
    """

    def __init__(self) -> None:
        self.current_column: str | None = None

    def extract_entity_values(self, text: Any) -> list[Any]:
        return []

    def extract_and_replace_entities(
        self, replace_fn: Callable[..., Any], text: Any, entities: Any = None
    ) -> Any:
        return text

    def batch_update_cache(self, texts: Iterable[str], entities: Any = None) -> None:
        return None
