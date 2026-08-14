# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed LLM seams for the two phases of PII replacement.

Discovery runs the heuristic detector first (pattern attachment and free-text
candidate selection). The discovery enhancer then confirms, revises, or adds to
that result before it becomes the final replacement plan.

Replacement is a separate phase: the replacement enhancer infers demographic
constraints used for persona conditioning; persona-backed and standalone
replacements are then generated programmatically. Once scoped mappings exist,
the replacement enhancer detects entities in original free-text cells and
returns spans and labels, never synthetic values. The replacement layer
resolves detections into scoped substitutions and applies them.

Classes:
    PiiDiscoveryEnhancer: Injected at the discovery review seam.
    PiiReplacementEnhancer: Injected at demographics and free-text detection seams.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd

from ...config.replace_pii import PiiReplacementPlan
from ..entities import Config
from ..models import DiscoveryResult, FreeTextDetection, PersonaInstance


class PiiDiscoveryEnhancer(ABC):
    """Provider interface injected at the discovery review seam."""

    @abstractmethod
    def review_discovery(
        self,
        df: pd.DataFrame,
        discovery: DiscoveryResult,
        cfg: Config,
    ) -> DiscoveryResult:
        """Return the final detection result used to build the replacement plan."""


class PiiReplacementEnhancer(ABC):
    """Provider interface injected at replacement-time LLM seams."""

    @abstractmethod
    def infer_persona_demographics(
        self,
        df: pd.DataFrame,
        instances: list[PersonaInstance],
        cfg: Config,
    ) -> list[PersonaInstance]:
        """Infer sex and race constraints before programmatic persona assignment."""

    @abstractmethod
    def detect_freetext_entities(
        self,
        df: pd.DataFrame,
        columns: Sequence[str],
        plan: PiiReplacementPlan,
        cfg: Config,
    ) -> list[FreeTextDetection]:
        """Detect PII spans in original text; do not generate replacements."""


def select_discovery_enhancer(
    *,
    llm_enhancement: bool,
    enhancer: PiiDiscoveryEnhancer | None = None,
) -> PiiDiscoveryEnhancer:
    """Pick the discovery enhancer for a run.

    Args:
        llm_enhancement: When ``True``, use ``NotImplementedEnhancer`` unless an
            enhancer is injected.
        enhancer: Explicit enhancer instance; takes precedence when not ``None``.

    Returns:
        ``NoopEnhancer`` when ``llm_enhancement`` is ``False`` and no enhancer is
        injected; otherwise the injected enhancer or ``NotImplementedEnhancer``.
    """
    if enhancer is not None:
        return enhancer
    if llm_enhancement:
        from .not_implemented import NotImplementedEnhancer

        return NotImplementedEnhancer()
    from .noop import NoopEnhancer

    return NoopEnhancer()


def select_replacement_enhancer(
    *,
    llm_enhancement: bool,
    enhancer: PiiReplacementEnhancer | None = None,
) -> PiiReplacementEnhancer:
    """Pick the replacement enhancer for a run.

    Args:
        llm_enhancement: When ``True``, use ``NotImplementedEnhancer`` unless an
            enhancer is injected.
        enhancer: Explicit enhancer instance; takes precedence when not ``None``.

    Returns:
        ``NoopEnhancer`` when ``llm_enhancement`` is ``False`` and no enhancer is
        injected; otherwise the injected enhancer or ``NotImplementedEnhancer``.
    """
    if enhancer is not None:
        return enhancer
    if llm_enhancement:
        from .not_implemented import NotImplementedEnhancer

        return NotImplementedEnhancer()
    from .noop import NoopEnhancer

    return NoopEnhancer()
