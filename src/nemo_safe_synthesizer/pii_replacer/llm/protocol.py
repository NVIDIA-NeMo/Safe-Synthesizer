# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed LLM seams for the two phases of PII replacement.

Discovery runs the heuristic detector first (pattern attachment and free-text
candidate selection). The enhancer then confirms, revises, or adds to that
result before it becomes the final replacement plan.

Replacement is a separate phase: the enhancer infers demographic constraints
used for persona conditioning; persona-backed and standalone replacements are
then generated programmatically. Once scoped mappings exist, the enhancer
detects entities in original free-text cells and returns spans and labels,
never synthetic values. The replacement layer resolves detections into scoped
substitutions and applies them.

Classes:
    PiiEnhancer: Provider interface injected at discovery and replacement seams.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import pandas as pd

from ...config.replace_pii import PiiReplacementPlan
from ..entities import Config
from ..models import DiscoveryResult, FreeTextDetection, PersonaInstance


@runtime_checkable
class PiiEnhancer(Protocol):
    """Provider interface injected at discovery and replacement-time seams."""

    def review_discovery(
        self,
        df: pd.DataFrame,
        discovery: DiscoveryResult,
        cfg: Config,
    ) -> DiscoveryResult:
        """Return the final detection result used to build the replacement plan."""
        ...

    def infer_persona_demographics(
        self,
        df: pd.DataFrame,
        instances: list[PersonaInstance],
        cfg: Config,
    ) -> list[PersonaInstance]:
        """Infer gender and race constraints before programmatic persona assignment."""
        ...

    def detect_freetext_entities(
        self,
        df: pd.DataFrame,
        columns: Sequence[str],
        plan: PiiReplacementPlan,
        cfg: Config,
    ) -> list[FreeTextDetection]:
        """Detect PII spans in original text; do not generate replacements."""
        ...


def select_enhancer(*, llm_enhancement: bool, enhancer: PiiEnhancer | None = None) -> PiiEnhancer:
    """Pick the enhancer for a run.

    Args:
        llm_enhancement: When ``True``, use ``NotImplementedEnhancer`` unless an
            enhancer is injected.
        enhancer: Explicit enhancer instance; takes precedence when not ``None``.

    Returns:
        ``NoopEnhancer`` when ``llm_enhancement`` is ``False`` and no enhancer is
        injected; otherwise the injected enhancer or ``NotImplementedEnhancer``.

    Example:
        ``llm_enhancement=False`` -> ``NoopEnhancer``
        ``llm_enhancement=True`` without injection -> ``NotImplementedEnhancer``
    """
    if enhancer is not None:
        return enhancer
    if llm_enhancement:
        from .not_implemented import NotImplementedEnhancer

        return NotImplementedEnhancer()
    from .noop import NoopEnhancer

    return NoopEnhancer()
