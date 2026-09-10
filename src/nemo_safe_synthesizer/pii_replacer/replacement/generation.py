# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interface for synthetic replacement value generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ...config.replace_pii import EntityType
from .types import CanonicalValue, EffectiveDependencyTuple


@dataclass(frozen=True, slots=True)
class ReplacementGenerationRequest:
    """Inputs required to deterministically generate one replacement value.

    Sensitive inputs are excluded from ``repr`` so request diagnostics do not
    disclose original or conditioning values.
    """

    entity_type: EntityType
    canonical_original_value: CanonicalValue = field(repr=False)
    effective_dependency_tuple: EffectiveDependencyTuple = field(repr=False)
    pattern: str | None
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.entity_type, EntityType):
            raise TypeError("replacement generation entity_type must be a normalized EntityType")
        if not isinstance(self.canonical_original_value, str):
            raise TypeError("canonical_original_value must be a string")
        if not isinstance(self.effective_dependency_tuple, tuple):
            raise TypeError("effective_dependency_tuple must be a tuple")
        if self.pattern is not None and not isinstance(self.pattern, str):
            raise TypeError("replacement generation pattern must be a string or None")
        if type(self.seed) is not int:
            raise TypeError("replacement generation seed must be an integer")


class ReplacementGenerator(Protocol):
    """Generate synthetic values behind the replacement executor's private seam.

    Implementations must be deterministic for equal requests. Mapping scope,
    cache reuse, call timing, and dataframe mutation remain responsibilities of
    the replacement executor.
    """

    def generate(self, request: ReplacementGenerationRequest) -> str:
        """Return one synthetic value satisfying ``request``."""
