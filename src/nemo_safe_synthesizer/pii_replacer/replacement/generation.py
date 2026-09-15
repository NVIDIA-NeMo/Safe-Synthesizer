# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract and compatibility exports for replacement value generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol

from ...config.replace_pii import EntityType, PiiSamplerBackend
from .generators import FakerReplacementGenerator, ManagedReplacementGenerator
from .generators._common import generated_value_is_valid
from .types import EffectiveDependencyTuple

__all__ = [
    "FakerReplacementGenerator",
    "ManagedReplacementGenerator",
    "ReplacementGenerationRequest",
    "ReplacementGenerator",
    "generated_value_is_valid",
]

ResolvedDependencyLabels = tuple[tuple[EntityType, tuple[str, ...] | None], ...]


@dataclass(frozen=True, slots=True)
class ReplacementGenerationRequest:
    """Inputs required to deterministically generate one replacement value.

    ``original_value`` is the exact accepted substring for free text and the
    ``normalized_value`` payload of a ``CanonicalValue`` for structured data.
    Sensitive inputs are excluded from ``repr`` so request diagnostics do not
    disclose original or conditioning values.
    """

    entity_type: EntityType
    original_value: str = field(repr=False)
    effective_dependency_tuple: EffectiveDependencyTuple = field(repr=False)
    pattern: str | None
    seed: int
    resolved_dependency_labels: ResolvedDependencyLabels = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.entity_type, EntityType):
            raise TypeError("replacement generation entity_type must be a normalized EntityType")
        if not isinstance(self.original_value, str):
            raise TypeError("replacement generation original_value must be a string")
        if not isinstance(self.effective_dependency_tuple, tuple):
            raise TypeError("effective_dependency_tuple must be a tuple")
        if self.pattern is not None and not isinstance(self.pattern, str):
            raise TypeError("replacement generation pattern must be a string or None")
        if type(self.seed) is not int:
            raise TypeError("replacement generation seed must be an integer")
        if not isinstance(self.resolved_dependency_labels, tuple):
            raise TypeError("resolved_dependency_labels must be a tuple")


class ReplacementGenerator(Protocol):
    """Generate synthetic values behind the replacement executor's private seam.

    Implementations must be deterministic for equal requests. Mapping scope,
    cache reuse, call timing, and dataframe mutation remain responsibilities of
    the replacement executor.
    """

    backend: ClassVar[PiiSamplerBackend]

    def generate(self, request: ReplacementGenerationRequest) -> str:
        """Return one synthetic value satisfying ``request``."""
