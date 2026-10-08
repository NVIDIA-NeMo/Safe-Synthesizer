# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract and compatibility exports for replacement value generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol

from ...config.replace_pii import EntityType, PiiSamplerBackend
from ...errors import InternalError
from .generators import FakerReplacementGenerator, NemotronPersonasReplacementGenerator
from .types import EffectiveDependencyTuple, require_effective_dependency_tuple

__all__ = [
    "FakerReplacementGenerator",
    "NemotronPersonasReplacementGenerator",
    "ReplacementGenerationRequest",
    "ReplacementGenerator",
]

ResolvedDependencyValues = tuple[tuple[EntityType, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class ReplacementGenerationRequest:
    """Inputs required to deterministically generate one replacement value."""

    entity_type: EntityType
    """Normalized entity type of the value to generate."""

    original_value: str = field(repr=False)
    """Exact accepted substring for free text, or the ``normalized_value`` of a
    ``CanonicalValue`` for structured data. Excluded from ``repr`` because it is PII."""

    effective_dependency_tuple: EffectiveDependencyTuple = field(repr=False)
    """Dependency values that condition generation. Excluded from ``repr`` because they may contain PII."""

    pattern: str | None
    """Plan pattern the generated value must follow, or ``None`` for the entity default."""

    seed: int
    """Seed that makes generation deterministic for equal requests."""

    resolved_dependency_values: ResolvedDependencyValues = field(default=(), repr=False)
    """Sampler values resolved for each dependency value. Excluded from ``repr`` because they derive from PII."""

    def __post_init__(self) -> None:
        if not isinstance(self.entity_type, EntityType):
            raise InternalError("replacement generation entity_type must be a normalized EntityType")
        if not isinstance(self.original_value, str):
            raise InternalError("replacement generation original_value must be a string")
        require_effective_dependency_tuple(self.effective_dependency_tuple, "effective_dependency_tuple")
        if self.pattern is not None and not isinstance(self.pattern, str):
            raise InternalError("replacement generation pattern must be a string or None")
        if type(self.seed) is not int:
            raise InternalError("replacement generation seed must be an integer")
        if not isinstance(self.resolved_dependency_values, tuple):
            raise InternalError("resolved_dependency_values must be a tuple")


class ReplacementGenerator(Protocol):
    """Generate synthetic values behind the replacement executor's private seam.

    Implementations must be deterministic for equal requests. Mapping scope,
    cache reuse, call timing, and dataframe mutation remain responsibilities of
    the replacement executor.
    """

    backend: ClassVar[PiiSamplerBackend]

    def generate(self, request: ReplacementGenerationRequest) -> str:
        """Return one synthetic value satisfying ``request``."""
