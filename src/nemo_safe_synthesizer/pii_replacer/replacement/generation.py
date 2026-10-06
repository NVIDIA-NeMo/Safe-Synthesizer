# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interface for synthetic replacement value generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol

from ...config.replace_pii import EntityType, PiiReplacementSettings, PiiSamplerBackend, PiiSamplerConfig
from ...errors import InternalError
from .types import EffectiveDependencyTuple, require_effective_dependency_tuple

__all__ = [
    "FakerReplacementGenerator",
    "ManagedReplacementGenerator",
    "ReplacementGenerationRequest",
    "ReplacementGenerator",
]


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


class ReplacementGenerator(Protocol):
    """Generate synthetic values behind the replacement executor's private seam.

    Implementations must be deterministic for equal requests. Mapping scope,
    cache reuse, call timing, and dataframe mutation remain responsibilities of
    the replacement executor.
    """

    backend: ClassVar[PiiSamplerBackend]

    def generate(self, request: ReplacementGenerationRequest) -> str:
        """Return one synthetic value satisfying ``request``."""


class _SamplerReplacementGenerator(ReplacementGenerator):
    """Shared construction for generators bound to one sampler backend."""

    def __init__(self, *, settings: PiiReplacementSettings, sampler: PiiSamplerConfig) -> None:
        if sampler.backend is not self.backend:
            raise InternalError(
                f"{type(self).__name__} requires the {self.backend.value} sampler backend, got {sampler.backend.value}"
            )
        self._settings = settings
        self._sampler = sampler


class ManagedReplacementGenerator(_SamplerReplacementGenerator):
    """Generate replacements using managed person-sampling assets.

    Args:
        settings: Locale and seed configuration shared by replacement
            generators.
        sampler: Managed sampler configuration, including its asset path.

    Replacement execution is introduced by a follow-up change.
    """

    backend: ClassVar[PiiSamplerBackend] = PiiSamplerBackend.MANAGED

    def generate(self, request: ReplacementGenerationRequest) -> str:
        """Generate a managed-asset replacement for ``request``."""
        raise NotImplementedError("managed replacement generation is not implemented")


class FakerReplacementGenerator(_SamplerReplacementGenerator):
    """Generate replacements using Faker for person-like values.

    Args:
        settings: Locale and seed configuration shared by replacement
            generators.
        sampler: Faker sampler configuration.

    Replacement execution is introduced by a follow-up change.
    """

    backend: ClassVar[PiiSamplerBackend] = PiiSamplerBackend.FAKER

    def generate(self, request: ReplacementGenerationRequest) -> str:
        """Generate a Faker-backed replacement for ``request``."""
        raise NotImplementedError("Faker replacement generation is not implemented")
