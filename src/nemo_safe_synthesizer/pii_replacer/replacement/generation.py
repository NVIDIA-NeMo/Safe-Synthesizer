# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract for replacement value generation.

Concrete generators live in ``replacement.generators``; this module does not import them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol

from ...config.replace_pii import EntityType, PiiSamplerBackend
from ...errors import InternalError
from .types import EffectiveDependencyTuple, require_effective_dependency_tuple

__all__ = ["ReplacementGenerationRequest", "ReplacementGenerator"]

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
    """Current value of each ``depends_on`` column for this row, in plan order.

    A dependency on another replaced column carries its already-replaced value, so an email can be built from the
    synthetic first and last names. A read-only conditioner such as gender carries the dataset value. Excluded from
    ``repr`` because the values may contain PII.
    """

    pattern: str | None
    """Plan pattern the generated value must follow, or ``None`` for the entity default.

    The default for an email that depends on names is ``{first}.{last}##@{domain}``, built from the dependency values.
    """

    seed: int
    """Seed that makes generation deterministic for equal requests."""

    resolved_dependency_values: ResolvedDependencyValues = field(default=(), repr=False)
    """Person-sampler values each conditioner may match, from ``data_to_sampler_value_mapping``.

    Used only to filter the sampler, for example gender ``Woman`` resolved to ``("female",)``. Generators still read
    ``effective_dependency_tuple`` for values they render, such as names in an email. Excluded from ``repr`` because
    the values derive from PII.
    """

    full_name_pattern: str | None = None
    """Pattern of the replaced full-name column this value depends on, or ``None``.

    Used to read the first, middle, and last names out of that column's value, for example from ``Hopper, Grace``
    written by ``{Last}, {First}``.
    """

    keep_original_locality: bool = False
    """Whether the original value's locality text is kept after the generated street.

    When ``True``, street-address generators return only a street and do not append linked city, state, ZIP, or
    country values; the executor adds back the original text that followed the street. For example, only the street
    of ``"489 Park Blvd, Mesa, IA 76237"`` is generated, and ``", Mesa, IA 76237"`` is kept.
    """

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
        if type(self.keep_original_locality) is not bool:
            raise InternalError("keep_original_locality must be a boolean")


class ReplacementGenerator(Protocol):
    """Generate synthetic values behind the replacement executor's private seam.

    Implementations must be deterministic for equal requests. Mapping scope,
    cache reuse, call timing, and dataframe mutation remain responsibilities of
    the replacement executor.
    """

    backend: ClassVar[PiiSamplerBackend]

    def generate(self, request: ReplacementGenerationRequest) -> str:
        """Return one synthetic value satisfying ``request``."""
