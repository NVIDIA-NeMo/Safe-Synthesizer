# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic seed derivation for structured and free-text mappings."""

from __future__ import annotations

import hashlib
import os

from ...errors import InternalError, ParameterError
from .types import CanonicalValue, FreeTextMappingKey, GroupMappingKey, RecordMappingKey

__all__ = ["derive_seed", "resolve_base_seed"]

_PERSON_RANDOM_SEED_ENV = "PERSON_RANDOM_SEED"


def resolve_base_seed(explicit_seed: int | None) -> int:
    """Resolve explicit, environment, and default seed precedence."""
    if explicit_seed is not None:
        return explicit_seed
    environment_seed = os.environ.get(_PERSON_RANDOM_SEED_ENV)
    if environment_seed is None:
        return 42
    try:
        return int(environment_seed)
    except ValueError as exc:
        raise ParameterError(f"{_PERSON_RANDOM_SEED_ENV} must be an integer") from exc


def derive_seed(
    base_seed: int,
    key: RecordMappingKey | GroupMappingKey | FreeTextMappingKey,
    *,
    purpose: str,
    attempt: int,
) -> int:
    """Derive a stable integer seed without serializing sensitive values into logs."""
    digest = hashlib.sha256()
    for component in _seed_components(base_seed, key, purpose, attempt):
        encoded = component.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big")


def _seed_components(
    base_seed: int,
    key: RecordMappingKey | GroupMappingKey | FreeTextMappingKey,
    purpose: str,
    attempt: int,
) -> tuple[str, ...]:
    """Return an unambiguous ordered representation of a mapping identity."""
    if isinstance(key, FreeTextMappingKey):
        scope = key.scope_identity
        if isinstance(scope, CanonicalValue):
            scope_components = (scope.type_tag, scope.normalized_value)
        else:
            scope_components = (type(scope).__qualname__, str(scope))
        return (
            "free_text",
            str(base_seed),
            purpose,
            str(attempt),
            key.entity_type.value,
            key.original_value,
            *scope_components,
        )

    original = key.canonical_original_value
    common = (
        str(base_seed),
        purpose,
        str(attempt),
        key.target_column,
        original.type_tag,
        original.normalized_value,
    )
    if isinstance(key, RecordMappingKey):
        return ("record", *common, str(key.row_position))
    group = key.original_group_identity
    if not isinstance(group, CanonicalValue):
        raise InternalError("group mapping identity must be a CanonicalValue")
    return ("group", *common, group.type_tag, group.normalized_value)
