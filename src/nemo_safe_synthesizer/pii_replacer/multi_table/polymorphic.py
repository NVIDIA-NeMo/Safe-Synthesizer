# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve polymorphic FK cells into a parent key domain."""

from __future__ import annotations

from ...observability import get_logger
from .store import DomainState, PolymorphicColumnRoute, SharedRuntimeStore

logger = get_logger(__name__)

__all__ = ["resolve_polymorphic_domain"]


def resolve_polymorphic_domain(
    store: SharedRuntimeStore,
    route: PolymorphicColumnRoute,
    *,
    original: str,
    type_value: str | None,
) -> str | None:
    """Return the domain id for a polymorphic Id cell, or ``None`` if unresolved.

    Prefer the type discriminator; fall back to unique membership across target
    domains' known originals / used sets. Warn on type/value mismatch.
    """
    type_domain = route.targets.get(type_value) if type_value else None

    membership: list[str] = []
    for domain_id in dict.fromkeys(route.targets.values()):
        state = store.domains.get(domain_id)
        if state is None:
            continue
        if original in state.values or original in state.used:
            membership.append(domain_id)

    if type_domain is not None:
        if membership and type_domain not in membership:
            logger.user.warning(
                f"[PII Replacement] Polymorphic {route.bare_column!r} type={type_value!r} "
                f"selects domain {type_domain!r}, but value is only found under "
                f"{', '.join(membership)}; using type column."
            )
        return type_domain

    if len(membership) == 1:
        return membership[0]
    if len(membership) > 1:
        logger.user.warning(
            f"[PII Replacement] Polymorphic {route.bare_column!r} value appears in multiple "
            f"parent domains ({', '.join(membership)}) and type is missing; leaving unresolved."
        )
        return None
    logger.user.warning(
        f"[PII Replacement] Polymorphic {route.bare_column!r} has no type and value is not in "
        "any loaded parent domain; leaving unresolved."
    )
    return None
