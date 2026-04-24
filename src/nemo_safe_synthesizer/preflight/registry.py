# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight registry: build, validate, register plugin checks."""

from __future__ import annotations

from collections.abc import Iterator, KeysView, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType

from .base import _CORE_NAMESPACES, PreflightCheck
from .checks import _CORE_CHECKS
from .types import PreflightStage

__all__ = [
    "PreflightRegistry",
    "build_registry",
    "get_registry",
    "register_preflight_check",
    "reset_preflight_plugins",
]


@dataclass(frozen=True)
class PreflightRegistry:
    """Ordered, name-keyed view of the checks the orchestrator will run.

    Constructed by :func:`build_registry` and treated as immutable.
    Iteration yields the ``PreflightCheck`` instances themselves (not
    their names), so ``for check in registry`` reads naturally;
    name-based access uses ``registry[name]`` / ``name in registry``.
    Extend by calling ``register_preflight_check`` or rebuilding via
    ``build_registry``; never mutate in place.

    Attributes:
        checks: Insertion-ordered mapping from ``PreflightCheck.name`` to
            the check instance. Backed by ``types.MappingProxyType`` so
            the underlying dict is hidden from callers.
    """

    checks: Mapping[str, PreflightCheck]

    def __iter__(self) -> Iterator[PreflightCheck]:
        return iter(self.checks.values())

    def __contains__(self, name: object) -> bool:
        return name in self.checks

    def __getitem__(self, name: str) -> PreflightCheck:
        return self.checks[name]

    def __len__(self) -> int:
        return len(self.checks)

    @property
    def names(self) -> KeysView[str]:
        """The check names, in registry order."""
        return self.checks.keys()


_STAGE_ORDER: dict[PreflightStage, int] = {
    PreflightStage.CONFIG: 0,
    PreflightStage.DATAFRAME: 1,
    PreflightStage.METADATA: 2,
    PreflightStage.ADVISORY: 3,
}


_PLUGIN_CHECKS: list[PreflightCheck] = []


def register_preflight_check(check: PreflightCheck) -> PreflightCheck:
    """Register a third-party ``PreflightCheck`` instance for inclusion in the registry.

    Plugin checks are appended after all core checks when the registry is
    (re)built by ``build_registry``. The function takes an *instance*
    (not a class) and returns it unchanged, e.g.::

        register_preflight_check(MyCheck())

    The first dotted segment of ``name`` must not match a reserved core
    namespace; this is enforced here rather than at class-definition
    time so that core checks -- which are registered by direct
    inclusion in ``_CORE_CHECKS`` -- can legitimately use those
    prefixes.

    Not thread-safe. Registration is expected at import / boot time --
    serialize externally if you call this from multiple threads.
    """
    name = check.name
    prefix = name.split(".", 1)[0] if "." in name else name
    if prefix in _CORE_NAMESPACES:
        raise ValueError(
            f"Plugin {type(check).__qualname__} uses reserved core namespace {prefix!r}; "
            "prefix your check name with a unique namespace. "
            f"Reserved namespaces: {sorted(_CORE_NAMESPACES)}"
        )
    # Build the prospective registry *before* mutating module state so a
    # failed validation (duplicate name, bad ``requires``, stage order)
    # doesn't leave a poisoned entry in ``_PLUGIN_CHECKS`` that would
    # break every subsequent registration.
    candidate_plugins = tuple(_PLUGIN_CHECKS) + (check,)
    new_registry = build_registry(_CORE_CHECKS, candidate_plugins)
    _PLUGIN_CHECKS.append(check)
    global _REGISTRY
    _REGISTRY = new_registry
    return check


def reset_preflight_plugins() -> None:
    """Clear all registered plugins and rebuild the registry from core checks.

    Intended for use in tests and notebooks where a clean registry is
    required between runs. Not thread-safe; see ``register_preflight_check``.
    """
    global _REGISTRY
    _PLUGIN_CHECKS.clear()
    _REGISTRY = build_registry(_CORE_CHECKS)


def _validate_registry(ordered: Sequence[PreflightCheck]) -> None:
    """Enforce registry-shape invariants on an ordered check sequence.

    * Each check's ``name`` is unique.
    * Every entry in ``requires`` refers to a check that appears earlier.
    * Checks are ordered by stage (``CONFIG`` -> ``DATAFRAME`` ->
      ``METADATA`` -> ``ADVISORY``) without back-tracking.
    """
    seen: set[str] = set()
    max_stage_idx = -1
    for check in ordered:
        if check.name in seen:
            raise RuntimeError(f"Duplicate preflight check name: {check.name!r}")
        seen.add(check.name)

        for dep in check.requires:
            if dep not in seen:
                raise RuntimeError(
                    f"Preflight check {check.name!r} requires {dep!r} which is "
                    "unknown or not declared earlier in the registry."
                )

        stage_idx = _STAGE_ORDER[check.stage]
        if stage_idx < max_stage_idx:
            raise RuntimeError(
                f"Preflight check {check.name!r} in stage {check.stage.value!r} "
                "appears after a later-stage check; registry must be stage-monotonic."
            )
        max_stage_idx = max(max_stage_idx, stage_idx)


def build_registry(*sources: Sequence[PreflightCheck]) -> PreflightRegistry:
    """Merge one or more check sequences into a validated ``PreflightRegistry``.

    Entries are stably sorted by ``stage`` so plugins registered from
    ``_PLUGIN_CHECKS`` (typically appended after core) slot into the
    appropriate stage block while preserving relative order within a
    stage.
    """
    merged = list(chain.from_iterable(sources))
    merged.sort(key=lambda c: _STAGE_ORDER[c.stage])
    _validate_registry(merged)
    return PreflightRegistry(checks=MappingProxyType({c.name: c for c in merged}))


_REGISTRY: PreflightRegistry = build_registry(_CORE_CHECKS)


def get_registry() -> PreflightRegistry:
    """Return the current preflight registry.

    The registry is rebuilt each time a plugin is registered via
    ``register_preflight_check`` or cleared via ``reset_preflight_plugins``.
    Always call this function rather than caching the result across
    registration boundaries.
    """
    return _REGISTRY
