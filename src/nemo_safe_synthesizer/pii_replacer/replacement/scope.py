# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope identity and scope-keyed map building.

A replacement unit is a group value, a row, or the whole dataframe. Everything
that has to agree within a unit -- the RNG seeds, the original→synthetic map,
and the key the structured and free-text sections join on -- is derived here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Hashable, Sequence
from datetime import date
from random import Random
from typing import Protocol, cast

import numpy as np
import pandas as pd

from ...errors import InternalError
from ..entities import Config
from ..models import ScopedValueMap, StandaloneColMap

_FAKER_IMPORT_ERROR: BaseException | None = None
try:  # Faker generates entity-driven values (and is an offline persona fallback).
    from faker import Faker
except Exception as exc:  # pragma: no cover - faker should be installed
    Faker = None  # type: ignore
    _FAKER_IMPORT_ERROR = exc

# Managed persona parquet locales that Faker does not ship. Apply still builds
# Faker for standalone maps / middle names / fallbacks, so these map to the
# closest installed provider locale.
FAKER_LOCALE_FALLBACKS: dict[str, str] = {
    "en_SG": "en_US",
    "hi_Deva_IN": "hi_IN",
    "hi_Latn_IN": "hi_IN",
}

__all__ = [
    "FAKER_LOCALE_FALLBACKS",
    "FakerLike",
    "ScopedValueMap",
    "StandaloneColMap",
    "build_scoped_col_map",
    "faker_locale_supported",
    "resolve_faker_locale",
    "seeded_faker",
    "stable_hash",
    "unit_key",
]


def faker_locale_supported(locale: str) -> bool:
    """Return whether the installed Faker package recognizes ``locale``."""
    if Faker is None:
        return True
    try:
        from faker.config import AVAILABLE_LOCALES
    except ImportError:
        return True
    return locale in AVAILABLE_LOCALES


def resolve_faker_locale(locale: str) -> tuple[str, str | None]:
    """Return ``(faker_locale, fallback_from)`` for constructing Faker.

    Managed parquet locales such as ``en_SG`` are kept for persona sampling but
    are not Faker providers. When ``locale`` is unsupported, a documented
    fallback is used and ``fallback_from`` is the original locale; otherwise
    ``fallback_from`` is ``None``.

    Args:
        locale: Configured ``replace_pii.replacement.locale``.

    Returns:
        Effective Faker locale and the original locale when a fallback applied.
    """
    if faker_locale_supported(locale):
        return locale, None
    fallback = FAKER_LOCALE_FALLBACKS.get(locale)
    if fallback is None:
        return locale, None
    return fallback, locale


class FakerLike(Protocol):
    """Structural subset of ``Faker`` used by entity and persona replacement."""

    random: Random

    def first_name(self) -> str: ...

    def first_name_male(self) -> str: ...

    def first_name_female(self) -> str: ...

    def last_name(self) -> str: ...

    def email(self) -> str: ...

    def phone_number(self) -> str: ...

    def ssn(self) -> str: ...

    def street_address(self) -> str: ...

    def city(self) -> str: ...

    def state_abbr(self) -> str: ...

    def postcode(self) -> str: ...

    def credit_card_number(self) -> str: ...

    def ipv4(self) -> str: ...

    def ipv6(self) -> str: ...

    def uuid4(self) -> object: ...

    def random_element(self, elements: Sequence[str]) -> str: ...

    def date_of_birth(self, minimum_age: int = ..., maximum_age: int = ...) -> date: ...

    def building_number(self) -> str: ...

    def street_name(self) -> str: ...


def stable_hash(s: str) -> int:
    """Return a stable 32-bit seed component for ``s`` (not a security digest)."""
    return int(hashlib.md5(s.encode("utf-8"), usedforsecurity=False).hexdigest()[:8], 16)


def seeded_faker(seed: int, locale: str = "en_US") -> FakerLike:
    """Return a Faker instance seeded for reproducible draws.

    Args:
        seed: Seed passed to ``Faker.seed_instance``.
        locale: Requested locale (default ``"en_US"``). Managed-only locales
            such as ``en_SG`` are remapped via ``FAKER_LOCALE_FALLBACKS``.

    Returns:
        A configured ``Faker`` instance.

    Raises:
        InternalError: If the ``faker`` package is not installed.

    Example:
        ``seeded_faker(42)`` always yields the same sequence for a given locale.
    """
    if Faker is None:
        raise InternalError(
            "The 'faker' package is required for PII replacement but is not installed. "
            "Install the project environment with: "
            "uv sync --frozen --extra cu129 --extra engine --group dev"
        ) from _FAKER_IMPORT_ERROR
    faker_locale, _fallback_from = resolve_faker_locale(locale)
    f = Faker(faker_locale)
    f.seed_instance(seed)
    return cast(FakerLike, f)


def unit_key(
    scope: str,
    group_value: Hashable | None,
    row_indices: Sequence[Hashable] | None,
) -> Hashable | list[Hashable] | None:
    """Shared unit identity tying structured replacements to free text.

    Args:
        scope: ``"group"``, ``"record"``, or dataframe scope label.
        group_value: Group key value when ``scope == "group"``.
        row_indices: Row indices covered by this unit.

    Returns:
        The group value, or a single row index (or list of indices).

    Example:
        ``unit_key("group", "patient-42", [3, 4, 5])`` -> ``"patient-42"``;
        ``unit_key("record", None, [7])`` -> ``7``.
    """
    match scope:
        case "group":
            return group_value
        case _:
            idxs: list[Hashable] = [int(i) if isinstance(i, (int, np.integer)) else i for i in (row_indices or [])]
            return idxs[0] if len(idxs) == 1 else idxs


def build_scoped_col_map(
    original_df: pd.DataFrame,
    col: str,
    scope: str,
    gk: str | None,
    cfg: Config,
    *,
    synthesize: Callable[[str, Random, FakerLike, set[str]], str | None],
    track_used: bool = False,
    seed_key: str | None = None,
    used: set[str] | None = None,
    preexisting: dict[str, str] | None = None,
) -> ScopedValueMap:
    """Build a scope-keyed original→synthetic map for one standalone column.

    Shared nesting for ``dataframe`` / ``group`` / ``record``: each scope unit
    gets a seeded RNG and one map over that unit's distinct originals.
    ``synthesize(sv, rng, fake, used)`` returns the replacement (or ``None``).

    When ``track_used`` is True, every original seeds ``used`` and each accepted
    synthetic is added so later values stay injective across the column.
    Within a unit the same original keeps the same synthetic; across units
    (group/record) the same original may get independent synthetics.

    ``scope="record"`` builds one unit map per row -- fine for typical training
    samples, costly on large frames (see ``standalone._RECORD_SCOPE_COST_WARN_ROWS``).

    Optional ``seed_key`` / ``used`` / ``preexisting`` support shared multi-table
    domains (pass ``seed_key=domain_id`` so RNG does not depend on which
    ``table.column`` first saw a value).

    Args:
        original_df: Source dataframe.
        col: Column to map.
        scope: Replacement scope (``"dataframe"``, ``"group"``, or ``"record"``).
        gk: Group-key column name when ``scope == "group"``.
        cfg: Replacement configuration (seed, locale).
        synthesize: Callable ``(sv, rng, fake, used) -> str | None``.
        track_used: When True, maintain an injective ``used`` set across the column.
        seed_key: Optional seed identity (defaults to ``col``).
        used: Optional shared ``used`` set (defaults to a fresh per-column set).
        preexisting: Optional original→synthetic entries to reuse before synthesizing.

    Returns:
        A ``ScopedValueMap`` keyed by scope unit.

    Example:
        Under ``scope="group"`` with groups A/B::

            ScopedValueMap("group", {"A": {"001": "syn-a"}, "B": {"002": "syn-b"}})
    """
    identity = seed_key if seed_key is not None else col
    fake = seeded_faker(cfg.random_seed ^ stable_hash(identity), cfg.locale)
    rng = fake.random
    if used is None:
        used = {str(v) for v in original_df[col].dropna().unique()} if track_used else set()
    elif track_used:
        used |= {str(v) for v in original_df[col].dropna().unique()}
    # Shared preexisting is only for multi-table domain reuse. Do not accumulate
    # synthetics into it across group/record units — that would collapse
    # per-unit independence for the same original.
    shared = preexisting

    def _unit_map(values: pd.Series, scope_key: Hashable) -> dict[str, str]:
        rng.seed(cfg.random_seed ^ stable_hash(f"{identity}\x00{scope_key}"))
        mapping: dict[str, str] = {}
        for sv in (str(v) for v in values.dropna().unique()):
            if shared is not None and sv in shared:
                mapping[sv] = shared[sv]
                if track_used:
                    used.add(shared[sv])
                continue
            new = synthesize(sv, rng, fake, used)
            if new and new != sv:
                mapping[sv] = new
                if shared is not None:
                    shared[sv] = new
                if track_used:
                    used.add(new)
        return mapping

    match scope:
        case "group" if gk and gk in original_df.columns:
            data: dict[Hashable, object] = {
                cast(Hashable, gval): _unit_map(gdf[col], cast(Hashable, gval))
                for gval, gdf in original_df.groupby(gk, dropna=True)
            }
            return ScopedValueMap("group", data)
        case "record":
            data = {
                cast(Hashable, idx): _unit_map(original_df.loc[[idx], col], cast(Hashable, idx))
                for idx in original_df.index
            }
            return ScopedValueMap("record", cast(dict[Hashable, object], data))
        case _:
            return ScopedValueMap("flat", cast(dict[Hashable, object], _unit_map(original_df[col], "dataframe")))
