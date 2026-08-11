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

__all__ = [
    "FakerLike",
    "ScopedValueMap",
    "StandaloneColMap",
    "build_scoped_col_map",
    "seeded_faker",
    "stable_hash",
    "unit_key",
]


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
        locale: Faker locale (default ``"en_US"``).

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
    f = Faker(locale)
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

    Args:
        original_df: Source dataframe.
        col: Column to map.
        scope: Replacement scope (``"dataframe"``, ``"group"``, or ``"record"``).
        gk: Group-key column name when ``scope == "group"``.
        cfg: Replacement configuration (seed, locale).
        synthesize: Callable ``(sv, rng, fake, used) -> str | None``.
        track_used: When True, maintain an injective ``used`` set across the column.

    Returns:
        A ``ScopedValueMap`` keyed by scope unit.

    Example:
        Under ``scope="group"`` with groups A/B::

            ScopedValueMap("group", {"A": {"001": "syn-a"}, "B": {"002": "syn-b"}})
    """
    fake = seeded_faker(cfg.random_seed ^ stable_hash(col), cfg.locale)
    rng = fake.random
    used: set[str] = {str(v) for v in original_df[col].dropna().unique()} if track_used else set()

    def _unit_map(values: pd.Series, scope_key: Hashable) -> dict[str, str]:
        rng.seed(cfg.random_seed ^ stable_hash(f"{col}\x00{scope_key}"))
        mapping: dict[str, str] = {}
        for sv in (str(v) for v in values.dropna().unique()):
            new = synthesize(sv, rng, fake, used)
            if new and new != sv:
                mapping[sv] = new
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
