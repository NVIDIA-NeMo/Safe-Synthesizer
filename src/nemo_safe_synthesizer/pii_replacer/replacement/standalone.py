# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity-driven columns replaced on their own, with no persona behind them."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta
from random import Random

import pandas as pd

from ...config.replace_pii import PiiColumnPlan, PiiEntity, PiiReplacementPlan
from ...observability import get_logger
from .. import entities
from ..detection import API_PREFIXES, UUID_RE
from ..entity_handlers import get_handler
from ..models import ScopedValueMap
from ..patterns import (
    detect_date_format,
    matching_template,
    pattern_preserving_token,
)
from .scope import FakerLike, build_scoped_col_map

logger = get_logger(__name__)

__all__ = ["build_standalone_maps", "synth_date_value"]

# Warn once when record-scope standalone maps would allocate one unit map per row
# on a large frame. Prefer ``group`` / ``dataframe`` when per-row independence is not required.
_RECORD_SCOPE_COST_WARN_ROWS = 25_000


def iter_standalone_specs(
    plan: PiiReplacementPlan,
    persona_backend: str,
) -> Iterator[PiiColumnPlan]:
    """Yield standalone column specs for the replacement pass.

    Entity-driven columns (and non-``pgm`` phones) are not persona-sourced, so they
    take the standalone path even when the plan associates them with a persona.
    With backend ``"faker"``, a plan phone under a persona set is yielded here;
    with ``"pgm"`` it is not.

    Args:
        plan: Resolved replacement plan.
        persona_backend: Effective persona backend (``"pgm"``, ``"managed"``, ``"faker"``).

    Yields:
        Column specs whose effective apply path is ``"standalone_map"``.
    """
    for col_set in plan.persona_backed_columns:
        for spec in col_set.columns_to_replace:
            if spec.entity_type is None or spec.entity_type == PiiEntity.free_text:
                continue
            if entities.effective_apply_path(spec.entity_type.value, persona_backend) == "standalone_map":
                yield spec
    for spec in plan.standalone_columns_to_replace:
        if spec.entity_type is None or spec.entity_type == PiiEntity.free_text:
            continue
        yield spec


def synth_date_value(original: str, fmt: str, rng: Random) -> str | None:
    """Return a synthetic date by shifting the original within +/- 1 year.

    Args:
        original: Original date string.
        fmt: ``strftime`` format used to parse and print the date.
        rng: Random source with ``randint``.

    Returns:
        Shifted date formatted with ``fmt``, or ``None`` if parsing fails.

    Example:
        ``"2020-01-15"`` with ``"%Y-%m-%d"`` -> e.g. ``"2021-01-15"``.
    """
    try:
        d = datetime.strptime(str(original).strip(), fmt).date()
    except (ValueError, TypeError):
        return None
    offset = rng.randint(-365, 365)
    if offset == 0:
        offset = 1
    return (d + timedelta(days=offset)).strftime(fmt)


def synth_dob_programmatic(original: str, rng: Random, fmt: str | None = None) -> str | None:
    """Return a synthetic DOB by perturbing the original birth date up to +/- 1 year.

    Args:
        original: Original date-of-birth string.
        rng: Random source passed to ``synth_date_value``.
        fmt: ``strftime`` format to parse and print the date. When ``None``, the
            format is inferred from the value itself.

    Returns:
        Perturbed date string, or ``None`` if parsing fails.

    Example:
        ``"1985-03-15"`` with ``fmt="%Y-%m-%d"`` -> e.g. ``"1986-03-15"`` (same format).
    """
    fmt = fmt or detect_date_format(original)
    return synth_date_value(original, fmt, rng)


def fake_value(entity: str, original: str, fake: FakerLike) -> str:
    """Return one Faker (or shape-preserving) draw for an entity-driven cell.

    Args:
        entity: Entity type label (for example ``"unique_identifier"``).
        original: Original cell value (used for shape-preserving tokens).
        fake: Seeded ``Faker`` instance.

    Returns:
        A synthetic value for the entity type.

    Example:
        ``fake_value("unique_identifier", "550e8400-e29b-41d4-a716-446655440000", fake)``
        returns a new UUID; an opaque token keeps its character classes via
        ``pattern_preserving_token``.
    """
    rng = fake.random
    match entity:
        case "credit_debit_card":
            return fake.credit_card_number()
        case "ipv4":
            return fake.ipv4()
        case "ipv6":
            return fake.ipv6()
        case "unique_identifier":
            if UUID_RE.match(original.strip()):
                return str(fake.uuid4())
            return pattern_preserving_token(original, rng)
        case "api_key":
            for pfx in API_PREFIXES:
                if original.startswith(pfx):
                    return pfx + pattern_preserving_token(original[len(pfx) :], rng)
            return pattern_preserving_token(original, rng)
        # Persona-sourced entities listed under standalone still get real Faker draws
        # (independent of any persona); pattern_preserving_token is only for opaque tokens.
        case "first_name":
            return fake.first_name()
        case "last_name":
            return fake.last_name()
        case "middle_name":
            return fake.first_name()
        case "full_name":
            return f"{fake.first_name()} {fake.last_name()}"
        case "email":
            return fake.email()
        case "phone_number":
            return fake.phone_number()
        case "ssn":
            return fake.ssn()
        case "street_address":
            return fake.street_address()
        case "city":
            return fake.city()
        case "state":
            return fake.state_abbr()
        case "zipcode":
            return fake.postcode()
        case "national_id":
            return fake.ssn()
        case _:
            return pattern_preserving_token(original, rng)


def unique_synthetic(
    sv: str,
    entity: str,
    patterns: Sequence[str],
    rng: Random,
    fake: FakerLike,
    used: set[str],
) -> str | None:
    """Return a fresh synthetic for one original, injective within ``used``.

    Tries the matching column template first, then a free handler draw, then a
    ``{base}-{n}`` suffix. Missing markers like ``"N/A"`` return ``None`` so they
    are not rewritten into lookalike placeholders.

    Args:
        sv: Original string value.
        entity: Entity type label.
        patterns: Column format templates from the plan.
        rng: Random source for handler draws.
        fake: Seeded ``Faker`` instance.
        used: Set of values already assigned (must stay injective).

    Returns:
        A synthetic string not in ``used``, or ``None`` for missing-value markers.
    """
    # A cell saying it holds nothing ('N/A') is nobody's identifier: it wears none
    # of the column's formats, and replacing it in its own shape would write 'E/S'
    # where the data said there was no value at all.
    if entities.is_missing_value(sv):
        return None

    def _fresh(cand: str | None) -> bool:
        return bool(cand) and cand != sv and cand not in used

    handler = get_handler(entity)
    generators = []
    if patterns:
        # The column names the formats it writes, and this value is written in the
        # first of them that describes it; one they all miss keeps its own shape.
        shape = (matching_template(sv, patterns),)
        generators.append(lambda: handler.generate(sv, fake, patterns=shape, rng=rng))
    generators.append(lambda: handler.generate(sv, fake, rng=rng))
    for gen in generators:
        cand = gen()
        for _ in range(200):
            if _fresh(cand):
                return cand
            cand = gen()
    base = handler.generate(sv, fake, rng=rng) or sv
    for suffix in range(1, 100000):
        cand = f"{base}-{suffix}"
        if _fresh(cand):
            return cand
    return None


def _build_identifier_map(
    original_df: pd.DataFrame,
    col: str,
    entity: str,
    patterns: Sequence[str],
    scope: str,
    gk: str | None,
    cfg: entities.Config,
) -> ScopedValueMap:
    """Build an identifier/phone/card map keyed by plan scope.

    Uses ``track_used=True`` so synthetics stay injective across the column.
    """

    def synthesize(sv: str, rng: Random, fake: FakerLike, used: set[str]) -> str | None:
        return unique_synthetic(sv, entity, patterns, rng, fake, used)

    return build_scoped_col_map(
        original_df,
        col,
        scope,
        gk,
        cfg,
        synthesize=synthesize,
        track_used=True,
    )


def _build_dob_map(
    original_df: pd.DataFrame,
    col: str,
    patterns: Sequence[str],
    scope: str,
    gk: str | None,
    cfg: entities.Config,
) -> ScopedValueMap:
    """Build a birth-date map keyed by plan scope.

    Does not track a global ``used`` set (dates need not be injective).
    """

    def synthesize(sv: str, rng: Random, fake: FakerLike, _used: set[str]) -> str | None:
        return get_handler("date_of_birth").generate(sv, fake, patterns=patterns, rng=rng)

    return build_scoped_col_map(
        original_df,
        col,
        scope,
        gk,
        cfg,
        synthesize=synthesize,
        track_used=False,
    )


def build_standalone_maps(
    original_df: pd.DataFrame,
    plan: PiiReplacementPlan,
    cfg: entities.Config,
    *,
    group_key: str | None = None,
) -> dict[str, ScopedValueMap]:
    """Build a replacement map per standalone column in the plan.

    Args:
        original_df: Source dataframe.
        plan: Resolved replacement plan.
        cfg: Replacement configuration.
        group_key: Training group-key column when ``plan.scope`` is ``"group"``.

    Returns:
        Mapping from column name to ``ScopedValueMap``.
    """
    scope = plan.scope.value
    gk = group_key
    maps: dict[str, ScopedValueMap] = {}
    backend = cfg.persona_backend
    n_rows = len(original_df)
    warned_record_cost = False

    for spec in iter_standalone_specs(plan, backend):
        col = spec.column_name
        entity = spec.entity_type.value if spec.entity_type else None
        if not entity or col not in original_df.columns:
            continue
        if entities.is_identify_only(entity):
            maps[col] = ScopedValueMap("flat", {})
            logger.runtime.debug(
                f"[PII Replacement] Temporal column {col!r} passes through unchanged (entity={entity})"
            )
            continue

        if scope == "record" and not warned_record_cost and n_rows > _RECORD_SCOPE_COST_WARN_ROWS:
            logger.user.warning(
                f"[PII Replacement] scope=record with {n_rows} rows builds one standalone-"
                f"identifier map per row, which can be costly. Prefer scope=group or "
                f"scope=dataframe when per-row independence is not required."
            )
            warned_record_cost = True

        patterns = list(spec.patterns)

        match entity:
            case "date_of_birth":
                maps[col] = _build_dob_map(original_df, col, patterns, scope, gk, cfg)
            case _:
                maps[col] = _build_identifier_map(original_df, col, entity, patterns, scope, gk, cfg)

    return maps
