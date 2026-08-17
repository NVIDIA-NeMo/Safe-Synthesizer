# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persona sources (PGM / managed / Faker) and writing a persona into a column."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from ...errors import ParameterError
from ...observability import get_logger
from ..entities import Config, is_missing_value
from ..models import PersonaInstance
from ..patterns import (
    conform_to_template,
    handle_email_pattern,
    infer_email_pattern,
    infer_persona_pattern,
    matching_template,
    name_parts,
    placeholder_tokens,
    render_email_pattern,
    render_persona_pattern,
    split_full_name,
    split_title,
)
from .demographics import norm_sex
from .scope import FakerLike, seeded_faker

logger = get_logger(__name__)

__all__ = [
    "PersonaEngine",
    "clear_managed_person_sampler_cache",
    "load_managed_person_sampler",
    "synth_value",
]

# Process-wide cache: managed persona parquets are multi-GB once materialized.
# Multi-table runs must not reload them once per table.
_MANAGED_SAMPLER_CACHE: dict[tuple[str, str], pd.DataFrame | None] = {}


def clear_managed_person_sampler_cache() -> None:
    """Drop cached managed persona dataframes (for tests / memory recovery)."""
    _MANAGED_SAMPLER_CACHE.clear()


# ===========================================================================
# PGM persona sampling
# ===========================================================================
def _pgm_persona(row: pd.Series) -> dict:
    p = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    if p.get("postcode") in (None, "") and p.get("zipcode") not in (None, ""):
        p["postcode"] = p["zipcode"]
    return p


def _load_pgm_generator(cfg: Config):
    """Import and return sdg-pgms' ``USPersonGenerator`` (``en_US`` only).

    Raises rather than falling back: the backend decides which columns are
    persona-sourced (see ``effective_apply_path``), so degrading to another
    backend mid-run would replace columns by a method the plan never described.

    Args:
        cfg: Replacement configuration (``locale``, ``sdg_pgms_src``).

    Returns:
        An initialized ``USPersonGenerator`` instance.

    Raises:
        ParameterError: If locale is not ``"en_US"`` or sdg-pgms cannot be loaded.
    """
    import sys

    if cfg.locale != "en_US":
        raise ParameterError(
            f"replace_pii.person.backend is 'pgm', which supports locale 'en_US' only, but the locale is "
            f"{cfg.locale!r}. Set the locale to 'en_US' or choose another persona backend."
        )
    if not cfg.sdg_pgms_src:
        raise ParameterError(
            "replace_pii.person.backend is 'pgm' but replace_pii.person.sdg_pgms_src is unset. "
            "Point it at an sdg-pgms checkout, or use the 'managed' or 'faker' backend."
        )
    try:
        if cfg.sdg_pgms_src not in sys.path:
            sys.path.insert(0, cfg.sdg_pgms_src)
        if "sudachipy" not in sys.modules:
            import types as _types

            _stub = _types.ModuleType("sudachipy")

            class _StubDict:
                def __init__(self, *a, **k):
                    pass

                def create(self, *a, **k):
                    return None

            _dict_mod = _types.ModuleType("sudachipy.dictionary")
            setattr(_dict_mod, "Dictionary", _StubDict)
            setattr(_stub, "dictionary", _dict_mod)
            sys.modules["sudachipy"] = _stub
            sys.modules["sudachipy.dictionary"] = _dict_mod
        from pgms.generators.us_person_generator import USPersonGenerator

        return USPersonGenerator()
    except Exception as exc:
        raise ParameterError(
            f"replace_pii.person.backend is 'pgm' but sdg-pgms could not be loaded from "
            f"{cfg.sdg_pgms_src!r} ({exc}). The PGM backend is internal-only and needs an sdg-pgms checkout "
            "with its dependencies installed; point replace_pii.person.sdg_pgms_src at one, or use the "
            "'managed' or 'faker' backend."
        ) from exc


class PgmPersonaPool:
    """Growable pool of fresh PGM personas with without-replacement matching.

    Persona names are conditioned (in the PGM) on sex and ethnic_background only;
    age, DOB, and occupation are not used for matching.
    """

    def __init__(self, generator, seed: int, initial: int):
        self._gen = generator
        self._rng = np.random.default_rng(seed)
        self._pool: list[dict] = []
        # Vectorized columns (rebuilt on grow) for fast candidate filtering. Persona
        # names are conditioned (in the PGM) on sex + ethnic_background only, so those
        # are the only attributes we match on -- age/DOB/occupation are NOT used.
        self._sex_arr = np.array([], dtype=object)
        self._eth_arr = np.array([], dtype=object)
        self._used = np.zeros(0, dtype=bool)
        self._grow(initial)

    def _grow(self, n: int) -> None:
        df = self._gen.generate_samples(int(max(1, n)))
        new = [_pgm_persona(r) for _, r in df.iterrows()]
        sex, eth = [], []
        for p in new:
            self._pool.append(p)
            sex.append(norm_sex(p.get("sex")) or "")
            eth.append(str(p.get("ethnic_background", "")).strip().lower())
        self._sex_arr = np.concatenate([self._sex_arr, np.array(sex, dtype=object)])
        self._eth_arr = np.concatenate([self._eth_arr, np.array(eth, dtype=object)])
        self._used = np.concatenate([self._used, np.zeros(len(new), dtype=bool)])

    def __len__(self) -> int:
        return len(self._pool)

    def _candidates(self, sex, eth_set, use_sex, use_eth) -> np.ndarray:
        mask = ~self._used
        if use_sex and sex:
            mask &= self._sex_arr == sex
        if use_eth and eth_set:
            mask &= np.isin(self._eth_arr, list(eth_set))
        return np.flatnonzero(mask)

    def match_one(self, sex, sfv) -> dict:
        eth_list = (sfv or {}).get("ethnic_background")
        eth_set = {str(e).strip().lower() for e in eth_list} if eth_list else None
        # Prefer sex+race, then relax to sex-only, race-only, then anything.
        relax = [(True, True), (True, False), (False, True), (False, False)]
        for _ in range(3):
            for flags in relax:
                idxs = self._candidates(sex, eth_set, *flags)
                if idxs.size:
                    pick = int(self._rng.choice(idxs))
                    self._used[pick] = True
                    return self._pool[pick]
            self._grow(1000)
        idxs = np.flatnonzero(~self._used)
        if not idxs.size:
            self._grow(1000)
            idxs = np.flatnonzero(~self._used)
        pick = int(self._rng.choice(idxs))
        self._used[pick] = True
        return self._pool[pick]


def _constraint_signature(
    inst: PersonaInstance,
) -> tuple[str | None, tuple[tuple[str, tuple[str, ...]], ...]]:
    sfv = inst.select_field_values
    sfv_key = tuple(sorted((k, tuple(v)) for k, v in (sfv or {}).items()))
    return (inst.sex, sfv_key)


# ===========================================================================
# Managed / Faker persona sources
# ===========================================================================
def load_managed_person_sampler(assets_root: str, locale: str) -> pd.DataFrame | None:
    """Load persona parquet assets without data_designer.

    Results are cached by resolved ``(assets_root, locale)`` so multi-table (and
    repeated single-table) runs share one in-memory copy instead of reloading a
    multi-GB parquet each time.

    Args:
        assets_root: Root directory containing ``datasets/{locale}.parquet``.
        locale: Locale subdirectory name (for example ``"en_US"``).

    Returns:
        Persona dataframe, or ``None`` if the asset is missing or unreadable.
    """
    root = Path(assets_root)
    try:
        cache_key = (str(root.resolve()), locale)
    except OSError:
        cache_key = (str(root), locale)
    if cache_key in _MANAGED_SAMPLER_CACHE:
        return _MANAGED_SAMPLER_CACHE[cache_key]

    path = root / "datasets" / f"{locale}.parquet"
    if not path.exists():
        _MANAGED_SAMPLER_CACHE[cache_key] = None
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.runtime.warning(f"Managed assets unreadable at {path} ({exc}); using Faker person sampler.")
        _MANAGED_SAMPLER_CACHE[cache_key] = None
        return None
    _MANAGED_SAMPLER_CACHE[cache_key] = df
    logger.user.info(
        f"[PII Replacement] Loaded managed persona assets for locale {locale!r} "
        f"from {path} ({len(df)} rows); cached for reuse"
    )
    return df


def _row_to_persona(row: pd.Series) -> dict:
    return _pgm_persona(row)


def _faker_persona(fake: FakerLike, sex: str | None = None) -> dict[str, object]:
    # Condition the given name on sex so the synthetic name matches the source
    # demographic; fall back to an unconditioned name when sex is unknown.
    match sex:
        case "Male":
            first_name, resolved_sex = fake.first_name_male(), "Male"
        case "Female":
            first_name, resolved_sex = fake.first_name_female(), "Female"
        case _:
            first_name, resolved_sex = fake.first_name(), fake.random_element(["Male", "Female"])
    return {
        "first_name": first_name,
        "last_name": fake.last_name(),
        "sex": resolved_sex,
        "ethnic_background": "white",
        "email_address": fake.email(),
        "phone_number": fake.phone_number(),
        "birth_date": fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat(),
        "street_number": fake.building_number(),
        "street_name": fake.street_name(),
        "city": fake.city(),
        "state": fake.state_abbr(),
        "postcode": fake.postcode(),
    }


class PersonaEngine:
    """Assign one synthetic persona per person instance.

    Selects and initializes the persona backend (PGM, managed parquet, or Faker)
    from ``cfg.persona_backend``. An unavailable PGM backend is fatal because the
    plan is routed on the configured backend; managed degrades to Faker when assets
    are missing.

    Args:
        cfg: Replacement configuration.
        n_instances: Number of persona instances to support (drives pool sizing).
    """

    def __init__(self, cfg: Config, n_instances: int):
        self.cfg = cfg
        self.backend = cfg.persona_backend if n_instances else "none"
        self.pgm_pool: PgmPersonaPool | None = None
        self.managed_df: pd.DataFrame | None = None
        self.source_counts: Counter = Counter()
        # An unavailable PGM is fatal (see _load_pgm_generator) because the
        # plan is routed on the configured backend. Managed still degrades to
        # Faker: the two supply the same attributes, so only name realism differs.
        match self.backend:
            case "pgm":
                gen = _load_pgm_generator(self.cfg)
                pool_n = max(cfg.pool_min_size, cfg.pool_oversample * max(1, n_instances))
                self.pgm_pool = PgmPersonaPool(gen, cfg.random_seed, pool_n)
            case "managed":
                if cfg.managed_assets_path:
                    self.managed_df = load_managed_person_sampler(cfg.managed_assets_path, cfg.locale)
                if self.managed_df is None:
                    searched = cfg.managed_assets_path or "an unset replace_pii.person.managed_assets_path"
                    logger.user.warning(
                        f"Managed persona assets for locale {cfg.locale!r} were not found under {searched}; "
                        "generating personas with Faker instead, which replaces the same columns with less "
                        "realistic names. Ethnic-background matching is not available under Faker "
                        "(names are conditioned on sex only)."
                    )
                    self.backend = "faker"
            case _:
                pass

    def _sample_managed(self, n: int, seed: int, sex: str | None = None, sfv: dict | None = None) -> list[dict]:
        df = self.managed_df
        if df is None or df.empty:
            return []
        eth = None
        if sfv:
            vals = sfv.get("ethnic_background")
            if vals:
                eth = {str(v).strip().lower() for v in vals}
        # Prefer matching sex+ethnicity, then relax to sex-only, ethnicity-only, and
        # finally the whole pool -- mirroring the PGM pool's matching (names are
        # conditioned on sex + ethnic_background only).
        sub = df
        for use_sex, use_eth in ((True, True), (True, False), (False, True), (False, False)):
            sub = df
            if use_sex and sex and "sex" in sub.columns:
                sub = sub[sub["sex"] == sex]
            if use_eth and eth and "ethnic_background" in sub.columns:
                sub = sub[sub["ethnic_background"].astype(str).str.strip().str.lower().isin(eth)]
            if not sub.empty:
                break
        if sub.empty:
            sub = df
        sample = sub.sample(n=n, replace=n > len(sub), random_state=seed)
        return [_row_to_persona(row) for _, row in sample.iterrows()]

    def _sample_faker(self, n: int, seed: int, sex: str | None = None) -> list[dict]:
        fake = seeded_faker(seed, self.cfg.locale)
        return [_faker_persona(fake, sex) for _ in range(n)]

    def assign(self, instances: list[PersonaInstance]) -> None:
        if not instances:
            return
        if self.pgm_pool is not None:
            for inst in instances:
                inst.synthetic_person = self.pgm_pool.match_one(inst.sex, inst.select_field_values)
                inst.synthetic_person_source = "pgm"
                self.source_counts["pgm"] += 1
            return

        # Bucket instances by their demographic constraint so every instance in a
        # bucket is sampled from the matching sub-population. Faker only conditions
        # on sex; ignore ethnicity even if the plan listed ethnic_background (e.g.
        # hand-authored plan, or managed→Faker fallback with a race column present).
        buckets: dict[tuple[str | None, tuple[tuple[str, tuple[str, ...]], ...]], list[int]] = {}
        for idx, inst in enumerate(instances):
            match self.backend:
                case "faker":
                    key = (inst.sex, ())
                case _:
                    key = _constraint_signature(inst)
            buckets.setdefault(key, []).append(idx)
        for b_idx, (sig, idxs) in enumerate(buckets.items()):
            sex, sfv_key = sig
            sfv = {k: list(v) for k, v in sfv_key} or None
            seed = self.cfg.random_seed + b_idx
            if self.backend == "managed" and self.managed_df is not None:
                personas = self._sample_managed(len(idxs), seed, sex, sfv)
                source = "managed"
            else:
                personas = self._sample_faker(len(idxs), seed, sex)
                source = "faker"
            if len(personas) < len(idxs):
                personas.extend(self._sample_faker(len(idxs) - len(personas), seed + 1, sex))
                source = "faker"
            self.source_counts[source] += len(idxs)
            for inst_idx, persona in zip(idxs, personas):
                instances[inst_idx].synthetic_person = persona
                instances[inst_idx].synthetic_person_source = source


# ===========================================================================
# Writing a persona into a column
# ===========================================================================
def format_persona_phone(number: str, original: str, patterns: Sequence[str] | None, fake: FakerLike | None) -> str:
    """Format the persona's phone number to match the column's convention.

    The generator gives a number in its own format (``"(206) 555-0181"``), which would
    otherwise land in a column that writes ``"+1-206-555-0181"``. Its digits are kept,
    since under the PGM the area code follows the persona's address; only the
    punctuation around them changes.

    Args:
        number: Persona phone number from the sampler.
        original: Original cell value (selects which plan pattern applies).
        patterns: Column format templates from the plan.
        fake: Seeded ``Faker`` instance (for ``conform_to_template``).

    Returns:
        Phone string formatted to match the column's pattern.

    Example:
        Original ``"+1-206-555-0100"``, persona ``"(415) 555-0181"``,
        pattern ``"+1-###-###-####"`` -> ``"+1-415-555-0181"``.
    """
    pats = list(patterns or [])
    if not pats or fake is None:
        return number
    return conform_to_template(number, matching_template(original, pats), fake.random)


def given_name(persona: Mapping[str, object], fake: FakerLike) -> str:
    """Return a given name matching the persona's sex.

    Args:
        persona: Sampled persona dict (may include ``"sex"``).
        fake: Seeded ``Faker`` instance.

    Returns:
        A first name conditioned on ``persona["sex"]`` when known.

    Example:
        ``sex="Female"`` -> ``fake.first_name_female()``.
    """
    match persona.get("sex"):
        case "Male":
            return fake.first_name_male()
        case "Female":
            return fake.first_name_female()
        case _:
            return fake.first_name()


def wants_middle_name(inst: PersonaInstance) -> bool:
    """Return whether this instance needs a synthetic middle name.

    True when a ``middle_name`` field exists or a pattern uses ``{M}`` / ``{Middle}``.

    Args:
        inst: Persona instance with ``field_cols`` and ``patterns_by_label``.

    Returns:
        ``True`` if a middle name should be drawn for this instance.
    """
    if "middle_name" in inst.field_cols:
        return True
    return any(
        token.lower() in ("m", "middle")
        for pattern_list in inst.patterns_by_label.values()
        for pattern in pattern_list or []
        for token in placeholder_tokens(pattern)
    )


def persona_written(
    label: str,
    original: str,
    persona: Mapping[str, object],
    patterns: Sequence[str] | None,
    originals: Mapping[str, object] | None = None,
    rng: random.Random | None = None,
) -> str | None:
    """Write the persona's name or address the way this column writes them.

    A column that reads ``"SMITH, John"`` or ``"j.smith@acme.com"`` says how it
    assembles a person, and the replacement follows it rather than imposing
    ``"Robert Jones"``. Each value is written the way it is already written, so a
    column of several conventions keeps all of them.

    An address is always written from the value itself, whether or not its column
    named a convention: one that reads as a person is written as that person, and
    one that reads as a handle is generated from its shape, keeping its domain
    either way.

    Args:
        label: Persona-sourced field label (for example ``"full_name"``).
        original: Original cell value.
        persona: Sampled persona dict.
        patterns: Column format templates from the plan.
        originals: Person's other original values (needed for email/local-part inference).
        rng: Optional random source for pattern rendering.

    Returns:
        Replacement string, or ``None`` when no pattern applies.

    Example:
        ``label="full_name"``, ``original="SMITH, Jane"``, persona Robert/Jones
        -> ``"JONES, Robert"`` via inferred ``"{LAST}, {First}"``.
    """
    pats = list(patterns or [])
    if label == "email":
        parts = name_parts(originals or {})
        own = infer_email_pattern(original, parts) if parts else None
        return render_email_pattern(
            own or handle_email_pattern(original, pats),
            cast(Mapping[str, str], persona),
            original,
            rng,
        )
    if not pats:
        return None

    title, rest = split_title(original) if label == "full_name" else (None, original)
    parts = split_full_name(rest) if label == "full_name" else {label: rest}
    own = infer_persona_pattern(rest, parts)
    written = render_persona_pattern(own or pats[0], cast(Mapping[str, str], persona), rng=rng)
    if written is None:
        return None
    return f"{title} {written}" if title else written


def synth_value(
    label: str,
    original: str,
    persona: Mapping[str, object],
    fake: FakerLike | None = None,
    patterns: Sequence[str] | None = None,
    originals: Mapping[str, object] | None = None,
) -> str | None:
    """Map a sampled persona onto one persona-sourced field value.

    Only persona-sourced fields are handled here. Entity-driven columns
    (``unique_identifier``, ``date_of_birth``, ...) are replaced via the standalone path
    and never reach this function. Per-label writes live on
    ``get_handler(label).persona_value``.

    Args:
        label: Persona-sourced field label.
        original: Original cell value.
        persona: Sampled persona dict.
        fake: Optional seeded ``Faker`` instance.
        patterns: Column format templates for fields that carry one.
        originals: Person's other original values (used for email/local-part inference).

    Returns:
        Synthetic cell value, or ``None`` to leave the cell unchanged.

    Example:
        ``original="N/A"`` -> ``None``;
        ``original="Jane"``, ``label="first_name"`` -> persona's first name.
    """
    if is_missing_value(original):
        return None
    # Deferred: handlers import persona helpers from this module.
    from ..entity_handlers import get_handler

    return get_handler(label).persona_value(
        str(original),
        persona or {},
        patterns=patterns,
        originals=originals,
        fake=fake,
    )
