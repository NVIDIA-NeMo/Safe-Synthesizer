# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine config, EntitySpec registry, and derived entity taxonomy views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from ..config.replace_pii import (
    ENTITY_BY_TYPE,
    EntityAction,
    EntityType,
    ReplacePiiConfig,
)
from ..defaults import default_managed_assets_path
from ..errors import InternalError


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class Config:
    """Engine knobs for heuristic discovery.

    Sampler fields are copied from user-facing ``ReplacePiiConfig`` via
    ``config_from_replace_pii``. The rest are fixed product defaults used by
    discovery; they are not exposed in YAML/SDK today, but can be overridden
    when constructing ``Config`` directly (e.g. in tests).

    Apply-time knobs (locale, seed, LLM, free-text aliasing) live on the
    execution PR checklist in ``pii_replacement_plan_spec.md``.
    """

    sampler_backend: str = "managed"
    """Synthetic value sampler backend: ``managed`` or ``faker``.

    Discovery uses this when deciding whether to attach ``ethnic_background``
    conditioners (``faker`` omits them).
    """
    managed_assets_path: str | None = None
    """Root of managed persona parquet assets; resolved in ``__post_init__`` when unset."""
    dominant_pattern_min_coverage: float = 85.0
    """Minimum percent of non-null values matching the dominant pattern for structured columns."""
    pattern_class_max: int = 6
    """Max distinct chars for a template position before it widens to a family token.

    Smaller alphabets become an explicit class (e.g. ``[68]``); larger use
    family tokens (``#`` / ``^`` / ``@`` / …).
    """
    pattern_min_evidence_per_char: int = 4
    """Samples required per character before pinning a literal or narrowing a class.

    Below this, the position widens to its family token so a few coincidences
    cannot freeze a template (e.g. three IDs that happen to start ``PMC``).
    """
    pattern_rare_char_frac: float = 0.01
    """Drop characters covering less than this fraction of a position as noise."""
    pattern_sample_cap: int = 5000
    """Cap on distinct sample values scanned when inferring a value template."""

    def __post_init__(self) -> None:
        if self.managed_assets_path is None:
            self.managed_assets_path = str(default_managed_assets_path())


def config_from_replace_pii(config: ReplacePiiConfig) -> Config:
    """Build engine ``Config`` from the user-facing ``ReplacePiiConfig``.

    Discovery only needs sampler backend/path; apply-time fields are restored
    in the execution PR.

    Args:
        config: User-facing PII replacement configuration.

    Returns:
        Engine ``Config`` with sampler settings resolved.
    """
    return Config(
        sampler_backend=config.sampler.backend.value,
        managed_assets_path=str(config.sampler.resolved_managed_assets_path()),
    )


# ===========================================================================
# Entity routing registry
# ===========================================================================
# Engine overlay keyed by catalog ``EntityType``. Product meaning (replace vs
# identify vs propagate, pattern language, may condition) lives on
# ``config.replace_pii.Entity`` / ``ENTITY_BY_TYPE``. Detection string aliases
# (``datetime`` → ``date``, ``sex`` → ``gender``) map via
# ``ENGINE_LABEL_TO_ENTITY_TYPE``. Derived frozensets / dicts below stay for
# call-site convenience; do not hand-edit them — change EntitySpec entries instead.

ApplyPath = Literal["persona", "standalone_map"]

# Engine detection labels → plan ``EntityType`` (aliases + temporal collapse).
ENGINE_LABEL_TO_ENTITY_TYPE: dict[str, EntityType] = {
    **{e.value: e for e in EntityType},
    "sex": EntityType.gender,
    "datetime": EntityType.date,
    "time": EntityType.date,
    "duration": EntityType.date,
}


def entity_type_for_label(label: str) -> EntityType | None:
    """Map an engine detection label to a plan ``EntityType``, if any."""
    return ENGINE_LABEL_TO_ENTITY_TYPE.get(label)


@dataclass(frozen=True)
class EntitySpec:
    """Engine overlay for one catalog ``EntityType``.

    Routing for *replace* synthesis (``persona`` vs ``standalone_map``), discovery
    gates, and column-name match data. Identify-only / propagate come from
    ``ENTITY_BY_TYPE[...].action`` — do not restate them as ``apply_path`` values.
    Registry keys are ``EntityType``; detection strings like ``datetime`` map onto
    ``EntityType.date`` via ``entity_type_for_label``.
    """

    label: EntityType
    """Catalog entity type this overlay describes."""
    apply_path: ApplyPath | None = None
    """How to synthesize a replaceable column, or ``None`` when not replaced.

    ``persona``: sample from the personas dataset (conditioned via ``depends_on``).
    ``standalone_map``: template / generator path. Must be set iff the catalog
    ``action`` is ``replace`` (enforced when the registry is built).
    """
    requires_value_match: bool = False
    """Discovery: values must classify as this entity before the column is allocated."""
    name_patterns: tuple[str, ...] = ()
    """Regex fragments matched against normalized column headers."""


def _build_registry() -> dict[EntityType, EntitySpec]:
    specs: list[EntitySpec] = [
        EntitySpec(
            label=EntityType.first_name,
            apply_path="persona",
            name_patterns=(
                "first[_ ]?name",
                "^fname$",
                "given[_ ]?name",
            ),
        ),
        EntitySpec(
            label=EntityType.last_name,
            apply_path="persona",
            name_patterns=(
                "last[_ ]?name",
                "^lname$",
                "surname",
                "family[_ ]?name",
            ),
        ),
        EntitySpec(
            label=EntityType.middle_name,
            apply_path="persona",
            name_patterns=(
                "middle[_ ]?name",
                "^mname$",
            ),
        ),
        EntitySpec(
            label=EntityType.full_name,
            apply_path="persona",
            name_patterns=(
                "full[_ ]?name",
                "legal[_ ]?name",
                "^name$",
                "provider[_ ]?name",
                "physician",
                "doctor",
                "clinician",
                "\\bnurse\\b",
                "attending",
                "surgeon",
                "referr\\w*[_ ]?provider",
                "patient[_ ]?name",
                "person[_ ]?name",
                "customer[_ ]?name",
                "client[_ ]?name",
                "employee[_ ]?name",
                "subscriber[_ ]?name",
                "claimant",
                "enrollee",
                "beneficiary",
                "contact[_ ]?name",
                "primary[_ ]?contact",
                "emergency[_ ]?contact",
                "next[_ ]?of[_ ]?kin",
                "guardian[_ ]?name",
                "(?<![a-z])guardian(?![a-z_])",
                "spouse[_ ]?name",
                "(?<![a-z])spouse(?![a-z_])",
                "dependent[_ ]?name",
                "policy[_ ]?holder",
                "insured[_ ]?name",
                "(?<![a-z])insured(?![a-z_])",
                "member[_ ]?name",
                "account[_ ]?(?:holder|owner)",
                "\\bcardholder\\b",
                "applicant[_ ]?name",
                "(?<![a-z])applicant(?![a-z_])",
                "borrower[_ ]?name",
                "(?<![a-z])borrower(?![a-z_])",
                "co[_ ]?borrower",
                "\\bcosigner\\b",
                "guarantor[_ ]?name",
                "(?<![a-z])guarantor(?![a-z_])",
                "(?<![a-z])payee(?![a-z_])",
                "manager[_ ]?name",
                "supervisor[_ ]?name",
                "(?<![a-z])supervisor(?![a-z_])",
                "agent[_ ]?name",
                "attorney[_ ]?name",
                "(?<![a-z])attorney(?![a-z_])",
                "(?<![a-z])counsel(?![a-z_])",
                "witness[_ ]?name",
                "(?<![a-z])witness(?![a-z_])",
                "plaintiff[_ ]?name",
                "defendant[_ ]?name",
                "recipient[_ ]?name",
                "sender[_ ]?name",
                "passenger[_ ]?name",
                "guest[_ ]?name",
                "driver[_ ]?name",
                "student[_ ]?name",
                "teacher[_ ]?name",
                "(?<![a-z])teacher(?![a-z_])",
                "instructor[_ ]?name",
                "(?<![a-z])instructor(?![a-z_])",
            ),
        ),
        EntitySpec(
            label=EntityType.email,
            apply_path="persona",
            requires_value_match=True,
            name_patterns=("e[-_ ]?mail",),
        ),
        EntitySpec(
            label=EntityType.phone_number,
            apply_path="standalone_map",
            requires_value_match=True,
            name_patterns=(
                "(?<![a-z])phone(?![a-z])",
                "mobile",
                "telephone",
                "\\bfax\\b",
            ),
        ),
        EntitySpec(
            label=EntityType.date_of_birth,
            apply_path="standalone_map",
            name_patterns=(
                "date[_ ]?of[_ ]?birth",
                "birth[_ ]?date",
                "\\bdob\\b",
                "born[_ ]?on",
                "birth[_ ]?dt",
                "birth[_ ]?ymd",
                "fecha[_ ]?nacimiento",
                "dob[_ ]?dt",
            ),
        ),
        EntitySpec(
            label=EntityType.street_address,
            apply_path="persona",
            name_patterns=(
                "street",
                "(?<!ip)(?<!ip[_ ])address",
                "(?<![a-z])addr(?![a-z_])",
            ),
        ),
        EntitySpec(
            label=EntityType.city,
            name_patterns=(
                "^city$",
                "\\btown\\b",
            ),
        ),
        EntitySpec(
            label=EntityType.state,
            name_patterns=(
                "^state$",
                "province",
            ),
        ),
        EntitySpec(
            label=EntityType.zipcode,
            name_patterns=(
                "\\bzip\\b",
                "postcode",
                "postal",
            ),
        ),
        EntitySpec(
            label=EntityType.ssn,
            apply_path="standalone_map",
            requires_value_match=True,
            name_patterns=(
                "\\bssn\\b",
                "social[_ ]?security",
            ),
        ),
        EntitySpec(
            label=EntityType.national_id,
            apply_path="standalone_map",
            name_patterns=(
                "national[_ ]?id",
                "\\bnino\\b",
                "passport",
                "tax[_ ]?id",
                "\\bnin\\b",
                "aadhaar",
                "\\bdni\\b",
                "\\bsin\\b",
                "cedula",
                "\\bc\u00e9dula\\b",
            ),
        ),
        EntitySpec(
            label=EntityType.credit_debit_card,
            apply_path="standalone_map",
            requires_value_match=True,
            name_patterns=(
                "credit[_ ]?card",
                "debit[_ ]?card",
                "\\bcard[_ ]?(no|number|num)\\b",
                "\\bccn\\b",
                "\\bpan\\b",
            ),
        ),
        EntitySpec(
            label=EntityType.api_key,
            apply_path="standalone_map",
            requires_value_match=True,
            name_patterns=(
                "api[_ ]?key",
                "secret[_ ]?key",
                "access[_ ]?key",
                "(?<![a-z])token(?![a-z])",
                "\\bapikey\\b",
            ),
        ),
        EntitySpec(
            label=EntityType.ipv4,
            apply_path="standalone_map",
            requires_value_match=True,
            name_patterns=(
                "ipv4",
                "ip[_ ]?addr",
                "^ip$",
                "ip[_ ]?address",
            ),
        ),
        EntitySpec(
            label=EntityType.ipv6,
            apply_path="standalone_map",
            requires_value_match=True,
            name_patterns=("ipv6",),
        ),
        EntitySpec(
            label=EntityType.unique_identifier,
            apply_path="standalone_map",
            # Known limitation: columns like ``userid`` / ``valid`` are not
            # auto-discovered as unique_identifier. Use hand-plan / LLM mode.
            name_patterns=(
                r"\buuid\b",
                r"\bguid\b",
                r"\bmrn\b",
                r"(?:^|[_ ])id$",
                r"\w+[_ ]id$",
                r"identifier",
                r"(?:^|_)key$",
                r"(?:^|_)ref$",
            ),
        ),
        EntitySpec(
            label=EntityType.date,
        ),
        EntitySpec(
            label=EntityType.free_text,
        ),
    ]
    registry = {s.label: s for s in specs}
    _validate_apply_paths(registry)
    return registry


def _validate_apply_paths(registry: Mapping[EntityType, EntitySpec]) -> None:
    """Require ``persona`` / ``standalone_map`` iff catalog ``action`` is ``replace``."""
    for entity_type, entity_spec in registry.items():
        apply_path = entity_spec.apply_path
        action = ENTITY_BY_TYPE[entity_type].action
        if apply_path is not None and action is not EntityAction.replace:
            raise InternalError(
                f"EntitySpec {entity_type!r} has apply_path={apply_path!r} but "
                f"ENTITY_BY_TYPE action is {action!r}; "
                "persona/standalone_map require action=replace."
            )
        if apply_path is None and action is EntityAction.replace:
            raise InternalError(
                f"EntitySpec {entity_type!r} is replaceable but apply_path is None; set 'persona' or 'standalone_map'."
            )


ENTITY_REGISTRY: dict[EntityType, EntitySpec] = _build_registry()


def spec(label: str) -> EntitySpec | None:
    """Return the overlay for a detection label, mapping aliases to ``EntityType``."""
    entity_type = entity_type_for_label(label)
    if entity_type is None:
        return None
    return ENTITY_REGISTRY.get(entity_type)


def is_identify_only(label: str) -> bool:
    """True when ``label`` maps to catalog ``action=none`` (never on columns_to_replace)."""
    entity_type = entity_type_for_label(label)
    return entity_type is not None and ENTITY_BY_TYPE[entity_type].action is EntityAction.none


def is_propagate(label: str) -> bool:
    """True when ``label`` maps to catalog ``action=propagate`` (free-text scan)."""
    entity_type = entity_type_for_label(label)
    return entity_type is not None and ENTITY_BY_TYPE[entity_type].action is EntityAction.propagate


# Name match tables derived from the registry (detect still consumes these).
# Keys are ``EntityType.value`` strings so header matchers stay string-keyed.
ENTITY_NAME_PATTERNS: dict[str, list[str]] = {
    s.label.value: list(s.name_patterns) for s in ENTITY_REGISTRY.values() if s.name_patterns
}


# Only gender and ethnic_background are used to condition synthetic-name generation.
# Faker only conditions given names on gender, so ethnic_background is omitted for that backend.
DEMO_KEYS: tuple[Literal["gender", "ethnic_background"], ...] = ("gender", "ethnic_background")


def demo_keys_for_backend(sampler_backend: str) -> tuple[Literal["gender", "ethnic_background"], ...]:
    """Demographics that may condition replacements for this sampler backend."""
    if sampler_backend == "faker":
        return ("gender",)
    return DEMO_KEYS


# Only gender and ethnic_background condition synthetic-name generation, so those
# are the only demographics detected. (Age/DOB/occupation do not constrain sampling.)
# Header ``sex`` maps to the ``gender`` demo key (plan EntityType.gender).
DEMO_LABEL_PATTERNS: dict[str, list[str]] = {
    "gender": [r"^sex$", r"gender"],
    "ethnic_background": [r"race", r"ethnic"],
}


def sval(value: object) -> str | None:
    # pandas-stubs omit ``object`` from ``isna`` overloads; runtime accepts it.
    return None if pd.isna(value) else str(value)  # ty: ignore[no-matching-overload]


# What a column writes in the cells it has nothing for. Spelled out rather than
# left null, these are still not values: they name no person, wear no format, and
# are nobody's PII.
_MISSING_VALUE_WORDS = frozenset({"", "-", "--", "?", "na", "n/a", "nan", "nil", "none", "null", "unknown"})


def is_missing_value(value: object) -> bool:
    """Return whether a cell indicates it has no value rather than holding one.

    Such a cell is read as nothing and written back as it was. A column's formats
    are not inferred from it.

    Example:
        ``"N/A"``, ``"-"``, ``""`` -> ``True``

    Args:
        value: Cell value to evaluate.

    Returns:
        ``True`` when the value is null, blank, or a known missing-value token.
    """
    text = sval(value)
    return text is None or text.strip().lower() in _MISSING_VALUE_WORDS
