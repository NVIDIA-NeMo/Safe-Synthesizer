# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine config, EntitySpec registry, and derived entity taxonomy views."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING, Literal, Protocol

import pandas as pd

if TYPE_CHECKING:
    from .replacement.scope import FakerLike

from ..config.pii_replacement import ReplacePiiConfig
from ..defaults import default_managed_assets_path


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class Config:
    """All knobs for detection + generation. Only sensible defaults required."""

    locale: str = "en_US"
    # Seed for persona/ID/faker generation. Env-overridable so a batched run can give
    # each row-batch a distinct seed -- otherwise every batch regenerates the SAME
    # unique-identifier sequence and they collide across batches (breaking injectivity).
    random_seed: int = field(default_factory=lambda: int(os.environ.get("PERSON_RANDOM_SEED", "42") or "42"))

    # Grouping (Safe-Synthesizer's group_training_records_by) is configured, not
    # auto-detected. A column is GROUP-level when single-valued within >= this
    # fraction of groups.
    group_constancy_threshold: float = 0.95

    # Persona sampling.
    persona_backend: str = "managed"  # pgm | managed | faker
    sdg_pgms_src: str = "/root/sdg-pgms/src"
    managed_assets_path: str | None = None
    pool_min_size: int = 3_000
    pool_oversample: int = 6

    # --- Structural / value-pattern detection ---
    # Minimum percent of non-null values matching the dominant concrete pattern
    # required to classify a column as structured (not free text).
    dominant_pattern_min_coverage: float = 85.0
    # Free-text columns: long, varied object columns.
    free_text_min_len: float = 25.0
    free_text_min_unique_ratio: float = 0.3
    # A free-text column must read like natural-language PROSE: at least this many
    # whitespace-separated tokens on average. Used to reject single-token columns
    # such as URLs or short code columns (avg ~1 word). The length +
    # unique_ratio gate above already excludes low-cardinality phrase columns.
    free_text_min_words: float = 1.5
    # Fuzzy column-name match acceptance.
    name_fuzzy_threshold: float = 0.86
    # When True, discovery/apply call the injected PiiEnhancer (stub raises until LLM lands).
    llm_enhancement: bool = False
    # Reserved inference settings propagated from ReplacePiiConfig.llm.
    llm_model_provider: str | None = None
    llm_max_workers: int = 64

    # --- Free-text name-token aliasing (BOTH modes) ---
    # When a person is identified only by a full name (no separate first/last columns),
    # also propagate the individual name TOKENS into free text so honorific/partial
    # mentions are caught consistently (e.g. provider "John Smith" -> synthetic "Robert
    # Jones" also rewrites a later "Dr. Smith" -> "Dr. Jones"). Tokens shorter than
    # freetext_alias_min_token_len are skipped to avoid over-matching short common words.
    freetext_name_token_aliases: bool = True
    freetext_alias_min_token_len: int = 3

    # --- Value-pattern inference (Faker template) ---
    # Identifier and phone columns are regenerated from an inferred template that
    # keeps constant characters literal (e.g. a 'pmc-' prefix) and constrains
    # low-entropy positions to their observed alphabet (e.g. first digit in {6,8})
    # instead of fully randomizing every character.
    # A variable position whose observed alphabet is <= this many distinct chars is
    # emitted as an explicit class (e.g. "[68]"); larger -> a family token (#/^/@/&/%/*).
    pattern_class_max: int = 6
    # Values required per character before a position may be pinned to a literal or
    # narrowed to a class. Below it the position widens to its family token, so a
    # handful of samples cannot freeze a coincidence (three IDs that happen to start
    # "PMC") into every replacement.
    pattern_min_evidence_per_char: int = 4
    # Characters covering < this fraction of a position are dropped as noise so a
    # rare outlier (e.g. a single 'pmc-7...') doesn't widen the template.
    pattern_rare_char_frac: float = 0.01
    # Cap on distinct sample values scanned when inferring a template.
    pattern_sample_cap: int = 5000

    def __post_init__(self) -> None:
        if self.managed_assets_path is None:
            self.managed_assets_path = str(default_managed_assets_path())


def config_from_replace_pii(config: ReplacePiiConfig) -> Config:
    """Build engine ``Config`` from the user-facing ``ReplacePiiConfig``.

    Only a handful of fields come from the user; the rest keep ``Config`` defaults.

    Args:
        config: User-facing PII replacement configuration.

    Returns:
        Engine ``Config`` with locale, seed, persona backend, and LLM settings resolved.
    """
    seed = config.replacement.seed
    if seed is None:
        seed = int(os.environ.get("PERSON_RANDOM_SEED", "42") or "42")
    return Config(
        locale=config.replacement.locale,
        random_seed=seed,
        persona_backend=config.person.backend.value,
        sdg_pgms_src=config.person.sdg_pgms_src,
        managed_assets_path=str(config.person.resolved_managed_assets_path()),
        llm_enhancement=config.llm_enhancement,
        llm_model_provider=config.llm.model_provider,
        llm_max_workers=config.llm.max_workers,
    )


# ===========================================================================
# Entity routing registry
# ===========================================================================
# Single source of truth for apply path, pattern language, and column-name match
# data. Derived frozensets / dicts below stay for call-site convenience; do not
# hand-edit them — change EntitySpec entries instead.

ApplyPath = Literal["persona", "standalone_map", "identify_only", "free_text"]
PatternKind = Literal["none", "strftime", "template", "persona_placeholder"]


@dataclass(frozen=True)
class EntitySpec:
    """Product rules for one engine entity label.

    Single source of truth for routing, pattern language, discovery gates, and
    column-name match data. Call sites read this registry (via helpers) instead
    of maintaining parallel label frozensets that can drift.
    """

    label: str
    """Engine entity name (e.g. ``first_name``, ``ssn``)."""
    apply_path: ApplyPath
    """Default fill channel: persona, standalone map, identify-only, or free-text.

    Resolved with ``effective_apply_path`` (may override per backend). Independent
    of which YAML plan section lists the column.
    """
    pattern_kind: PatternKind = "none"
    """How plan patterns are interpreted for this entity (strftime, template, …)."""
    entity_driven: bool = False
    """If True, apply uses entity type for generation, not YAML section placement."""
    persona_only_backends: frozenset[str] | None = None
    """Backends for which ``apply_path`` is forced to ``persona``; ``None`` means none."""
    persona_field: bool = False
    """Whether this label is a field on a persona record (name, email, address, …)."""
    valid_form: bool = False
    """If True, replacements must pass a format validator (e.g. IP addresses)."""
    requires_value_match: bool = False
    """Discovery: values must classify as this entity before the column is allocated."""
    name_shape_gates: bool = False
    """Discovery: reject multi-person or org-shaped samples (person-name labels)."""
    transform_method: str | None = None
    """Stats/report method label (e.g. ``propagation``, ``perturbation``); ``None`` if N/A."""
    name_patterns: tuple[str, ...] = ()
    """Regex fragments matched against normalized column headers."""
    fuzzy_keywords: tuple[str, ...] = ()
    """Tokens used for fuzzy header matching when exact regexes miss."""
    role_strip_tokens: tuple[str, ...] = ()
    """Extra column-name tokens stripped when deriving a persona role.

    Label path segments (``first_name`` → ``first``, ``name``) are always
    included by ``ROLE_STRIP_TOKENS``. Use this for aliases that are not in the
    label (``fname``, ``dob``, ``telephone``, …). Do not list role words that
    appear in ``name_patterns`` (``patient``, ``provider``, …).
    """


def _build_registry() -> dict[str, EntitySpec]:
    specs: list[EntitySpec] = [
        EntitySpec(
            label="first_name",
            apply_path="persona",
            pattern_kind="persona_placeholder",
            persona_field=True,
            name_shape_gates=True,
            name_patterns=(
                "first[_ ]?name",
                "^fname$",
                "given[_ ]?name",
            ),
            fuzzy_keywords=(
                "firstname",
                "fname",
                "givenname",
                "forename",
            ),
            role_strip_tokens=("given", "forename", "fname", "names"),
        ),
        EntitySpec(
            label="last_name",
            apply_path="persona",
            pattern_kind="persona_placeholder",
            persona_field=True,
            name_shape_gates=True,
            name_patterns=(
                "last[_ ]?name",
                "^lname$",
                "surname",
                "family[_ ]?name",
            ),
            fuzzy_keywords=(
                "lastname",
                "lname",
                "surname",
                "familyname",
            ),
            role_strip_tokens=("family", "surname", "lname", "names"),
        ),
        EntitySpec(
            label="middle_name",
            apply_path="persona",
            pattern_kind="persona_placeholder",
            persona_field=True,
            name_shape_gates=True,
            name_patterns=(
                "middle[_ ]?name",
                "^mname$",
            ),
            fuzzy_keywords=(
                "middlename",
                "mname",
            ),
            role_strip_tokens=("mname", "names"),
        ),
        EntitySpec(
            label="full_name",
            apply_path="persona",
            pattern_kind="persona_placeholder",
            persona_field=True,
            name_shape_gates=True,
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
            fuzzy_keywords=(
                "fullname",
                "patientname",
                "providername",
                "personname",
                "customername",
                "clientname",
                "employeename",
                "physicianname",
                "legalname",
                "subscribername",
                "claimant",
                "enrollee",
                "beneficiary",
                "surgeonname",
                "attendingname",
                "contactname",
                "primarycontact",
                "emergencycontact",
                "nextofkin",
                "guardianname",
                "spousename",
                "dependentname",
                "policyholder",
                "insuredname",
                "membername",
                "accountholder",
                "accountowner",
                "cardholder",
                "applicantname",
                "borrowername",
                "coborrower",
                "cosigner",
                "guarantorname",
                "managername",
                "supervisorname",
                "agentname",
                "attorneyname",
                "witnessname",
                "plaintiffname",
                "defendantname",
                "recipientname",
                "sendername",
                "passengername",
                "guestname",
                "drivername",
                "studentname",
                "teachername",
                "instructorname",
            ),
            role_strip_tokens=("names",),
        ),
        EntitySpec(
            label="email",
            apply_path="persona",
            pattern_kind="persona_placeholder",
            persona_field=True,
            requires_value_match=True,
            name_patterns=("e[-_ ]?mail",),
            fuzzy_keywords=(
                "email",
                "emailaddress",
                "emailaddr",
            ),
            role_strip_tokens=("mail",),
        ),
        EntitySpec(
            label="phone_number",
            apply_path="persona",
            pattern_kind="template",
            persona_only_backends=frozenset({"pgm"}),
            persona_field=True,
            requires_value_match=True,
            name_patterns=(
                "(?<![a-z])phone(?![a-z])",
                "mobile",
                "telephone",
                "\\bfax\\b",
            ),
            fuzzy_keywords=(
                "phonenumber",
                "telephone",
                "mobilephone",
                "phoneno",
            ),
            role_strip_tokens=("telephone", "mobile", "cell", "fax", "num", "no"),
        ),
        EntitySpec(
            label="date_of_birth",
            apply_path="standalone_map",
            pattern_kind="strftime",
            entity_driven=True,
            persona_field=True,
            transform_method="perturbation",
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
            fuzzy_keywords=(
                "dateofbirth",
                "birthdate",
                "birthday",
                "bornon",
                "birthdt",
                "birthymd",
                "fechanacimiento",
                "dobdt",
            ),
            role_strip_tokens=("dob", "birthday", "born"),
        ),
        EntitySpec(
            label="street_address",
            apply_path="persona",
            persona_field=True,
            name_patterns=(
                "street",
                "(?<!ip)(?<!ip[_ ])address",
                "(?<![a-z])addr(?![a-z_])",
            ),
            fuzzy_keywords=(
                "streetaddress",
                "homeaddress",
                "mailingaddress",
            ),
            role_strip_tokens=("addr",),
        ),
        EntitySpec(
            label="city",
            apply_path="identify_only",
            name_patterns=(
                "^city$",
                "\\btown\\b",
            ),
            fuzzy_keywords=(
                "city",
                "town",
                "cityname",
            ),
            role_strip_tokens=("town",),
        ),
        EntitySpec(
            label="state",
            apply_path="identify_only",
            name_patterns=(
                "^state$",
                "province",
            ),
            fuzzy_keywords=(
                "state",
                "province",
                "statename",
            ),
            role_strip_tokens=("province",),
        ),
        EntitySpec(
            label="zipcode",
            apply_path="identify_only",
            name_patterns=(
                "\\bzip\\b",
                "postcode",
                "postal",
            ),
            fuzzy_keywords=(
                "zipcode",
                "zip",
                "postalcode",
                "postcode",
            ),
            role_strip_tokens=("zip", "postal", "postcode"),
        ),
        EntitySpec(
            label="ssn",
            apply_path="standalone_map",
            entity_driven=True,
            requires_value_match=True,
            name_patterns=(
                "\\bssn\\b",
                "social[_ ]?security",
            ),
            fuzzy_keywords=(
                "socialsecurity",
                "socialsecuritynumber",
            ),
            role_strip_tokens=("social",),
        ),
        EntitySpec(
            label="national_id",
            apply_path="standalone_map",
            entity_driven=True,
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
            fuzzy_keywords=(
                "nationalid",
                "passportnumber",
                "taxid",
                "aadhaar",
                "dni",
                "cedula",
            ),
        ),
        EntitySpec(
            label="credit_debit_card",
            apply_path="standalone_map",
            pattern_kind="template",
            entity_driven=True,
            requires_value_match=True,
            name_patterns=(
                "credit[_ ]?card",
                "debit[_ ]?card",
                "\\bcard[_ ]?(no|number|num)\\b",
                "\\bccn\\b",
                "\\bpan\\b",
            ),
            fuzzy_keywords=(
                "creditcard",
                "debitcard",
                "cardnumber",
            ),
            role_strip_tokens=("card", "ccn", "pan", "num", "no"),
        ),
        EntitySpec(
            label="api_key",
            apply_path="standalone_map",
            pattern_kind="template",
            entity_driven=True,
            name_patterns=(
                "api[_ ]?key",
                "secret[_ ]?key",
                "access[_ ]?key",
                "(?<![a-z])token(?![a-z])",
                "\\bapikey\\b",
            ),
            fuzzy_keywords=(
                "apikey",
                "secretkey",
                "accesskey",
                "apitoken",
                "accesstoken",
                "secrettoken",
            ),
            role_strip_tokens=("secret", "token", "access"),
        ),
        EntitySpec(
            label="ipv4",
            apply_path="standalone_map",
            entity_driven=True,
            valid_form=True,
            requires_value_match=True,
            name_patterns=(
                "ipv4",
                "ip[_ ]?addr",
                "^ip$",
                "ip[_ ]?address",
            ),
            fuzzy_keywords=(
                "ipaddress",
                "ipaddr",
            ),
            role_strip_tokens=("ip", "addr"),
        ),
        EntitySpec(
            label="ipv6",
            apply_path="standalone_map",
            entity_driven=True,
            valid_form=True,
            requires_value_match=True,
            name_patterns=("ipv6",),
            fuzzy_keywords=(
                "ipv6",
                "ipv6address",
            ),
            role_strip_tokens=("ip",),
        ),
        EntitySpec(
            label="unique_identifier",
            apply_path="standalone_map",
            pattern_kind="template",
            entity_driven=True,
            name_patterns=(
                "\\buuid\\b",
                "\\bguid\\b",
                "\\bmrn\\b",
                "\\b\\w*_?id$",
                "identifier",
                "(?:^|_)key$",
                "(?:^|_)ref$",
            ),
            fuzzy_keywords=(
                "uniqueidentifier",
                "mrn",
            ),
            role_strip_tokens=("uuid", "guid", "id", "key", "ref", "mrn"),
        ),
        EntitySpec(
            label="date",
            apply_path="identify_only",
        ),
        EntitySpec(
            label="datetime",
            apply_path="identify_only",
        ),
        EntitySpec(
            label="time",
            apply_path="identify_only",
        ),
        EntitySpec(
            label="duration",
            apply_path="identify_only",
        ),
        EntitySpec(
            label="free_text",
            apply_path="free_text",
            transform_method="propagation",
        ),
    ]
    return {s.label: s for s in specs}


ENTITY_REGISTRY: dict[str, EntitySpec] = _build_registry()


def spec(label: str) -> EntitySpec | None:
    """Return the registry entry for ``label``, or None if unknown."""
    return ENTITY_REGISTRY.get(label)


def effective_apply_path(label: str, persona_backend: str) -> ApplyPath | None:
    """Return the apply path for ``label`` under the configured persona backend.

    Selects the generation channel: ``persona`` uses persona ``synth_value``;
    ``standalone_map`` uses scoped maps or ``fake_value`` (or DOB perturbation);
    ``identify_only`` and ``free_text`` skip structured persona fill.

    Example:
        ``effective_apply_path("phone_number", "pgm")`` -> ``"persona"``
        ``effective_apply_path("phone_number", "faker")`` -> ``"standalone_map"``

    Args:
        label: Engine entity label.
        persona_backend: Persona sampling backend (``managed``, ``pgm``, ``faker``).

    Returns:
        Resolved apply path, or ``None`` when ``label`` is unknown.
    """
    s = ENTITY_REGISTRY.get(label)
    if s is None:
        return None
    if s.persona_only_backends is not None and persona_backend not in s.persona_only_backends:
        return "standalone_map"
    if s.entity_driven:
        return "standalone_map"
    return s.apply_path


def is_identify_only(label: str) -> bool:
    """True when ``label`` is discovered/validated but never replaced."""
    s = ENTITY_REGISTRY.get(label)
    return s is not None and s.apply_path == "identify_only"


# ===========================================================================
# Entity behavior protocol
# ===========================================================================
class EntityHandler(Protocol):
    """What one entity label does, as opposed to what ``EntitySpec`` declares it is.

    ``EntitySpec`` is data: routing, pattern language, discovery gates. This is
    the matching attach point for the behavior those gates and channels invoke,
    so a rule that today lives in an ``if label == ...`` branch has one place to
    move to. Implementations live in ``entity_handlers.py`` and delegate to the
    shared detection / pattern / replacement functions; look one up with
    ``entity_handlers.get_handler(label)``.
    """

    label: str
    """Engine entity name this handler speaks for."""

    def match_value(self, value: object, *, phone_min_digits: int = 10) -> str | None:
        """Return this handler's entity label when ``value`` matches, else ``None``.

        Args:
            value: Cell value to classify.
            phone_min_digits: Minimum digit count for phone matches.

        Returns:
            ``self.label`` on a match, otherwise ``None``.
        """
        ...

    def skip_reason(self, series: pd.Series, value_entity: str | None, apply_path: str) -> str | None:
        """Return why discovery must not allocate this name-matched column.

        Args:
            series: Column values under consideration.
            value_entity: Entity inferred from values, if any.
            apply_path: Effective apply path for this label.

        Returns:
            A human-readable skip reason, or ``None`` when the column may allocate.
        """
        ...

    def generate(
        self,
        original: str,
        fake: FakerLike,
        *,
        patterns: Sequence[str] | None = None,
        rng: Random | None = None,
    ) -> str | None:
        """Return a synthetic stand-in for one standalone value.

        When ``patterns`` are given, the replacement is written in the template
        that describes ``original``; otherwise it is drawn from ``fake``.

        Args:
            original: Original cell text to replace.
            fake: Seeded Faker (or compatible) instance.
            patterns: Optional column format templates.
            rng: Optional RNG; defaults to ``fake.random`` when omitted.

        Returns:
            The synthetic value, or ``None`` when none can be made.
        """
        ...


# Name / fuzzy match tables derived from the registry (detect still consumes these).
ENTITY_NAME_PATTERNS: dict[str, list[str]] = {
    s.label: list(s.name_patterns) for s in ENTITY_REGISTRY.values() if s.name_patterns
}
FUZZY_KEYWORDS: dict[str, list[str]] = {
    s.label: list(s.fuzzy_keywords) for s in ENTITY_REGISTRY.values() if s.fuzzy_keywords
}

# Function words dropped when splitting entity labels into role-strip tokens
# (``date_of_birth`` → ``date``, ``birth``, not ``of``).
_ROLE_LABEL_STOPWORDS = frozenset({"of", "the", "a", "an", "and", "or", "to", "for"})

# Weak column qualifiers that are never persona roles.
_ROLE_WEAK_QUALIFIERS = frozenset(
    {
        "primary",
        "secondary",
        "main",
        "other",
        "alt",
        "alternate",
        "additional",
    }
)

# Demographic column tokens (``match_persona_by`` sources), plus common aliases.
_ROLE_DEMO_STRIP_TOKENS = frozenset(
    {
        "sex",
        "gender",
        "race",
        "ethnic",
        "ethnicity",
        "background",
    }
)


def _build_role_strip_tokens() -> frozenset[str]:
    """Suffix lexicon for persona-role derivation from column names.

    Built from entity labels + per-entity ``role_strip_tokens`` aliases, demographic
    tokens, and weak qualifiers. Deliberately does **not** harvest ``name_patterns`` /
    ``fuzzy_keywords``, which include role words (``patient``, ``provider``, …).
    """
    tokens: set[str] = set(_ROLE_WEAK_QUALIFIERS) | set(_ROLE_DEMO_STRIP_TOKENS)
    for s in ENTITY_REGISTRY.values():
        if s.apply_path == "free_text":
            continue
        tokens.update(part for part in s.label.split("_") if part and part not in _ROLE_LABEL_STOPWORDS)
        tokens.update(s.role_strip_tokens)
    return frozenset(tokens)


ROLE_STRIP_TOKENS: frozenset[str] = _build_role_strip_tokens()

# Only gender and ethnic_background are used to condition synthetic-name generation.
# Faker only conditions given names on sex, so ethnic_background is omitted for that backend.
DEMO_KEYS: tuple[Literal["gender", "ethnic_background"], ...] = ("gender", "ethnic_background")


def demo_keys_for_backend(persona_backend: str) -> tuple[Literal["gender", "ethnic_background"], ...]:
    """Demographics that may appear in ``match_persona_by`` for this persona backend."""
    if persona_backend == "faker":
        return ("gender",)
    return DEMO_KEYS


# Only gender and ethnic_background condition synthetic-name generation, so those
# are the only demographics detected. (Age/DOB/occupation do not constrain persona sampling.)
DEMO_LABEL_PATTERNS: dict[str, list[str]] = {
    "gender": [r"^sex$", r"gender"],
    "ethnic_background": [r"race", r"ethnic"],
}

ORG_KEYWORDS = [
    "hospital",
    "clinic",
    "center",
    "centre",
    "health",
    "medical",
    "institute",
    "department",
    "dept",
    "university",
    "inc",
    "llc",
    "ltd",
    "corp",
    "system",
    "associates",
    "partners",
    "regional",
    "group",
    "services",
]


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
