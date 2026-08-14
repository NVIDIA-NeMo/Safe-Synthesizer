# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine config, EntitySpec registry, and derived entity taxonomy views."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from .replacement.scope import FakerLike

from ..config.replace_pii import PiiReplacerConfig
from ..defaults import default_managed_assets_path


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class Config:
    """Engine knobs for detection and generation.

    A few fields are copied from user-facing ``PiiReplacerConfig`` (locale, seed,
    persona backend/paths, LLM flags) via ``config_from_replace_pii``. The rest
    are fixed product defaults used by discovery and replacement; they are not
    exposed in YAML/SDK today, but can be overridden when constructing
    ``Config`` directly (e.g. in tests).
    """

    locale: str = "en_US"
    """Locale for generated names, addresses, and phone numbers."""
    random_seed: int = field(default_factory=lambda: int(os.environ.get("PERSON_RANDOM_SEED", "42") or "42"))
    """Seed for persona/ID/Faker generation.

    Env-overridable via ``PERSON_RANDOM_SEED`` so batched runs can give each
    row-batch a distinct seed; otherwise every batch regenerates the same
    unique-identifier sequence and values collide across batches.
    """
    group_constancy_threshold: float = 0.95
    """Fraction of groups that must be single-valued for GROUP-level treatment.

    Grouping itself comes from Safe Synthesizer's ``group_training_records_by``;
    it is configured, not auto-detected.
    """
    persona_backend: str = "managed"
    """Persona sampler backend: ``pgm``, ``managed``, or ``faker``."""
    sdg_pgms_src: str | None = None
    """Source tree for sdg-pgms when ``persona_backend`` is ``pgm``; else unused."""
    managed_assets_path: str | None = None
    """Root of managed persona parquet assets; resolved in ``__post_init__`` when unset."""
    pool_min_size: int = 3_000
    """Floor on how many synthetic personas to pre-generate for the PGM pool."""
    pool_oversample: int = 6
    """Multiplier of persona instances used with ``pool_min_size`` to size the PGM pool."""
    dominant_pattern_min_coverage: float = 85.0
    """Minimum percent of non-null values matching the dominant pattern for structured columns."""
    name_fuzzy_threshold: float = 0.86
    """Acceptance threshold for fuzzy column-name matching."""
    llm_enhancement: bool = False
    """When True, discovery/apply call the injected discovery and replacement enhancers (stubs raise until LLM lands)."""
    llm_model_provider: str | None = None
    """Reserved inference model provider propagated from ``PiiReplacerConfig.llm``."""
    llm_max_workers: int = 64
    """Reserved max workers for LLM enhancement calls."""
    freetext_name_token_aliases: bool = True
    """Also propagate individual name tokens into free text for honorific/partial mentions.

    When a person is identified only by a full name (no separate first/last
    columns), rewriting tokens catches later mentions such as ``Dr. Smith``
    after ``John Smith`` was replaced.
    """
    freetext_alias_min_token_len: int = 3
    """Skip free-text name-token aliases shorter than this to avoid over-matching."""
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


def config_from_replace_pii(config: PiiReplacerConfig) -> Config:
    """Build engine ``Config`` from the user-facing ``PiiReplacerConfig``.

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

    Resolved with ``effective_apply_path`` (entity type + backend; may override
    per backend). That channel is not chosen by YAML section — but section still
    matters for consistency: persona-sourced columns share one synthetic person
    only when listed together under ``persona_backed_columns``.
    """
    pattern_kind: PatternKind = "none"
    """How plan patterns are interpreted for this entity (strftime, template, …)."""
    persona_only_backends: frozenset[str] | None = None
    """Backends for which ``apply_path`` is forced to ``persona``; ``None`` means none."""
    requires_value_match: bool = False
    """Discovery: values must classify as this entity before the column is allocated."""
    name_shape_gates: bool = False
    """Discovery: reject multi-person or org-shaped samples (person-name labels)."""
    transform_method: str | None = None
    """Stats/report method label (e.g. ``propagation``, ``perturbation``); ``None`` if N/A."""
    name_patterns: tuple[str, ...] = ()
    """Regex fragments matched against normalized column headers."""
    strong_name_patterns: tuple[str, ...] = ()
    """Subset of ``name_patterns`` that skip weak-name content gates.

    Empty means every name match is treated as strong (no weak tier). When set,
    a header that matches ``name_patterns`` but none of these is a weak match and
    the entity handler may require extra value evidence (e.g. dominant template).
    """
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
    unique_id_strong_name_patterns = (
        r"\buuid\b",
        r"\bguid\b",
        r"\bmrn\b",
        r"(?:^|[_ ])id$",
        r"\w+[_ ]id$",
        r"identifier",
        r"(?:^|_)key$",
        r"(?:^|_)ref$",
    )
    unique_id_weak_name_pattern = r"\b\w*_?id$"

    specs: list[EntitySpec] = [
        EntitySpec(
            label="first_name",
            apply_path="persona",
            pattern_kind="persona_placeholder",
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
            # Strong names skip the weak-``*id`` dominant-template gate; the catch-all
            # weak pattern still matches English leftovers (``valid``, ``userid``) but
            # ``UniqueIdentifierHandler`` requires a dominant identifier template then.
            strong_name_patterns=unique_id_strong_name_patterns,
            name_patterns=unique_id_strong_name_patterns + (unique_id_weak_name_pattern,),
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
    return s.apply_path


def is_identify_only(label: str) -> bool:
    """True when ``label`` is discovered/validated but never replaced."""
    s = ENTITY_REGISTRY.get(label)
    return s is not None and s.apply_path == "identify_only"


# ===========================================================================
# Entity behavior base
# ===========================================================================
class EntityHandler(ABC):
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

    @abstractmethod
    def match_value(self, value: object, *, phone_min_digits: int = 10) -> str | None:
        """Return this handler's entity label when ``value`` matches, else ``None``.

        Args:
            value: Cell value to classify.
            phone_min_digits: Minimum digit count for phone matches.

        Returns:
            ``self.label`` on a match, otherwise ``None``.
        """

    @abstractmethod
    def skip_reason(
        self,
        series: pd.Series,
        value_entity: str | None,
        apply_path: str,
        *,
        column_name: str | None = None,
        cfg: Config | None = None,
    ) -> str | None:
        """Return why discovery must not allocate this name-matched column.

        Args:
            series: Column values under consideration.
            value_entity: Entity inferred from values, if any.
            apply_path: Effective apply path for this label.
            column_name: Header for entity-specific name tiers (e.g. strong/weak).
            cfg: Engine configuration for entity-specific content gates.

        Returns:
            A human-readable skip reason, or ``None`` when the column may allocate.
        """

    @abstractmethod
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

    @abstractmethod
    def persona_value(
        self,
        original: str,
        persona: Mapping[str, object],
        *,
        patterns: Sequence[str] | None = None,
        originals: Mapping[str, object] | None = None,
        fake: FakerLike | None = None,
    ) -> str | None:
        """Return a persona-sourced replacement for one cell, or ``None``.

        Only persona-channel labels implement this; standalone entities leave it
        unused (``synth_value`` never reaches them).

        Args:
            original: Original cell text to replace.
            persona: Sampled persona dict for this instance.
            patterns: Optional column format templates.
            originals: This person's other original values (email local-part inference).
            fake: Optional seeded Faker for fallbacks / phone formatting.

        Returns:
            The synthetic value, or ``None`` to leave the cell unchanged.
        """

    @abstractmethod
    def plan_pattern_rejection(self, column_name: str) -> str | None:
        """Return why plan ``patterns`` are illegal for this entity, or ``None`` if allowed.

        Args:
            column_name: Column that listed patterns in the replacement plan.

        Returns:
            A human-readable rejection message, or ``None`` when patterns are fine.
        """


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

# Only sex and ethnic_background are used to condition synthetic-name generation.
# Faker only conditions given names on sex, so ethnic_background is omitted for that backend.
DEMO_KEYS: tuple[Literal["sex", "ethnic_background"], ...] = ("sex", "ethnic_background")


def demo_keys_for_backend(persona_backend: str) -> tuple[Literal["sex", "ethnic_background"], ...]:
    """Demographics that may appear in ``match_persona_by`` for this persona backend."""
    if persona_backend == "faker":
        return ("sex",)
    return DEMO_KEYS


# Only sex and ethnic_background condition synthetic-name generation, so those
# are the only demographics detected. (Age/DOB/occupation do not constrain persona sampling.)
DEMO_LABEL_PATTERNS: dict[str, list[str]] = {
    "sex": [r"^sex$", r"gender"],
    "ethnic_background": [r"race", r"ethnic"],
}

# Keyword spellings for fuzzy resolution when a header matches both an entity and a
# demographic label (or multiple demos). Kept separate from ``FUZZY_KEYWORDS`` so the
# no-regex fuzzy backstop can still prefer entity typos over bare demographic words.
DEMO_FUZZY_KEYWORDS: dict[str, list[str]] = {
    "sex": ["sex", "gender"],
    "ethnic_background": ["race", "ethnic", "ethnicity", "ethnicbackground"],
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
