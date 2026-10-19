# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-specific PII detection/replacement plan.

This is the declarative, column-oriented plan a user (or an upstream detector)
provides to describe *how* to replace PII in a dataset.

Every label is an ``Entity`` with an ``EntityAction`` and whether it
``can_condition`` other columns. User-facing plans name the entity on each
replace target and on read-only ``depends_on`` entries. A conditioner that is
also a replace target inherits the entity from that target.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, ClassVar, Literal, Self, cast

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    SkipValidation,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from ..configurator.parameters import Parameters
from ..defaults import DEFAULT_NSS_INFERENCE_LOCAL_MODEL, NSS_NEMOTRON_PERSONAS_PATH_ENV, default_nemotron_personas_path
from ..errors import ParameterError
from .base import NSSBaseModel
from .unknown_fields import raise_if_removed_legacy_fields
from .validation import format_pydantic_validation_error

__all__ = [
    "ALLOWED_DEPENDS_ON",
    "AUTO_DISCOVERY",
    "ConditioningColumn",
    "DEFAULT_GLINER2_MODEL_ID",
    "DEFAULT_GLINER2_MODEL_REVISION",
    "DataToSamplerValueMapping",
    "ENTITIES",
    "ENTITY_BY_TYPE",
    "EXCLUSIVE_DEPENDS_ON_GROUPS",
    "FREE_TEXT_DETECTION_ENTITY_TYPES",
    "GLINER_DETECTION_ENTITY_TYPES",
    "REGEX_DETECTION_ENTITY_TYPES",
    "Entity",
    "EntityAction",
    "EntityType",
    "FreeTextDetectionConfig",
    "LLMConfig",
    "PatternSyntax",
    "PiiColumnPlan",
    "PiiReplacementPlan",
    "PiiReplacementSettings",
    "PiiSamplerBackend",
    "PiiSamplerConfig",
    "ReplacePiiConfig",
    "can_condition",
    "is_columns_to_replace_type",
    "validate_dependency_relationship",
    "validate_pattern_eligibility",
]

# Sentinel value for ``ReplacePiiConfig.replacement_plan`` requesting automatic
# entity discovery instead of an explicit plan.
AUTO_DISCOVERY = "auto_discovery"
DEFAULT_GLINER2_MODEL_ID = "fastino/gliner2-privacy-filter-PII-multi"
DEFAULT_GLINER2_MODEL_REVISION = "c153999da5f4c509df4322b0c6a1baf3d2c284d7"
"""Hugging Face commit of ``DEFAULT_GLINER2_MODEL_ID`` that the default thresholds were tuned on."""
# Unversioned configurations are permanently interpreted as v3. Adding a new
# schema may expand the supported set, but must not advance this implicit value.
_IMPLICIT_REPLACE_PII_SCHEMA_VERSION = 3
_SUPPORTED_REPLACE_PII_SCHEMA_VERSIONS = frozenset({3})


class EntityType(StrEnum):
    """Closed vocabulary for discovery and plan ``entity_type`` fields.

    ``FREE_TEXT`` marks columns whose accepted PII spans are detected by
    GLiNER2 and applicable deterministic regex rules, then replaced without
    changing the surrounding text.
    """

    FIRST_NAME = "first_name"
    MIDDLE_NAME = "middle_name"
    LAST_NAME = "last_name"
    FULL_NAME = "full_name"
    EMAIL = "email"
    PHONE_NUMBER = "phone_number"
    DATE_OF_BIRTH = "date_of_birth"
    STREET_ADDRESS = "street_address"
    SSN = "ssn"
    NATIONAL_ID = "national_id"
    CREDIT_DEBIT_CARD = "credit_debit_card"
    API_KEY = "api_key"  # pragma: allowlist secret
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    UNIQUE_IDENTIFIER = "unique_identifier"

    # Detect and replace PII spans in free-form text with GLiNER2 plus the
    # applicable deterministic built-in regex rules.
    FREE_TEXT = "free_text"

    # Identify-only (and often original-value conditioners): discovery may classify
    # these so they are not mistaken for free text / replace targets.
    DATE = "date"  # generic (non-birth) date
    GENDER = "gender"
    ETHNIC_BACKGROUND = "ethnic_background"
    CITY = "city"
    STATE = "state"
    ZIPCODE = "zipcode"
    COUNTRY = "country"
    ORGANIZATION = "organization"


DataToSamplerValueMapping = dict[str, dict[str, list[str]]]
"""Dependency-column values mapped to values understood by one sampler.

Keyed by ``depends_on`` column name, then by dataset value. All keys and values are compared ignoring case.
"""


class EntityAction(Enum):
    """What the engine does to a column of this entity type."""

    REPLACE = auto()
    """Synthesize a new cell value."""

    REPLACE_IN_TEXT = auto()
    """Rewrite PII spans inside free-form text, leaving the surrounding text intact.

    Unlike ``REPLACE``, the cell is not itself a single entity value: only the
    spans accepted from GLiNER2 and applicable deterministic regex rules are
    rewritten. Existing structured values and mappings do not create spans.
    """

    IDENTIFY_ONLY = auto()
    """Label the column but never write to it.

    Cannot appear on ``columns_to_replace``; may still condition other columns
    when ``can_condition`` is set.
    """


class PatternSyntax(Enum):
    """Grammar used by ``PiiColumnPlan.pattern`` for an entity type."""

    STRFTIME = auto()
    """Python ``strftime`` codes, e.g. ``%m/%d/%Y`` or ``%d.%m.%y``."""

    CHARACTER_MASK = auto()
    """One drawn character per token, e.g. ``pmc-######`` or ``CUST-10[01]###``."""

    NAME_PARTS = auto()
    """Whole-value templates composed of literals and name-part placeholders."""


@dataclass(frozen=True, slots=True)
class Entity:
    """One label in the entity vocabulary."""

    entity_type: EntityType
    """Label this entry defines, as named by ``entity_type`` in user-facing plans."""

    action: EntityAction
    """Engine treatment of a column carrying this entity."""

    can_condition: bool
    """Whether this entity may appear as ``entity_type`` on a ``depends_on`` entry.

    The executor applies ``columns_to_replace`` in DAG order and reads the
    current cell as the conditioner, so a conditioner that is itself a replace
    target supplies its post-replacement value.
    """

    description: str
    """Short, general definition shown to the LLM classifier, so it can tell this entity from similar-looking data."""

    pattern_syntax: PatternSyntax | None = None
    """Notation ``pattern`` uses, or ``None`` when this entity allows no pattern."""


ENTITIES: tuple[Entity, ...] = (
    # Replace and may condition later columns.
    Entity(
        EntityType.FIRST_NAME,
        action=EntityAction.REPLACE,
        can_condition=True,
        description="A person's given name.",
        pattern_syntax=PatternSyntax.NAME_PARTS,
    ),
    Entity(
        EntityType.MIDDLE_NAME,
        action=EntityAction.REPLACE,
        can_condition=True,
        description="A person's middle name or middle initial.",
        pattern_syntax=PatternSyntax.NAME_PARTS,
    ),
    Entity(
        EntityType.LAST_NAME,
        action=EntityAction.REPLACE,
        can_condition=True,
        description="A person's family name or surname.",
        pattern_syntax=PatternSyntax.NAME_PARTS,
    ),
    Entity(
        EntityType.FULL_NAME,
        action=EntityAction.REPLACE,
        can_condition=True,
        description="A person's full name.",
        pattern_syntax=PatternSyntax.NAME_PARTS,
    ),
    # Replace only.
    Entity(
        EntityType.EMAIL,
        action=EntityAction.REPLACE,
        can_condition=False,
        description="An email address.",
        pattern_syntax=PatternSyntax.NAME_PARTS,
    ),
    Entity(
        EntityType.PHONE_NUMBER,
        action=EntityAction.REPLACE,
        can_condition=False,
        description="A telephone or fax number.",
        pattern_syntax=PatternSyntax.CHARACTER_MASK,
    ),
    Entity(
        EntityType.DATE_OF_BIRTH,
        action=EntityAction.REPLACE,
        can_condition=False,
        description="A person's date of birth, written as a full or partial calendar date.",
        pattern_syntax=PatternSyntax.STRFTIME,
    ),
    Entity(
        EntityType.STREET_ADDRESS,
        action=EntityAction.REPLACE,
        can_condition=False,
        description="A street address, possibly including city, state, or postal code.",
    ),
    Entity(
        EntityType.SSN, action=EntityAction.REPLACE, can_condition=False, description="A US Social Security number."
    ),
    Entity(
        EntityType.NATIONAL_ID,
        action=EntityAction.REPLACE,
        can_condition=False,
        description="A government-issued personal identification number other than a US Social Security number.",
    ),
    Entity(
        EntityType.CREDIT_DEBIT_CARD,
        action=EntityAction.REPLACE,
        can_condition=False,
        description="A payment card number.",
        pattern_syntax=PatternSyntax.CHARACTER_MASK,
    ),
    Entity(
        EntityType.API_KEY,
        action=EntityAction.REPLACE,
        can_condition=False,
        description="A secret access credential, such as an API key or access token.",
        pattern_syntax=PatternSyntax.CHARACTER_MASK,
    ),
    Entity(EntityType.IPV4, action=EntityAction.REPLACE, can_condition=False, description="An IPv4 address."),
    Entity(EntityType.IPV6, action=EntityAction.REPLACE, can_condition=False, description="An IPv6 address."),
    Entity(
        EntityType.UNIQUE_IDENTIFIER,
        action=EntityAction.REPLACE,
        can_condition=False,
        description=(
            "An identifier that refers to a specific person or their records. "
            "Identifiers of products, places, or events, and row numbers, are not this type."
        ),
        pattern_syntax=PatternSyntax.CHARACTER_MASK,
    ),
    # Rewrite spans inside cell text (username/url use this too).
    Entity(
        EntityType.FREE_TEXT,
        action=EntityAction.REPLACE_IN_TEXT,
        can_condition=False,
        description="Free-form text, such as notes or comments, that may mention personal details.",
    ),
    # Identify-only.
    Entity(
        EntityType.DATE,
        action=EntityAction.IDENTIFY_ONLY,
        can_condition=False,
        description="A date other than a date of birth.",
    ),
    # Identify-only and may condition replacements.
    Entity(
        EntityType.GENDER,
        action=EntityAction.IDENTIFY_ONLY,
        can_condition=True,
        description="A person's gender or sex.",
    ),
    Entity(
        EntityType.ETHNIC_BACKGROUND,
        action=EntityAction.IDENTIFY_ONLY,
        can_condition=True,
        description="A person's race or ethnicity.",
    ),
    Entity(EntityType.CITY, action=EntityAction.IDENTIFY_ONLY, can_condition=False, description="A city or town."),
    Entity(
        EntityType.STATE,
        action=EntityAction.IDENTIFY_ONLY,
        can_condition=False,
        description="A state, province, or region.",
    ),
    Entity(
        EntityType.ZIPCODE, action=EntityAction.IDENTIFY_ONLY, can_condition=False, description="A postal or ZIP code."
    ),
    Entity(EntityType.COUNTRY, action=EntityAction.IDENTIFY_ONLY, can_condition=False, description="A country."),
    Entity(
        EntityType.ORGANIZATION,
        action=EntityAction.IDENTIFY_ONLY,
        can_condition=True,
        description="The name of a company, institution, or other organization.",
    ),
)

ENTITY_BY_TYPE: dict[EntityType, Entity] = {entity.entity_type: entity for entity in ENTITIES}

FREE_TEXT_DETECTION_ENTITY_TYPES: tuple[EntityType, ...] = tuple(
    entity.entity_type for entity in ENTITIES if entity.action is EntityAction.REPLACE
)
"""Entity types eligible for fresh detection and replacement inside free text."""

REGEX_DETECTION_ENTITY_TYPES: tuple[EntityType, ...] = (
    EntityType.EMAIL,
    EntityType.CREDIT_DEBIT_CARD,
    EntityType.IPV4,
    EntityType.IPV6,
)
"""Entity types detected exclusively by structurally validated built-in regex rules."""

GLINER_DETECTION_ENTITY_TYPES: tuple[EntityType, ...] = tuple(
    entity_type for entity_type in FREE_TEXT_DETECTION_ENTITY_TYPES if entity_type not in REGEX_DETECTION_ENTITY_TYPES
)
"""Entity types requested from GLiNER2 rather than the built-in regex detector.

The checkpoint's specific ``full_name`` label is used for complete names; its
broader ``person`` label is intentionally not requested.
"""

# Thresholds come from the nss-pii-ner-eval v6 tuning on its dev split (entity-type exact F0.5),
# rounded. Name thresholds were tuned together with the name exclusion labels below and assume them.
# Dev has no gold middle names, so middle_name was set by review and relies on the middle-name
# neighbour rule to reject words such as clothing sizes.
_DEFAULT_GLINER_ENTITY_THRESHOLDS: Mapping[EntityType, float] = MappingProxyType(
    {
        EntityType.FIRST_NAME: 0.946,
        EntityType.MIDDLE_NAME: 0.7,
        EntityType.LAST_NAME: 0.618,
        EntityType.FULL_NAME: 0.964,
        EntityType.PHONE_NUMBER: 0.990,
        EntityType.DATE_OF_BIRTH: 0.9998,
        EntityType.STREET_ADDRESS: 0.985,
        EntityType.SSN: 0.050,
        EntityType.NATIONAL_ID: 0.982,
        EntityType.API_KEY: 0.992,
        EntityType.UNIQUE_IDENTIFIER: 0.947,
    }
)
# Same v6 tuning: GLiNER2 tags medical terms and words such as "mother" or "male" as names, so a name
# span overlapping one of these labels is dropped. Earlier tunings also requested medical_procedure, but tuned it to
# 1.0, which never fires, so it is not requested.
_DEFAULT_NAME_EXCLUSION_THRESHOLDS: Mapping[str, float] = MappingProxyType(
    {
        "body_part": 0.062,
        "family_relationship": 0.062,
        "gender": 0.385,
        "medical_condition": 0.393,
        "organism": 0.051,
    }
)
_ConfidenceThreshold = Annotated[float, Field(ge=0, le=1)]


def _default_gliner_entity_thresholds() -> dict[EntityType, float]:
    """Return a fresh, complete set of precision-first GLiNER2 thresholds."""
    return {
        entity_type: _DEFAULT_GLINER_ENTITY_THRESHOLDS[entity_type] for entity_type in GLINER_DETECTION_ENTITY_TYPES
    }


def _default_name_exclusion_thresholds() -> dict[str, float]:
    """Return a fresh copy of the tuned name exclusion labels."""
    return dict(_DEFAULT_NAME_EXCLUSION_THRESHOLDS)


def _normalize_gliner_label(label: str) -> str:
    """Normalize a GLiNER2 label's case and separators so equivalent spellings compare equal."""
    return label.strip().casefold().replace("-", "_").replace(" ", "_")


# entity_type → allowed depends_on entity types (optional edges may be omitted).
ALLOWED_DEPENDS_ON: dict[EntityType, frozenset[EntityType]] = {
    EntityType.FIRST_NAME: frozenset({EntityType.GENDER, EntityType.ETHNIC_BACKGROUND, EntityType.FULL_NAME}),
    EntityType.MIDDLE_NAME: frozenset({EntityType.GENDER, EntityType.ETHNIC_BACKGROUND, EntityType.FULL_NAME}),
    EntityType.LAST_NAME: frozenset({EntityType.ETHNIC_BACKGROUND, EntityType.FULL_NAME}),
    EntityType.FULL_NAME: frozenset({EntityType.GENDER, EntityType.ETHNIC_BACKGROUND}),
    EntityType.EMAIL: frozenset(
        {
            EntityType.FIRST_NAME,
            EntityType.MIDDLE_NAME,
            EntityType.LAST_NAME,
            EntityType.FULL_NAME,
            EntityType.ORGANIZATION,
        }
    ),
}

# Each inner tuple is one exclusivity family: at most one group from that family
# may appear in a single depends_on list. Families are independent.
# Applied to every columns_to_replace entry after entity_type inference.
EXCLUSIVE_DEPENDS_ON_GROUPS: tuple[tuple[frozenset[EntityType], ...], ...] = (
    (
        frozenset({EntityType.FIRST_NAME, EntityType.MIDDLE_NAME, EntityType.LAST_NAME}),
        frozenset({EntityType.FULL_NAME}),
    ),
    (
        frozenset({EntityType.FULL_NAME}),
        frozenset({EntityType.GENDER, EntityType.ETHNIC_BACKGROUND}),
    ),
)


def is_columns_to_replace_type(entity_type: EntityType) -> bool:
    """Whether ``entity_type`` may appear on ``columns_to_replace``."""
    return ENTITY_BY_TYPE[entity_type].action is not EntityAction.IDENTIFY_ONLY


def can_condition(entity_type: EntityType) -> bool:
    """Whether ``entity_type`` may appear on a ``depends_on`` entry."""
    return ENTITY_BY_TYPE[entity_type].can_condition


def validate_pattern_eligibility(entity_type: EntityType | None, pattern: str | None) -> None:
    """Validate context-free eligibility of a classification or plan pattern."""
    if pattern is None:
        return
    if not pattern.strip():
        raise ParameterError("pattern must be non-empty when provided")
    if entity_type is None:
        raise ParameterError("unclassified columns cannot include a pattern")
    if ENTITY_BY_TYPE[entity_type].pattern_syntax is None:
        raise ParameterError(f"entity_type {entity_type.value!r} does not allow pattern")


def validate_dependency_relationship(
    target_entity_type: EntityType,
    source_entity_type: EntityType | None,
    *,
    target_column: str,
) -> None:
    """Validate one dependency edge against the entity relationship catalog.

    A missing source type is deferred until ``PiiReplacementPlan`` can infer it
    from the complete replacement graph; the target must still support
    dependencies at this earlier validation seam.
    """
    allowed = ALLOWED_DEPENDS_ON.get(target_entity_type, frozenset())
    if not allowed:
        raise ParameterError(
            f"column {target_column!r}: entity_type {target_entity_type.value!r} does not allow depends_on"
        )
    if source_entity_type is None or source_entity_type in allowed:
        return
    raise ParameterError(
        f"column {target_column!r}: depends_on entity_type "
        f"{source_entity_type.value!r} is not allowed for entity_type "
        f"{target_entity_type.value!r} (allowed: {sorted(entity.value for entity in allowed)})"
    )


class ConditioningColumn(NSSBaseModel):
    """An existing column that conditions the replacement of another column.

    ``entity_type`` is required for read-only conditioners (``gender``,
    ``ethnic_background``, ``city``, …) that are not listed under
    ``columns_to_replace``.

    When the conditioner is itself a replace target (e.g. email depends on
    ``first_name``), ``entity_type`` may be omitted: the plan fills it in from that
    column's entry, and an explicit value must match it. Downstream nodes see the
    cell after upstream replacements.
    """

    column_name: str = Field(description="Existing dataframe column that supplies the conditioning value.")
    entity_type: EntityType | None = Field(
        default=None,
        description=(
            "Entity this conditioning column holds. Required for read-only "
            "conditioners. Optional when this column_name is also in columns_to_replace: "
            "it is filled in from that entry's entity_type, and an explicit value must match it."
        ),
    )

    @model_validator(mode="after")
    def _require_can_condition_when_set(self) -> Self:
        if self.entity_type is not None and not can_condition(self.entity_type):
            raise ParameterError(
                f"entity_type {self.entity_type.value!r} cannot be used in depends_on "
                f"(allowed: {sorted(e.entity_type.value for e in ENTITIES if e.can_condition)})"
            )
        return self


class PiiColumnPlan(NSSBaseModel):
    """Replacement spec for one named column.

    ``entity_type`` must have action ``REPLACE`` or ``REPLACE_IN_TEXT``, not
    ``IDENTIFY_ONLY``.
    Identify-only types (``date``, ``gender``, ``city``, …) are invalid here.
    ``depends_on`` is allowed only for types in ``ALLOWED_DEPENDS_ON``.
    """

    column_name: str = Field(description="Name of the dataframe column to replace or scan.")
    entity_type: EntityType = Field(
        description="Entity this column holds. Identify-only entity types are not allowed here.",
    )
    pattern: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Optional whole-value format using the grammar associated with this entity type. "
            "Only entity types that define a pattern syntax may set this. "
            "When provided, the whole column is replaced with the pattern if it "
            "covers at least 99% of non-null values (checked against the dataframe, "
            "not here)."
        ),
    )
    depends_on: list[ConditioningColumn] = Field(
        default_factory=list,
        exclude_if=lambda dependencies: not dependencies,
        description=(
            "Columns that condition the replacement of this column. "
            "Only entity types in ALLOWED_DEPENDS_ON may set this. Can omit "
            "ConditioningColumn.entity_type when the conditioner is listed in "
            "columns_to_replace. Matrix checks for explicit entity_type run here; "
            "omitted types and exclusive-group checks are resolved on PiiReplacementPlan."
        ),
    )

    @model_validator(mode="after")
    def _validate_entity_and_depends_on(self) -> Self:
        entity = ENTITY_BY_TYPE[self.entity_type]
        if entity.action is EntityAction.IDENTIFY_ONLY:
            raise ParameterError(
                f"column {self.column_name!r}: entity_type {self.entity_type.value!r} is "
                "identify-only (or otherwise not replaceable); omit it from columns_to_replace "
            )
        validate_pattern_eligibility(self.entity_type, self.pattern)
        for dependency in self.depends_on:
            if dependency.column_name == self.column_name:
                raise ParameterError(f"column {self.column_name!r} cannot depend on itself")
            validate_dependency_relationship(
                self.entity_type,
                dependency.entity_type,
                target_column=self.column_name,
            )
        return self


class PiiReplacementPlan(Parameters):
    """Dataset-specific detection/replacement plan (column-oriented).

    Flat ``columns_to_replace`` list with adjacent data-to-sampler value mappings;
    cross-column relationships are expressed via ``depends_on`` edges (a DAG).
    Context-free dependency and graph checks are enforced here; plan-vs-dataframe checks live in
    ``pii_replacer.planning.validation``.
    """

    columns_to_replace: list[PiiColumnPlan] = Field(
        default_factory=list,
        description="Columns to replace or (for free_text) scan for PII spans to rewrite.",
    )
    data_to_sampler_value_mapping: DataToSamplerValueMapping = Field(
        default_factory=dict,
        description=(
            "Maps dataset values in depends_on columns to the values used by the person sampler "
            "(replace_pii.sampler). Unlike most NSS settings, all matching here ignores case: dataset values, "
            "mapping keys, and sampler values, so Female, female, and FEMALE are the same value. For example, "
            "{sex: {Woman: [female], Non-binary: [female, male]}} makes rows where sex is Woman sample a female "
            "persona and rows where sex is Non-binary sample either. A dataset value without an entry is looked "
            "up in the sampler as is; a value that has no entry and matches no sampler value is an error. "
            "When replace_pii.replacement_plan is 'auto_discovery', NSS generates this mapping for you."
        ),
    )

    @model_validator(mode="after")
    def _validate_data_to_sampler_value_mapping(self) -> Self:
        for column_name, mappings in self.data_to_sampler_value_mapping.items():
            if not column_name.strip():
                raise ParameterError("data_to_sampler_value_mapping column names must be non-empty")
            seen_sources: set[str] = set()
            for source, targets in mappings.items():
                source_key = source.casefold()
                if not source.strip():
                    raise ParameterError("data_to_sampler_value_mapping dataset values must be non-empty")
                if source_key in seen_sources:
                    raise ParameterError(
                        f"data_to_sampler_value_mapping for column {column_name!r} contains duplicate "
                        "dataset values after case-folding"
                    )
                seen_sources.add(source_key)

                if not targets:
                    raise ParameterError("data_to_sampler_value_mapping sampler value lists must be non-empty")

                seen_targets: set[str] = set()
                for target in targets:
                    target_key = target.casefold()
                    if not target.strip():
                        raise ParameterError("data_to_sampler_value_mapping sampler values must be non-empty")
                    if target_key in seen_targets:
                        raise ParameterError(
                            f"data_to_sampler_value_mapping for column {column_name!r} contains duplicate "
                            "sampler values after case-folding"
                        )
                    seen_targets.add(target_key)
        return self

    @model_validator(mode="after")
    def _reject_duplicate_replace_columns(self) -> Self:
        seen: set[str] = set()
        duplicates: list[str] = []
        for spec in self.columns_to_replace:
            if spec.column_name in seen:
                duplicates.append(spec.column_name)
            seen.add(spec.column_name)
        if duplicates:
            raise ParameterError(
                "columns_to_replace has duplicate column_name values: " + ", ".join(repr(name) for name in duplicates)
            )
        return self

    @model_validator(mode="after")
    def _resolve_omitted_depends_on_types(self) -> Self:
        """Fill in or check depends_on.entity_type for columns that are also replaced.

        A dependency on a column in ``columns_to_replace`` may omit ``entity_type`` for convenience; it is filled in
        from that column's entry and serialized like any other value. An explicit ``entity_type`` must match the entry.
        Inference is written to copies: callers may reuse a ``PiiColumnPlan`` across plans, where the same instance
        would otherwise carry one plan's inferred types into the next.
        """
        targets = {spec.column_name: spec for spec in self.columns_to_replace}
        updated: list[PiiColumnPlan] = []
        for spec in self.columns_to_replace:
            if not spec.depends_on:
                updated.append(spec)
                continue
            resolved: list[ConditioningColumn] = []
            inferred_any = False
            for dep in spec.depends_on:
                source = targets.get(dep.column_name)
                if source is not None:
                    inferred = source.entity_type
                    if dep.entity_type is not None and dep.entity_type is not inferred:
                        raise ParameterError(
                            f"column {spec.column_name!r}: depends_on column {dep.column_name!r} has entity_type "
                            f"{dep.entity_type.value!r}, but its columns_to_replace entry has "
                            f"{inferred.value!r}; omit entity_type or make them match"
                        )
                    if not can_condition(inferred):
                        raise ParameterError(
                            f"column {spec.column_name!r}: depends_on column "
                            f"{dep.column_name!r} has entity_type {inferred.value!r}, which "
                            "cannot be used as a conditioner"
                        )
                    validate_dependency_relationship(
                        spec.entity_type,
                        inferred,
                        target_column=spec.column_name,
                    )
                    if dep.entity_type is None:
                        dep = dep.model_copy(update={"entity_type": inferred})
                        inferred_any = True
                elif dep.entity_type is None:
                    raise ParameterError(
                        f"column {spec.column_name!r}: depends_on column "
                        f"{dep.column_name!r} is missing entity_type; set entity_type "
                        "for read-only conditioners (e.g. gender, ethnic_background)"
                    )
                elif is_columns_to_replace_type(dep.entity_type):
                    raise ParameterError(
                        f"column {spec.column_name!r}: depends_on column "
                        f"{dep.column_name!r} has entity_type {dep.entity_type.value!r}, which is "
                        "a replace target; list it in columns_to_replace"
                    )
                else:
                    validate_dependency_relationship(
                        spec.entity_type,
                        dep.entity_type,
                        target_column=spec.column_name,
                    )
                resolved.append(dep)
            updated.append(spec.model_copy(update={"depends_on": resolved}) if inferred_any else spec)
        self.columns_to_replace = updated
        return self

    @model_validator(mode="after")
    def _reject_duplicate_depends_on_entity_types(self) -> Self:
        """At most one depends_on edge per conditioner entity_type on a target."""
        for spec in self.columns_to_replace:
            seen: dict[EntityType, str] = {}
            for dep in spec.depends_on:
                entity_type = dep.entity_type
                if entity_type is None:
                    continue
                prior = seen.get(entity_type)
                if prior is not None:
                    raise ParameterError(
                        f"column {spec.column_name!r}: depends_on entity_type "
                        f"{entity_type.value!r} appears more than once "
                        f"({prior!r} and {dep.column_name!r})"
                    )
                seen[entity_type] = dep.column_name
        return self

    @model_validator(mode="after")
    def _reject_exclusive_depends_on_groups(self) -> Self:
        """Forbid mixing mutually exclusive conditioner groups within a family."""
        for spec in self.columns_to_replace:
            if not spec.depends_on:
                continue
            dep_types = {dep.entity_type for dep in spec.depends_on}
            for groups in EXCLUSIVE_DEPENDS_ON_GROUPS:
                hit = [group for group in groups if dep_types & group]
                if len(hit) < 2:
                    continue
                formatted = " vs ".join(
                    "{" + ", ".join(sorted(member.value for member in group)) + "}" for group in hit
                )
                raise ParameterError(
                    f"column {spec.column_name!r}: depends_on mixes mutually exclusive conditioner groups: {formatted}"
                )
        return self

    @model_validator(mode="after")
    def _reject_dependency_cycles(self) -> Self:
        """Reject cycles defensively even though the current catalog is acyclic."""
        targets = {spec.column_name for spec in self.columns_to_replace}
        adjacency: dict[str, set[str]] = {column: set() for column in targets}
        indegree = dict.fromkeys(targets, 0)

        for spec in self.columns_to_replace:
            for dependency in spec.depends_on:
                source = dependency.column_name
                if source not in targets or spec.column_name in adjacency[source]:
                    continue
                adjacency[source].add(spec.column_name)
                indegree[spec.column_name] += 1

        ready = [column for column, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            source = ready.pop()
            visited += 1
            for target in adjacency[source]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)

        if visited != len(targets):
            cycle_columns = sorted(column for column, degree in indegree.items() if degree > 0)
            raise ParameterError(
                "replacement dependencies contain a cycle involving: "
                + ", ".join(repr(column) for column in cycle_columns)
            )
        return self


class LLMConfig(NSSBaseModel):
    """Inference behavior for LLM-assisted PII plan discovery.

    LLM behavior is disabled when ``ReplacePiiConfig.llm`` is ``None``. Without
    an endpoint, NSS runs the model in a managed local vLLM server. An
    OpenAI-compatible endpoint is supplied at runtime through
    ``NSS_INFERENCE_ENDPOINT`` or ``--inference-endpoint-url`` rather than persisted
    in NSS configuration.
    """

    model_id: str | None = Field(
        default=None,
        description=(
            "Model identifier for PII inference. When unset, uses NSS_INFERENCE_MODEL. Without an inference "
            "endpoint, selects the bundled local vLLM profile and defaults to "
            f"{DEFAULT_NSS_INFERENCE_LOCAL_MODEL}; with an endpoint, a model ID is required."
        ),
    )
    max_workers: int = Field(
        default=8,
        ge=1,
        description="Maximum concurrent requests for LLM-assisted plan discovery. Must be at least 1.",
    )


class FreeTextDetectionConfig(NSSBaseModel):
    """GLiNER2 and deterministic regex settings for free-text PII detection."""

    @model_validator(mode="before")
    @classmethod
    def _drop_default_revision_for_other_models(cls, data: object) -> object:
        """Load the latest revision of a non-default model unless the caller pins one.

        The default revision is a commit of the default model and does not exist in other repositories.
        """
        if not isinstance(data, Mapping):
            return data
        values = {str(key): value for key, value in data.items()}
        if "model_revision" in values or values.get("model_id", DEFAULT_GLINER2_MODEL_ID) == DEFAULT_GLINER2_MODEL_ID:
            return data
        return {**values, "model_revision": None}

    model_id: str = Field(
        default=DEFAULT_GLINER2_MODEL_ID,
        min_length=1,
        description="GLiNER2 model identifier used to detect PII spans in free-text columns. Must be nonempty.",
    )
    model_revision: str | None = Field(
        default=DEFAULT_GLINER2_MODEL_REVISION,
        min_length=1,
        description=(
            "Hugging Face revision (commit, branch, or tag) of model_id to load. Defaults to the commit of the "
            "default model that the default thresholds were tuned on. When model_id is changed and "
            "model_revision is not set, the latest revision is loaded. Set null to load the latest revision."
        ),
    )
    entity_thresholds: dict[EntityType, _ConfidenceThreshold] = Field(
        default_factory=_default_gliner_entity_thresholds,
        description=(
            "Minimum GLiNER2 confidence score per entity type. Entity types left out use the defaults: 0.946 for "
            "first_name; 0.7 for middle_name; 0.618 for last_name; 0.964 for full_name; 0.990 for phone_number; 0.9998 for "
            "date_of_birth; 0.985 for street_address; 0.050 for ssn; 0.982 for national_id; 0.992 for api_key; "
            "and 0.947 for unique_identifier. Email, credit_debit_card, ipv4, and ipv6 are detected by regex and "
            "take no threshold."
        ),
    )
    name_exclusion_thresholds: dict[str, _ConfidenceThreshold] = Field(
        default_factory=_default_name_exclusion_thresholds,
        description=(
            "Extra GLiNER2 labels, with their minimum confidence, for terms often mistaken for names, such as "
            "medical conditions or family relationships. A detected name that overlaps one of these spans is not "
            "replaced. A value replaces the whole default set; use an empty mapping to turn exclusion off, which "
            "finds more real names at the cost of more false names in clinical text."
        ),
    )
    require_middle_name_neighbors: bool = Field(
        default=True,
        description=(
            "Replace a detected middle name only when it sits inside a detected full name or directly between a "
            "detected first and last name. This rejects words such as clothing sizes that GLiNER2 tags as "
            "middle names."
        ),
    )
    batch_size: int = Field(
        default=8,
        gt=0,
        description="Number of text chunks processed in one GLiNER2 inference batch. Must be positive.",
    )
    chunk_length: int = Field(
        default=384,
        gt=0,
        description="Maximum length of each GLiNER2 text chunk. Must be positive.",
    )
    chunk_overlap: int = Field(
        default=128,
        ge=0,
        description="Overlap between adjacent GLiNER2 text chunks. Must be nonnegative and smaller than chunk_length.",
    )

    @field_validator("entity_thresholds", mode="before")
    @classmethod
    def _fill_default_entity_thresholds(cls, value: object) -> object:
        """Let users override selected thresholds; unspecified entity types keep their defaults."""
        if isinstance(value, Mapping):
            return {**_default_gliner_entity_thresholds(), **value}
        return value

    @field_validator("entity_thresholds")
    @classmethod
    def _reject_non_gliner_entity_thresholds(
        cls,
        value: dict[EntityType, float],
    ) -> dict[EntityType, float]:
        unsupported = sorted(entity_type.value for entity_type in set(value) - set(GLINER_DETECTION_ENTITY_TYPES))
        if unsupported:
            raise ParameterError(
                "free_text_detection.entity_thresholds accepts only GLiNER-detected entity types; "
                f"unsupported: {unsupported}"
            )
        return value

    @field_validator("name_exclusion_thresholds")
    @classmethod
    def _validate_name_exclusion_labels(cls, value: dict[str, float]) -> dict[str, float]:
        """Reject blank, duplicate, and entity-type labels, since each would silently change detection."""
        normalized = {_normalize_gliner_label(label) for label in value}
        if "" in normalized:
            raise ParameterError("free_text_detection.name_exclusion_thresholds labels must be nonempty")
        if len(normalized) != len(value):
            raise ParameterError("free_text_detection.name_exclusion_thresholds labels must be distinct")
        colliding = sorted(normalized & {entity_type.value for entity_type in FREE_TEXT_DETECTION_ENTITY_TYPES})
        if colliding:
            raise ParameterError(
                f"free_text_detection.name_exclusion_thresholds cannot use detected entity types: {colliding}"
            )
        return value

    @model_validator(mode="after")
    def _validate_chunk_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_length:
            raise ParameterError(
                "free_text_detection.chunk_overlap must be smaller than chunk_length, "
                f"got chunk_overlap={self.chunk_overlap}, chunk_length={self.chunk_length}"
            )
        return self


class PiiReplacementSettings(NSSBaseModel):
    """Apply-time replacement basics."""

    locale: str = Field(
        default="en_US",
        description="Locale for generated names, addresses, and phone numbers.",
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Seed for synthetic value generation; when unset the sampler will fall back to PERSON_RANDOM_SEED or 42."
        ),
    )


class PiiSamplerBackend(StrEnum):
    """Source of synthetic values for names and related person-like fields."""

    NEMOTRON_PERSONAS = "nemotron-personas"
    """Draw from downloaded Nemotron-Personas locale assets."""

    FAKER = "faker"
    """Draw from the Faker library; ignores ``ethnic_background`` conditioners."""


class PiiSamplerConfig(NSSBaseModel):
    """Settings for Nemotron-Personas or Faker-backed person sampling."""

    backend: PiiSamplerBackend = Field(
        default=PiiSamplerBackend.NEMOTRON_PERSONAS,
        description="Person sampler backend: downloaded Nemotron-Personas assets or Faker.",
    )
    nemotron_personas_path: str | None = Field(
        default=None,
        description=(
            "Root directory containing downloaded Nemotron-Personas files as datasets/{locale}.parquet. "
            f"Defaults to {NSS_NEMOTRON_PERSONAS_PATH_ENV} or ~/.data-designer/managed-assets."
        ),
    )

    def resolved_nemotron_personas_path(self) -> Path:
        """Return ``nemotron_personas_path`` if set, else the environment or built-in default."""
        if self.nemotron_personas_path is not None:
            return Path(self.nemotron_personas_path)
        return default_nemotron_personas_path()


class ReplacePiiConfig(Parameters):
    """Top-level ``replace_pii`` config wrapping the replacement plan.

    ``replacement_plan`` accepts three forms:

    * ``"auto_discovery"`` (default) detects and plans replacements automatically;
    * an inline ``PiiReplacementPlan`` mapping in the main NSS config; or
    * a string path to a separate plan YAML containing that same mapping.

    ``llm=None`` leaves auto-discovery at the heuristic baseline. Supplying an
    ``llm`` mapping enables LLM enhancement after heuristic discovery. An
    inline plan or plan file bypasses discovery, so it does not require an LLM.
    The API key remains a runtime secret supplied through ``NSS_INFERENCE_KEY``
    or the corresponding CLI option; it is never stored in this model.

    v2 fields ``globals`` and ``steps`` are rejected with a removal error.
    """

    removed_legacy_fields: ClassVar[frozenset[str]] = frozenset({"globals", "steps"})
    removed_legacy_fields_message: ClassVar[str] = (
        "This configuration uses the PII replacement v2 schema, which is not supported by PII replacement v3. "
        "Configure replace_pii.replacement_plan instead. "
        "See docs/user-guide/configuration.md#replacing-pii "
        "for the current configuration."
    )

    schema_version: Literal[3] = Field(
        default=_IMPLICIT_REPLACE_PII_SCHEMA_VERSION,
        description=(
            "Version of this replace_pii configuration schema. Missing versions are treated as version 3; "
            "this release accepts only version 3."
        ),
    )
    replacement_plan: SkipValidation[PiiReplacementPlan] | str = Field(
        default=AUTO_DISCOVERY,
        description=(
            f"{AUTO_DISCOVERY!r} to discover the plan from the data, an inline plan "
            "mapping in the main NSS config, or a path to a separate plan YAML. "
            "Other strings are treated as paths. The CLI option only accepts the "
            "sentinel or a path."
        ),
    )
    llm: LLMConfig | None = Field(
        default=None,
        description=(
            "Optional inference behavior for LLM-assisted plan discovery. "
            "The endpoint is configured at runtime through NSS_INFERENCE_ENDPOINT or --inference-endpoint-url. "
            "Use an empty mapping to enable NSS inference defaults."
        ),
    )
    free_text_detection: FreeTextDetectionConfig = Field(
        default_factory=FreeTextDetectionConfig,
        description="GLiNER2 and deterministic built-in regex settings for detecting PII spans in free-text columns.",
    )
    replacement: PiiReplacementSettings = Field(
        default_factory=PiiReplacementSettings,
        description="Locale and seed for synthetic value generation.",
    )
    sampler: PiiSamplerConfig = Field(
        default_factory=PiiSamplerConfig,
        description="Synthetic value sampler backend and asset paths.",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_v2_fields(cls, value: object) -> object:
        if isinstance(value, Mapping):
            raise_if_removed_legacy_fields(cls, cast(Mapping[str, object], value), path=())
        return value

    @model_validator(mode="before")
    @classmethod
    def _validate_schema_version(cls, value: object) -> object:
        """Reject invalid or unsupported versions before validating the schema body."""
        if not isinstance(value, Mapping):
            return value
        mapping = cast(Mapping[str, object], value)
        version = mapping.get("schema_version", _IMPLICIT_REPLACE_PII_SCHEMA_VERSION)
        if type(version) is not int:
            raise ParameterError("replace_pii.schema_version must be an integer")
        if version not in _SUPPORTED_REPLACE_PII_SCHEMA_VERSIONS:
            supported = ", ".join(str(item) for item in sorted(_SUPPORTED_REPLACE_PII_SCHEMA_VERSIONS))
            raise ParameterError(
                f"replace_pii schema version {version} is unsupported; this NSS release supports version {supported}"
            )
        return value

    @model_serializer(mode="wrap")
    def _serialize_schema_version(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Include the selected schema version even in sparse serialization."""
        serialized = cast(dict[str, object], handler(self))
        serialized["schema_version"] = self.schema_version
        return serialized

    @field_validator("replacement_plan", mode="before")
    @classmethod
    def _resolve_replacement_plan(cls, value: object) -> object:
        """Resolve the plan/string union here so errors describe the plan, not the union.

        Left to the union, a malformed inline plan reports the plan's own errors
        *and* "input should be a valid string", which reads as though a file path
        was expected. Validating a mapping as a plan up front keeps the report to
        the fields the user actually got wrong.
        """
        if isinstance(value, Mapping):
            try:
                return PiiReplacementPlan.model_validate(value)
            except ValidationError as exc:
                details = format_pydantic_validation_error(exc)
                raise ParameterError(f"invalid inline replacement plan ({details})") from exc
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, str | PiiReplacementPlan):
            return value
        raise ParameterError(
            f"replacement_plan must be {AUTO_DISCOVERY!r}, an inline plan, or a path to a plan file; "
            f"got {type(value).__name__}"
        )

    @property
    def is_auto_discovery(self) -> bool:
        """Whether replacements should be auto-discovered (no explicit plan)."""
        return self.replacement_plan == AUTO_DISCOVERY

    @property
    def plan_path(self) -> str | None:
        """Path to an external plan file, or ``None`` if not a path reference."""
        if isinstance(self.replacement_plan, str) and self.replacement_plan != AUTO_DISCOVERY:
            return self.replacement_plan
        return None

    @property
    def inline_plan(self) -> PiiReplacementPlan | None:
        """The inline plan, or ``None`` for auto-discovery or a plan file."""
        return self.replacement_plan if isinstance(self.replacement_plan, PiiReplacementPlan) else None
