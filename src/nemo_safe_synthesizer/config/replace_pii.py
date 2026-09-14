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
from typing import ClassVar, Literal, Self, cast

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
from ..defaults import NSS_MANAGED_ASSETS_PATH_ENV, default_managed_assets_path
from ..errors import ParameterError
from .base import NSSBaseModel
from .unknown_fields import raise_if_removed_legacy_fields
from .validation import format_pydantic_validation_error

__all__ = [
    "ALLOWED_DEPENDS_ON",
    "AUTO_DISCOVERY",
    "ConditioningColumn",
    "ENTITIES",
    "ENTITY_BY_TYPE",
    "EXCLUSIVE_DEPENDS_ON_GROUPS",
    "Entity",
    "EntityAction",
    "EntityType",
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
# Unversioned configurations are permanently interpreted as v3. Adding a new
# schema may expand the supported set, but must not advance this implicit value.
_IMPLICIT_REPLACE_PII_SCHEMA_VERSION = 3
_SUPPORTED_REPLACE_PII_SCHEMA_VERSIONS = frozenset({3})


class EntityType(StrEnum):
    """Closed vocabulary for discovery and plan ``entity_type`` fields."""

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

    # Propagate already-replaced values into spans of cell text (username/url use this too).
    # Plus fresh NER and replacements when an LLM is available.
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


class EntityAction(Enum):
    """What the engine does to a column of this entity type."""

    REPLACE = auto()
    """Synthesize a new cell value."""

    REPLACE_IN_TEXT = auto()
    """Rewrite PII spans inside free-form text, leaving the surrounding text intact.

    Unlike ``REPLACE``, the cell is not itself a single entity value: only the
    spans within it are rewritten. Spans are sourced from values already
    replaced in other columns or newly discovered by NER when an LLM is available.
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

    pattern_syntax: PatternSyntax | None = None
    """Notation ``pattern`` uses, or ``None`` when this entity allows no pattern."""


ENTITIES: tuple[Entity, ...] = (
    # Replace and may condition later columns.
    Entity(
        EntityType.FIRST_NAME,
        action=EntityAction.REPLACE,
        can_condition=True,
        pattern_syntax=PatternSyntax.NAME_PARTS,
    ),
    Entity(
        EntityType.MIDDLE_NAME,
        action=EntityAction.REPLACE,
        can_condition=True,
        pattern_syntax=PatternSyntax.NAME_PARTS,
    ),
    Entity(
        EntityType.LAST_NAME,
        action=EntityAction.REPLACE,
        can_condition=True,
        pattern_syntax=PatternSyntax.NAME_PARTS,
    ),
    Entity(
        EntityType.FULL_NAME,
        action=EntityAction.REPLACE,
        can_condition=True,
        pattern_syntax=PatternSyntax.NAME_PARTS,
    ),
    # Replace only.
    Entity(
        EntityType.EMAIL,
        action=EntityAction.REPLACE,
        can_condition=False,
        pattern_syntax=PatternSyntax.NAME_PARTS,
    ),
    Entity(
        EntityType.PHONE_NUMBER,
        action=EntityAction.REPLACE,
        can_condition=False,
        pattern_syntax=PatternSyntax.CHARACTER_MASK,
    ),
    Entity(
        EntityType.DATE_OF_BIRTH,
        action=EntityAction.REPLACE,
        can_condition=False,
        pattern_syntax=PatternSyntax.STRFTIME,
    ),
    Entity(EntityType.STREET_ADDRESS, action=EntityAction.REPLACE, can_condition=False),
    Entity(EntityType.SSN, action=EntityAction.REPLACE, can_condition=False),
    Entity(EntityType.NATIONAL_ID, action=EntityAction.REPLACE, can_condition=False),
    Entity(
        EntityType.CREDIT_DEBIT_CARD,
        action=EntityAction.REPLACE,
        can_condition=False,
        pattern_syntax=PatternSyntax.CHARACTER_MASK,
    ),
    Entity(
        EntityType.API_KEY,
        action=EntityAction.REPLACE,
        can_condition=False,
        pattern_syntax=PatternSyntax.CHARACTER_MASK,
    ),
    Entity(EntityType.IPV4, action=EntityAction.REPLACE, can_condition=False),
    Entity(EntityType.IPV6, action=EntityAction.REPLACE, can_condition=False),
    Entity(
        EntityType.UNIQUE_IDENTIFIER,
        action=EntityAction.REPLACE,
        can_condition=False,
        pattern_syntax=PatternSyntax.CHARACTER_MASK,
    ),
    # Rewrite spans inside cell text (username/url use this too).
    Entity(EntityType.FREE_TEXT, action=EntityAction.REPLACE_IN_TEXT, can_condition=False),
    # Identify-only.
    Entity(EntityType.DATE, action=EntityAction.IDENTIFY_ONLY, can_condition=False),
    # Identify-only and may condition replacements.
    Entity(EntityType.GENDER, action=EntityAction.IDENTIFY_ONLY, can_condition=True),
    Entity(EntityType.ETHNIC_BACKGROUND, action=EntityAction.IDENTIFY_ONLY, can_condition=True),
    Entity(EntityType.CITY, action=EntityAction.IDENTIFY_ONLY, can_condition=True),
    Entity(EntityType.STATE, action=EntityAction.IDENTIFY_ONLY, can_condition=True),
    Entity(EntityType.ZIPCODE, action=EntityAction.IDENTIFY_ONLY, can_condition=True),
    Entity(EntityType.COUNTRY, action=EntityAction.IDENTIFY_ONLY, can_condition=True),
    Entity(EntityType.ORGANIZATION, action=EntityAction.IDENTIFY_ONLY, can_condition=True),
)

ENTITY_BY_TYPE: dict[EntityType, Entity] = {entity.entity_type: entity for entity in ENTITIES}

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
    EntityType.STREET_ADDRESS: frozenset({EntityType.CITY, EntityType.STATE, EntityType.ZIPCODE, EntityType.COUNTRY}),
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
    (
        frozenset({EntityType.ZIPCODE}),
        frozenset({EntityType.CITY, EntityType.STATE, EntityType.COUNTRY}),
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
    ``first_name``), omit ``entity_type``: the plan infers it from that column's
    ``entity_type``. Downstream nodes see the cell after upstream replacements.
    """

    column_name: str = Field(description="Existing dataframe column that supplies the conditioning value.")
    entity_type: EntityType | None = Field(
        default=None,
        description=(
            "Entity this conditioning column holds. Required for read-only "
            "conditioners; omit when this column_name is also in columns_to_replace "
            "(inferred from that entry's entity_type)."
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
        description=(
            "Optional whole-value format using the grammar associated with this entity type. "
            "Only entity types that define a pattern syntax may set this. "
            "When provided, the whole column is replaced with the pattern if it "
            "covers at least 85% of non-null values (checked against the dataframe, "
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

    Flat ``columns_to_replace`` list; cross-column relationships are expressed
    via ``depends_on`` edges (a DAG). Context-free dependency and graph checks
    are enforced here; plan-vs-dataframe checks live in
    ``pii_replacer.planning.validation``.
    """

    columns_to_replace: list[PiiColumnPlan] = Field(
        default_factory=list,
        description="Columns to replace or (for free_text) scan for PII spans to rewrite.",
    )

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
        """Infer missing depends_on.entity_type from columns_to_replace entity_type.

        Inference is written to copies: callers may reuse a ``PiiColumnPlan``
        across plans, where the same instance would otherwise carry one plan's
        inferred types into the next.
        """
        by_name = {spec.column_name: spec for spec in self.columns_to_replace}
        updated: list[PiiColumnPlan] = []
        for spec in self.columns_to_replace:
            if not spec.depends_on:
                updated.append(spec)
                continue
            resolved: list[ConditioningColumn] = []
            inferred_any = False
            for dep in spec.depends_on:
                source = by_name.get(dep.column_name)
                explicitly_typed = "entity_type" in dep.model_fields_set
                if source is not None:
                    if explicitly_typed:
                        raise ParameterError(
                            f"column {spec.column_name!r}: depends_on column "
                            f"{dep.column_name!r} is listed in columns_to_replace; omit its "
                            "entity_type because it is inferred from that replacement entry"
                        )
                    inferred = source.entity_type
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
                    # Keep entity_type out of model_fields_set so canonical plan
                    # serialization omits this runtime-inferred value.
                    dep = ConditioningColumn.model_construct(
                        _fields_set=set(dep.model_fields_set),
                        column_name=dep.column_name,
                        entity_type=inferred,
                    )
                    inferred_any = True
                elif not explicitly_typed or dep.entity_type is None:
                    raise ParameterError(
                        f"column {spec.column_name!r}: depends_on column "
                        f"{dep.column_name!r} omits entity_type but is not listed in "
                        "columns_to_replace; set entity_type for read-only conditioners "
                        "(e.g. gender, ethnic_background, city)"
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
    """Inference behavior shared by PII planning and replacement.

    LLM behavior is disabled when ``ReplacePiiConfig.llm`` is ``None``. The
    OpenAI-compatible endpoint is supplied at runtime through
    ``NSS_INFERENCE_ENDPOINT`` or ``--inference-endpoint-url`` rather than persisted
    in NSS configuration.
    """

    model_id: str | None = Field(
        default=None,
        description=(
            "Model identifier served by the inference endpoint. When unset, uses "
            "NSS_INFERENCE_MODEL or the NSS default model."
        ),
    )
    max_workers: int = Field(
        default=8,
        ge=1,
        description=(
            "Maximum concurrent requests for LLM-assisted plan discovery and free-text replacement. Must be at least 1."
        ),
    )


class PiiReplacementSettings(NSSBaseModel):
    """Apply-time replacement basics."""

    locale: str = Field(
        default="en_US",
        description="Locale for generated names, addresses, and phone numbers.",
    )
    seed: int | None = Field(
        default=None,
        description="Seed for synthetic value generation; when unset the sampler will fall back to PERSON_RANDOM_SEED or 42.",
    )


class PiiSamplerBackend(StrEnum):
    """Source of synthetic values for names and related person-like fields."""

    MANAGED = "managed"
    """Draw from managed locale assets (see ``PiiSamplerConfig.managed_assets_path``)."""

    FAKER = "faker"
    """Draw from the Faker library; ignores ``ethnic_background`` conditioners."""


class PiiSamplerConfig(NSSBaseModel):
    """Settings for the synthetic value sampler (names and related person-like fields)."""

    backend: PiiSamplerBackend = Field(
        default=PiiSamplerBackend.MANAGED,
        description="Synthetic value sampler backend: managed assets or Faker.",
    )
    managed_assets_path: str | None = Field(
        default=None,
        description=(
            "Root directory containing a datasets/ folder of locale parquet files. "
            f"Defaults to {NSS_MANAGED_ASSETS_PATH_ENV} or ~/.data-designer/managed-assets."
        ),
    )

    def resolved_managed_assets_path(self) -> Path:
        """Return ``managed_assets_path`` if set, else the environment or built-in default."""
        if self.managed_assets_path is not None:
            return Path(self.managed_assets_path)
        return default_managed_assets_path()


class ReplacePiiConfig(Parameters):
    """Top-level ``replace_pii`` config wrapping the replacement plan.

    ``replacement_plan`` accepts three forms:

    * ``"auto_discovery"`` (default) detects and plans replacements automatically;
    * an inline ``PiiReplacementPlan`` mapping in the main NSS config; or
    * a string path to a separate plan YAML containing that same mapping.

    ``llm=None`` leaves auto-discovery at the heuristic baseline and disables
    LLM-assisted free-text replacement. Supplying an ``llm`` mapping enables
    LLM enhancement after heuristic discovery and provides the same inference
    settings to replacement-time free-text processing. An inline plan or plan
    file bypasses only discovery; it does not disable replacement-time LLM use.
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
            "Optional inference behavior shared by plan enhancement and free-text replacement. "
            "The endpoint is configured at runtime through NSS_INFERENCE_ENDPOINT or --inference-endpoint-url. "
            "Use an empty mapping to enable NSS inference defaults."
        ),
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
