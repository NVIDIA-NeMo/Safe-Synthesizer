# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-specific PII detection/replacement plan.

This is the declarative, column-oriented plan a user (or an upstream detector)
provides to describe *how* to replace PII in a dataset.

Every label is an ``Entity`` with an ``EntityAction`` and whether it
``can_condition`` other columns. User-facing plans name them with
``entity_type`` on both replace targets and ``depends_on`` entries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from pathlib import Path
from typing import ClassVar, Literal, Self, cast

from pydantic import Field, ValidationError, field_validator, model_validator

from ..configurator.parameters import Parameters
from ..defaults import NSS_MANAGED_ASSETS_PATH_ENV, default_managed_assets_path
from ..errors import ParameterError
from .base import NSSBaseModel
from .unknown_fields import raise_if_removed_legacy_fields

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
    "PiiReplacementScope",
    "PiiReplacementSettings",
    "PiiSamplerBackend",
    "PiiSamplerConfig",
    "ReplacePiiConfig",
    "can_condition",
    "is_columns_to_replace_type",
]

# Sentinel value for ``ReplacePiiConfig.replacement_plan`` requesting automatic
# entity discovery instead of an explicit plan.
AUTO_DISCOVERY = "auto_discovery"


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

    # Propagate already-replaced values into cell text (username/url use this too).
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

    PROPAGATE = auto()
    """Scan existing text and substitute values already replaced elsewhere."""

    IDENTIFY_ONLY = auto()
    """Label the column but never write to it.

    Cannot appear on ``columns_to_replace``; may still condition other columns
    when ``can_condition`` is set.
    """


class PatternSyntax(Enum):
    """Notation ``PiiColumnPlan.pattern`` is written in for this entity type.

    An entity with no syntax (``None``) accepts no ``pattern`` at all.
    """

    STRFTIME = auto()
    """Python ``strftime`` codes, e.g. ``%m/%d/%Y`` or ``%d.%m.%y``."""

    CHARACTER_MASK = auto()
    r"""One drawn character per token, e.g. ``pmc-######`` or ``CUST-10[01]###``.

    ``#`` digit, ``^`` A-Z, ``@`` a-z, ``&`` 0-9A-Z, ``%`` 0-9a-z, ``*``
    0-9A-Za-z, ``[abc]`` one of the listed characters, ``\x`` a literal ``x``.
    Note ``[...]`` is a literal set rather than a range, so ``[0-9]`` draws from
    ``0``, ``-``, ``9``.
    """

    NAME_PARTS = auto()
    """``{first}`` / ``{middle}`` / ``{last}`` placeholders for names and email.

    Case variants ``{First}`` (title), ``{FIRST}`` (upper), and ``{f}``
    (initial) apply to each part. Email also takes ``{domain}`` and ``#``.
    """


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
    # Scan cell text and propagate already-replaced values (username/url use this too).
    Entity(EntityType.FREE_TEXT, action=EntityAction.PROPAGATE, can_condition=False),
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

    ``entity_type`` must have action ``replace`` or ``propagate`` (not ``none``).
    Identify-only types (``date``, ``gender``, ``city``, …) are invalid here.
    ``depends_on`` is allowed only for types in ``ALLOWED_DEPENDS_ON``.
    """

    column_name: str = Field(description="Name of the dataframe column to replace or scan.")
    entity_type: EntityType = Field(
        description="Entity this column holds. Must have action replace or propagate.",
    )
    pattern: str | None = Field(
        default=None,
        description=(
            "Format this column writes the entity in: strftime for birth "
            "dates (%m/%d/%Y), character templates for identifiers/phones (pmc-######, "
            "+1-###-555-####), or person-part placeholders for names/emails ({LAST}, {First}, "
            "{f}.{last}@{domain}). Empty string is treated as omitted. Only "
            "entity types that define a pattern syntax may set this. "
            "When provided, the whole column is replaced with the pattern if it "
            "covers at least 85% of non-null values (checked against the dataframe, "
            "not here)."
        ),
    )
    depends_on: list[ConditioningColumn] = Field(
        default_factory=list,
        description=(
            "Columns that condition the replacement of this column. "
            "Only entity types in ALLOWED_DEPENDS_ON may set this. Can omit "
            "ConditioningColumn.entity_type when the conditioner is listed in "
            "columns_to_replace. Matrix checks for explicit entity_type run here; "
            "omitted types and exclusive-group checks are resolved on PiiReplacementPlan."
        ),
    )

    @field_validator("pattern", mode="before")
    @classmethod
    def _empty_pattern_is_omitted(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @model_validator(mode="after")
    def _validate_entity_and_depends_on(self) -> Self:
        entity = ENTITY_BY_TYPE[self.entity_type]
        if entity.action is EntityAction.IDENTIFY_ONLY:
            raise ParameterError(
                f"column {self.column_name!r}: entity_type {self.entity_type.value!r} is "
                "identify-only (or otherwise not replaceable); omit it from columns_to_replace "
            )
        if self.pattern is not None and entity.pattern_syntax is None:
            raise ParameterError(
                f"column {self.column_name!r}: entity_type {self.entity_type.value!r} does "
                "not allow pattern; omit pattern (replacement uses the entity generator)"
            )
        if self.depends_on:
            allowed = ALLOWED_DEPENDS_ON.get(self.entity_type, frozenset())
            if not allowed:
                raise ParameterError(
                    f"column {self.column_name!r}: entity_type {self.entity_type.value!r} does not allow depends_on"
                )
            for dep in self.depends_on:
                if dep.entity_type is None:
                    continue
                if dep.entity_type not in allowed:
                    raise ParameterError(
                        f"column {self.column_name!r}: depends_on entity_type "
                        f"{dep.entity_type.value!r} is not allowed for entity_type "
                        f"{self.entity_type.value!r} (allowed: "
                        f"{sorted(k.value for k in allowed)})"
                    )
        return self


class PiiReplacementScope(StrEnum):
    """Unit at which original→synthetic mappings stay consistent."""

    RECORD = "record"
    """Map each row independently; one original may differ across rows."""

    GROUP = "group"
    """One mapping per group, keyed by ``data.group_training_examples_by``."""

    DATAFRAME = "dataframe"
    """One mapping dataset-wide; an original always yields the same value."""


class PiiReplacementPlan(Parameters):
    """Dataset-specific detection/replacement plan (column-oriented).

    Flat ``columns_to_replace`` list; cross-column consistency is expressed only
    via ``depends_on`` edges (a DAG). Plan-vs-dataframe and graph checks live in
    ``pii_replacer.planning.validation``.
    """

    scope: PiiReplacementScope = Field(
        default=PiiReplacementScope.DATAFRAME,
        description="How widely one original value keeps the same synthetic value: record, group, or dataframe.",
    )
    columns_to_replace: list[PiiColumnPlan] = Field(
        default_factory=list,
        description="Columns to replace or (for free_text) scan for value propagation.",
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
            allowed = ALLOWED_DEPENDS_ON.get(spec.entity_type, frozenset())
            resolved: list[ConditioningColumn] = []
            inferred_any = False
            for dep in spec.depends_on:
                entity_type = dep.entity_type
                if entity_type is None:
                    source = by_name.get(dep.column_name)
                    if source is None:
                        raise ParameterError(
                            f"column {spec.column_name!r}: depends_on column "
                            f"{dep.column_name!r} omits entity_type but is not listed in "
                            "columns_to_replace; set entity_type for read-only conditioners "
                            "(e.g. gender, ethnic_background, city)"
                        )
                    inferred = source.entity_type
                    if not can_condition(inferred):
                        raise ParameterError(
                            f"column {spec.column_name!r}: depends_on column "
                            f"{dep.column_name!r} has entity_type {inferred.value!r}, which "
                            "cannot be used as a conditioner"
                        )
                    if inferred not in allowed:
                        raise ParameterError(
                            f"column {spec.column_name!r}: depends_on column "
                            f"{dep.column_name!r} (entity_type {inferred.value!r}) is not "
                            f"allowed for entity_type {spec.entity_type.value!r} (allowed: "
                            f"{sorted(k.value for k in allowed)})"
                        )
                    dep = dep.model_copy(update={"entity_type": inferred})
                    inferred_any = True
                elif dep.column_name in by_name:
                    planned = by_name[dep.column_name].entity_type
                    if planned != entity_type:
                        raise ParameterError(
                            f"column {spec.column_name!r}: depends_on column "
                            f"{dep.column_name!r} has entity_type {entity_type.value!r} but "
                            f"columns_to_replace lists entity_type {planned.value!r}"
                        )
                elif is_columns_to_replace_type(entity_type):
                    raise ParameterError(
                        f"column {spec.column_name!r}: depends_on column "
                        f"{dep.column_name!r} has entity_type {entity_type.value!r}, which is "
                        "a replace target; list it in columns_to_replace"
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


class LLMConfig(NSSBaseModel):
    """Shared LLM settings for discovery and free-text replacement.

    Presence on ``ReplacePiiConfig.llm`` is itself the enable signal; there is no
    separate boolean flag. Reserved for future use.
    """

    model_provider: str | None = Field(
        default=None,
        description="Reserved: inference provider for LLM-assisted discovery/replacement.",
    )
    max_workers: int = Field(
        default=64,
        description="Reserved: maximum parallel workers for LLM-assisted operations.",
    )


class PiiReplacementSettings(NSSBaseModel):
    """Apply-time replacement basics."""

    locale: str = Field(
        default="en_US",
        description="Locale for generated names, addresses, and phone numbers.",
    )
    seed: int | None = Field(
        default=None,
        description="Seed for synthetic value generation; unset uses PERSON_RANDOM_SEED or 42.",
    )


class PiiSamplerBackend(StrEnum):
    """Source of synthetic values for names and related person-like fields."""

    # PGM = "pgm" # Internal generator

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

    ``replacement_plan`` is one of:

    * ``"auto_discovery"`` (default) -- detect and plan replacements automatically;
    * a path (string) to a plan file;
    * an inline ``PiiReplacementPlan``.

    v2 fields ``globals`` and ``steps`` are rejected with a removal error.
    """

    removed_legacy_fields: ClassVar[frozenset[str]] = frozenset({"globals", "steps"})
    removed_legacy_fields_message: ClassVar[str] = (
        "PII replacement v2 configuration was removed. "
        "See docs/user-guide/configuration.md#replacing-pii "
        "for the current configuration."
    )

    schema_version: Literal[1] = Field(
        default=1,
        description=(
            "Version of this replace_pii config shape. Bump when fields or plan "
            "semantics change incompatibly; only 1 is accepted in this release."
        ),
    )
    replacement_plan: PiiReplacementPlan | str = Field(
        default=AUTO_DISCOVERY,
        description=(
            f"{AUTO_DISCOVERY!r} to discover the plan from the data, a path to a plan "
            "file, or an inline plan (YAML/SDK mapping or PiiReplacementPlan). "
            f"Other strings are treated as paths. The CLI option only accepts the "
            "sentinel or a path."
        ),
    )
    llm: LLMConfig | None = Field(
        default=None,
        description=(
            "LLM-assisted discovery and free-text replacement settings, or null "
            "(the default) to run without LLM assistance. Reserved: supplying a "
            "value is rejected at config validation in this release."
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

    @field_validator("llm")
    @classmethod
    def _reject_unsupported_llm(cls, value: LLMConfig | None) -> LLMConfig | None:
        if value is not None:
            raise ParameterError(
                "replace_pii.llm is not supported in this release; omit it or set "
                "replace_pii.llm to null (the default)."
            )
        return value

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
                details = "; ".join(
                    f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()
                )
                raise ParameterError(f"invalid inline replacement plan ({details})") from exc
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, str | PiiReplacementPlan):
            return value
        raise ParameterError(
            f"replacement_plan must be {AUTO_DISCOVERY!r}, a path to a plan file, or an inline plan; "
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
        """The inline plan, or ``None`` if auto-discovery or a path reference."""
        return self.replacement_plan if isinstance(self.replacement_plan, PiiReplacementPlan) else None
