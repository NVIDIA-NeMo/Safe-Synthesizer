# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-specific PII detection/replacement plan.

This is the declarative, column-oriented plan a user (or an upstream detector)
provides to describe *how* to replace PII in a dataset.

Every label is an :class:`Entity` with an :class:`EntityAction` and whether it
``can_condition`` other columns. User-facing plans name them with
``entity_type`` on both replace targets and ``depends_on`` entries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
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
    "ENTITIES",
    "ENTITY_BY_TYPE",
    "EXCLUSIVE_DEPENDS_ON_GROUPS",
    "ConditioningColumn",
    "Entity",
    "EntityAction",
    "EntityType",
    "LLMConfig",
    "PiiColumnPlan",
    "PiiSamplerBackend",
    "PiiSamplerConfig",
    "PiiReplacementPlan",
    "PiiReplacementScope",
    "PiiReplacementSettings",
    "ReplacePiiConfig",
    "can_condition",
    "is_columns_to_replace_type",
]

# Sentinel value for ``ReplacePiiConfig.replacement_plan`` requesting automatic
# entity discovery instead of an explicit plan.
AUTO_DISCOVERY = "auto_discovery"


class EntityType(StrEnum):
    """Closed vocabulary for discovery and plan ``entity_type`` fields."""

    first_name = "first_name"
    middle_name = "middle_name"
    last_name = "last_name"
    full_name = "full_name"
    email = "email"
    phone_number = "phone_number"
    date_of_birth = "date_of_birth"
    street_address = "street_address"
    ssn = "ssn"
    national_id = "national_id"
    credit_debit_card = "credit_debit_card"
    api_key = "api_key"  # pragma: allowlist secret
    ipv4 = "ipv4"
    ipv6 = "ipv6"
    unique_identifier = "unique_identifier"

    # Propagate already-replaced values into cell text (username/url use this too).
    free_text = "free_text"

    # Identify-only (and often original-value conditioners): discovery may classify
    # these so they are not mistaken for free text / replace targets.
    date = "date"  # generic (non-birth) date
    gender = "gender"
    ethnic_background = "ethnic_background"
    city = "city"
    state = "state"
    zip_code = "zip_code"
    country = "country"
    organization = "organization"


class EntityAction(StrEnum):
    """What the engine does to a column of this entity type.

    ``replace``: synthesize a new cell value.
    ``propagate``: scan existing text and substitute already-replaced values.
    ``none``: do not appear on ``columns_to_replace`` (identify and/or condition only).
    """

    replace = "replace"
    propagate = "propagate"
    none = "none"


@dataclass(frozen=True, slots=True)
class Entity:
    """One label in the entity vocabulary.

    ``action``: engine treatment (see :class:`EntityAction`). ``can_condition``:
    may appear as ``entity_type`` on a ``depends_on`` entry. The executor
    applies ``columns_to_replace`` in DAG order and reads the current cell
    as the conditioner.
    """

    entity_type: EntityType
    action: EntityAction
    can_condition: bool


ENTITIES: tuple[Entity, ...] = (
    # Replace and may condition later columns.
    Entity(EntityType.first_name, action=EntityAction.replace, can_condition=True),
    Entity(EntityType.middle_name, action=EntityAction.replace, can_condition=True),
    Entity(EntityType.last_name, action=EntityAction.replace, can_condition=True),
    Entity(EntityType.full_name, action=EntityAction.replace, can_condition=True),
    # Replace only.
    Entity(EntityType.email, action=EntityAction.replace, can_condition=False),
    Entity(EntityType.phone_number, action=EntityAction.replace, can_condition=False),
    Entity(EntityType.date_of_birth, action=EntityAction.replace, can_condition=False),
    Entity(EntityType.street_address, action=EntityAction.replace, can_condition=False),
    Entity(EntityType.ssn, action=EntityAction.replace, can_condition=False),
    Entity(EntityType.national_id, action=EntityAction.replace, can_condition=False),
    Entity(EntityType.credit_debit_card, action=EntityAction.replace, can_condition=False),
    Entity(EntityType.api_key, action=EntityAction.replace, can_condition=False),
    Entity(EntityType.ipv4, action=EntityAction.replace, can_condition=False),
    Entity(EntityType.ipv6, action=EntityAction.replace, can_condition=False),
    Entity(EntityType.unique_identifier, action=EntityAction.replace, can_condition=False),
    # Scan cell text and propagate already-replaced values (username/url use this too).
    Entity(EntityType.free_text, action=EntityAction.propagate, can_condition=False),
    # Identify-only.
    Entity(EntityType.date, action=EntityAction.none, can_condition=False),
    # Identify-only and may condition replacements.
    Entity(EntityType.gender, action=EntityAction.none, can_condition=True),
    Entity(EntityType.ethnic_background, action=EntityAction.none, can_condition=True),
    Entity(EntityType.city, action=EntityAction.none, can_condition=True),
    Entity(EntityType.state, action=EntityAction.none, can_condition=True),
    Entity(EntityType.zip_code, action=EntityAction.none, can_condition=True),
    Entity(EntityType.country, action=EntityAction.none, can_condition=True),
    Entity(EntityType.organization, action=EntityAction.none, can_condition=True),
)

ENTITY_BY_TYPE: dict[EntityType, Entity] = {entity.entity_type: entity for entity in ENTITIES}

# entity_type → allowed depends_on entity types (optional edges may be omitted).
ALLOWED_DEPENDS_ON: dict[EntityType, frozenset[EntityType]] = {
    EntityType.first_name: frozenset({EntityType.gender, EntityType.ethnic_background, EntityType.full_name}),
    EntityType.middle_name: frozenset({EntityType.gender, EntityType.ethnic_background, EntityType.full_name}),
    EntityType.last_name: frozenset({EntityType.ethnic_background, EntityType.full_name}),
    EntityType.full_name: frozenset({EntityType.gender, EntityType.ethnic_background}),
    EntityType.email: frozenset(
        {
            EntityType.first_name,
            EntityType.middle_name,
            EntityType.last_name,
            EntityType.full_name,
            EntityType.organization,
        }
    ),
    EntityType.street_address: frozenset(
        {EntityType.city, EntityType.state, EntityType.zip_code, EntityType.country}
    ),
}

# Each inner tuple is one exclusivity family: at most one group from that family
# may appear in a single depends_on list. Families are independent.
# Applied to every columns_to_replace entry after entity_type inference.
EXCLUSIVE_DEPENDS_ON_GROUPS: tuple[tuple[frozenset[EntityType], ...], ...] = (
    (
        frozenset({EntityType.first_name, EntityType.middle_name, EntityType.last_name}),
        frozenset({EntityType.full_name}),
    ),
    (
        frozenset({EntityType.full_name}),
        frozenset({EntityType.gender, EntityType.ethnic_background}),
    ),
    (
        frozenset({EntityType.zip_code}),
        frozenset({EntityType.city, EntityType.state, EntityType.country}),
    ),
)


def is_columns_to_replace_type(entity_type: EntityType) -> bool:
    """Whether ``entity_type`` may appear on ``columns_to_replace``."""
    return ENTITY_BY_TYPE[entity_type].action is not EntityAction.none


def can_condition(entity_type: EntityType) -> bool:
    """Whether ``entity_type`` may appear on a ``depends_on`` entry."""
    return ENTITY_BY_TYPE[entity_type].can_condition


class ConditioningColumn(NSSBaseModel):
    """An existing column that conditions the replacement of another column.

    ``entity_type`` is required for **read-only** conditioners (``gender``,
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
            "{f}.{last}@{domain}). When a pattern is provided, the whole column is "
            "replaced with the pattern."
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

    @model_validator(mode="after")
    def _validate_entity_and_depends_on(self) -> Self:
        action = ENTITY_BY_TYPE[self.entity_type].action
        if action is EntityAction.none:
            raise ParameterError(
                f"column {self.column_name!r}: entity_type {self.entity_type.value!r} is "
                "identify-only (or otherwise not replaceable); omit it from columns_to_replace "
            )
        if self.depends_on:
            allowed = ALLOWED_DEPENDS_ON.get(self.entity_type, frozenset())
            if not allowed:
                raise ParameterError(
                    f"column {self.column_name!r}: entity_type {self.entity_type.value!r} "
                    "does not allow depends_on"
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

    record = "record"
    group = "group"
    dataframe = "dataframe"


class PiiReplacementPlan(Parameters):
    """Dataset-specific detection/replacement plan (column-oriented).

    Flat ``columns_to_replace`` list; cross-column consistency is expressed only
    via ``depends_on`` edges (a DAG). Plan-vs-dataframe and graph checks live in
    ``pii_replacer.planning.validation``.
    """

    scope: PiiReplacementScope = Field(
        default=PiiReplacementScope.dataframe,
        description="How widely one original value keeps the same synthetic value: record, group, or dataframe.",
    )
    columns_to_replace: list[PiiColumnPlan] = Field(
        default_factory=list,
        description="Columns to replace or (for free_text) scan for value propagation.",
    )

    @model_validator(mode="after")
    def _resolve_omitted_depends_on_types(self) -> Self:
        """Infer missing depends_on.entity_type from columns_to_replace entity_type."""
        by_name = {spec.column_name: spec for spec in self.columns_to_replace}
        for spec in self.columns_to_replace:
            if not spec.depends_on:
                continue
            allowed = ALLOWED_DEPENDS_ON.get(spec.entity_type, frozenset())
            resolved: list[ConditioningColumn] = []
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
                elif dep.column_name in by_name:
                    planned = by_name[dep.column_name].entity_type
                    if planned != entity_type:
                        raise ParameterError(
                            f"column {spec.column_name!r}: depends_on column "
                            f"{dep.column_name!r} has entity_type {entity_type.value!r} but "
                            f"columns_to_replace lists entity_type {planned.value!r}"
                        )
                resolved.append(dep)
            spec.depends_on = resolved
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
                    f"column {spec.column_name!r}: depends_on mixes mutually exclusive "
                    f"conditioner groups: {formatted}"
                )
        return self


class LLMConfig(NSSBaseModel):
    """Shared LLM settings for discovery and free-text replacement (reserved for future use)."""

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
    # pgm = "pgm" # Internal generator
    managed = "managed"
    faker = "faker"


class PiiSamplerConfig(NSSBaseModel):
    """Settings for the synthetic value sampler (names and related person-like fields)."""

    backend: PiiSamplerBackend = Field(
        default=PiiSamplerBackend.managed,
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
        if self.managed_assets_path is not None:
            return Path(self.managed_assets_path)
        return default_managed_assets_path()


class ReplacePiiConfig(Parameters):
    """Top-level ``replace_pii`` config wrapping the replacement plan.

    ``replacement_plan`` is one of:
    * ``"auto_discovery"`` (default) -- detect and plan replacements automatically;
    * a path (string) to a plan file;
    * an inline :class:`PiiReplacementPlan`.

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
    llm_enhancement: bool = Field(
        default=False,
        description=(
            "Reserved for LLM-assisted discovery and free-text detection. "
            "Must remain false in this release; true is rejected at config validation."
        ),
    )
    replacement_plan: PiiReplacementPlan | str = Field(
        default=AUTO_DISCOVERY,
        description=(
            f"{AUTO_DISCOVERY!r} to discover the plan from the data, or a path to a plan file. "
            "An inline plan can only be given in a config file."
        ),
    )
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Reserved LLM settings for discovery and free-text replacement.",
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

    @field_validator("llm_enhancement")
    @classmethod
    def _reject_unsupported_llm_enhancement(cls, value: bool) -> bool:
        if value:
            raise ParameterError(
                "replace_pii.llm_enhancement=True is not supported in this release; "
                "set replace_pii.llm_enhancement to false (the default)."
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
