# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-specific PII detection/replacement plan.

This is the declarative, column-oriented plan a user (or an upstream detector)
provides to describe *how* to replace PII in a dataset.

Column semantics use one vocabulary (:class:`ColumnKind`) with explicit
capabilities (:data:`COLUMN_CAPABILITIES`) instead of parallel enums for
replaceable vs conditioning types.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, ValidationError, field_validator, model_validator

from ..configurator.parameters import Parameters
from ..defaults import NSS_MANAGED_ASSETS_PATH_ENV, default_managed_assets_path
from ..errors import ParameterError
from .base import NSSBaseModel

__all__ = [
    "ALLOWED_DEPENDS_ON",
    "AUTO_DISCOVERY",
    "COLUMN_CAPABILITIES",
    "ColumnCapability",
    "ColumnKind",
    "ConditioningColumn",
    "LLMConfig",
    "PiiColumnPlan",
    "PiiPersonBackend",
    "PiiPersonConfig",
    "PiiReplacementPlan",
    "PiiReplacementScope",
    "PiiReplacementSettings",
    "ReplacePiiConfig",
    "can_be_entity_type",
    "can_condition",
    "uses_original_conditioner",
    "uses_synthetic_conditioner",
]

# Sentinel value for ``ReplacePiiConfig.replacement_plan`` requesting automatic
# entity discovery instead of an explicit plan.
AUTO_DISCOVERY = "auto_discovery"


class ColumnCapability(StrEnum):
    """Roles a :class:`ColumnKind` may play.

    Overlaps are intentional (e.g. ``first_name`` is both replaceable and a
    synthetic conditioner). Identify-only kinds are for discovery/classification
    and must not appear as ``entity_type`` on a replace plan entry.
    """

    replace = "replace"
    propagate = "propagate"
    identify = "identify"
    condition_synthetic = "condition_synthetic"
    condition_original = "condition_original"


class ColumnKind(StrEnum):
    """Single closed vocabulary for discovery, plan ``entity_type``, and ``depends_on``.

    Capabilities are declared in :data:`COLUMN_CAPABILITIES`, not by splitting into
    parallel enums.
    """

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
    sex = "sex"
    ethnic_background = "ethnic_background"
    city = "city"
    state = "state"
    zip_code = "zip_code"
    country = "country"
    organization = "organization"


COLUMN_CAPABILITIES: dict[ColumnKind, frozenset[ColumnCapability]] = {
    # Name parts: replace + condition later fields on the synthetic value.
    ColumnKind.first_name: frozenset(
        {ColumnCapability.replace, ColumnCapability.condition_synthetic}
    ),
    ColumnKind.middle_name: frozenset(
        {ColumnCapability.replace, ColumnCapability.condition_synthetic}
    ),
    ColumnKind.last_name: frozenset(
        {ColumnCapability.replace, ColumnCapability.condition_synthetic}
    ),
    ColumnKind.full_name: frozenset({ColumnCapability.replace}),
    ColumnKind.email: frozenset({ColumnCapability.replace}),
    ColumnKind.phone_number: frozenset({ColumnCapability.replace}),
    ColumnKind.date_of_birth: frozenset({ColumnCapability.replace}),
    ColumnKind.street_address: frozenset({ColumnCapability.replace}),
    ColumnKind.ssn: frozenset({ColumnCapability.replace}),
    ColumnKind.national_id: frozenset({ColumnCapability.replace}),
    ColumnKind.credit_debit_card: frozenset({ColumnCapability.replace}),
    ColumnKind.api_key: frozenset({ColumnCapability.replace}),
    ColumnKind.ipv4: frozenset({ColumnCapability.replace}),
    ColumnKind.ipv6: frozenset({ColumnCapability.replace}),
    ColumnKind.unique_identifier: frozenset({ColumnCapability.replace}),
    ColumnKind.free_text: frozenset({ColumnCapability.propagate}),
    # Identify-only / read-only conditioners.
    ColumnKind.date: frozenset({ColumnCapability.identify}),
    ColumnKind.sex: frozenset(
        {ColumnCapability.identify, ColumnCapability.condition_original}
    ),
    ColumnKind.ethnic_background: frozenset(
        {ColumnCapability.identify, ColumnCapability.condition_original}
    ),
    ColumnKind.city: frozenset(
        {ColumnCapability.identify, ColumnCapability.condition_original}
    ),
    ColumnKind.state: frozenset(
        {ColumnCapability.identify, ColumnCapability.condition_original}
    ),
    ColumnKind.zip_code: frozenset(
        {ColumnCapability.identify, ColumnCapability.condition_original}
    ),
    ColumnKind.country: frozenset(
        {ColumnCapability.identify, ColumnCapability.condition_original}
    ),
    ColumnKind.organization: frozenset(
        {ColumnCapability.identify, ColumnCapability.condition_original}
    ),
}

# entity_type → allowed depends_on column kinds (optional edges may be omitted).
ALLOWED_DEPENDS_ON: dict[ColumnKind, frozenset[ColumnKind]] = {
    ColumnKind.first_name: frozenset({ColumnKind.sex, ColumnKind.ethnic_background}),
    ColumnKind.middle_name: frozenset({ColumnKind.sex, ColumnKind.ethnic_background}),
    ColumnKind.last_name: frozenset({ColumnKind.ethnic_background}),
    ColumnKind.full_name: frozenset({ColumnKind.sex, ColumnKind.ethnic_background}),
    ColumnKind.email: frozenset(
        {
            ColumnKind.first_name,
            ColumnKind.middle_name,
            ColumnKind.last_name,
            ColumnKind.organization,
        }
    ),
    ColumnKind.street_address: frozenset(
        {ColumnKind.city, ColumnKind.state, ColumnKind.zip_code, ColumnKind.country}
    ),
}


def can_be_entity_type(kind: ColumnKind) -> bool:
    """Whether ``kind`` may appear as ``PiiColumnPlan.entity_type``."""
    caps = COLUMN_CAPABILITIES[kind]
    return ColumnCapability.replace in caps or ColumnCapability.propagate in caps


def can_condition(kind: ColumnKind) -> bool:
    """Whether ``kind`` may appear as ``ConditioningColumn.column_type``."""
    caps = COLUMN_CAPABILITIES[kind]
    return (
        ColumnCapability.condition_synthetic in caps
        or ColumnCapability.condition_original in caps
    )


def uses_synthetic_conditioner(kind: ColumnKind) -> bool:
    """Conditioner that must use the already-replaced (synthetic) value."""
    return ColumnCapability.condition_synthetic in COLUMN_CAPABILITIES[kind]


def uses_original_conditioner(kind: ColumnKind) -> bool:
    """Conditioner that must use the original dataframe value (read-only)."""
    return ColumnCapability.condition_original in COLUMN_CAPABILITIES[kind]


class ConditioningColumn(NSSBaseModel):
    """An existing column that conditions the replacement of another column.

    ``column_type`` must be a :class:`ColumnKind` with a ``condition_*`` capability.
    Synthetic conditioners (e.g. ``first_name``) use replacement values; original
    conditioners (e.g. ``sex``) use the row's existing value and are not replaced.
    """

    column_name: str = Field(description="Existing dataframe column that supplies the conditioning value.")
    column_type: ColumnKind = Field(
        description="Semantic kind of the conditioning column (must be a conditioner capability).",
    )

    @model_validator(mode="after")
    def _require_conditioner_capability(self) -> Self:
        if not can_condition(self.column_type):
            raise ParameterError(
                f"column_type {self.column_type.value!r} cannot be used in depends_on "
                f"(allowed conditioner kinds: "
                f"{sorted(k.value for k in ColumnKind if can_condition(k))})"
            )
        return self


class PiiColumnPlan(NSSBaseModel):
    """Replacement spec for one named column.

    ``entity_type`` must be a :class:`ColumnKind` with ``replace`` or ``propagate``.
    Identify-only kinds (``date``, ``sex``, ``city``, …) are invalid here.
    Free-text columns cannot declare ``depends_on``.
    """

    column_name: str = Field(description="Name of the dataframe column to replace or scan.")
    entity_type: ColumnKind | None = Field(
        default=None,
        description=(
            "Column kind this entry holds. Must be replaceable or free_text; "
            "identify-only kinds are refused."
        ),
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
            "Not allowed when entity_type is free_text. Full matrix / DAG checks "
            "live in planning validation; this model only enforces cheap shape rules."
        ),
    )

    @model_validator(mode="after")
    def _validate_entity_and_depends_on(self) -> Self:
        if self.entity_type is not None and not can_be_entity_type(self.entity_type):
            raise ParameterError(
                f"column {self.column_name!r}: entity_type {self.entity_type.value!r} is "
                "identify-only (or otherwise not replaceable); omit it from columns_to_replace "
                "or use it only as a depends_on conditioner"
            )
        if self.entity_type == ColumnKind.free_text and self.depends_on:
            raise ParameterError(
                f"column {self.column_name!r}: depends_on is not allowed when entity_type is free_text"
            )
        if self.entity_type is not None and self.depends_on:
            allowed = ALLOWED_DEPENDS_ON.get(self.entity_type, frozenset())
            if not allowed:
                raise ParameterError(
                    f"column {self.column_name!r}: entity_type {self.entity_type.value!r} "
                    "does not allow depends_on"
                )
            for dep in self.depends_on:
                if dep.column_type not in allowed:
                    raise ParameterError(
                        f"column {self.column_name!r}: depends_on column_type "
                        f"{dep.column_type.value!r} is not allowed for entity_type "
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

    Flat ``columns_to_replace`` list; person-like consistency is expressed only
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


class PiiPersonBackend(StrEnum):
    # pgm = "pgm" # Internal generator
    managed = "managed"
    faker = "faker"


class PiiPersonConfig(NSSBaseModel):
    """Synthetic-person generation settings."""

    backend: PiiPersonBackend = Field(
        default=PiiPersonBackend.managed,
        description=(
            "Persona sampler backend: managed assets or Faker."
        ),
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
    """

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
    person: PiiPersonConfig = Field(
        default_factory=PiiPersonConfig,
        description="Synthetic-person sampler backend and asset paths.",
    )

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
