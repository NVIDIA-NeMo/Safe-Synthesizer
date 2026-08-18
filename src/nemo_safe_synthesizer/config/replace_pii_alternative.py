# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Archived dual-enum draft of the PII replacement plan config.

This is the pre–option-A shape that used separate :class:`PiiEntity` and
:class:`ConditioningColumnType` enums (with identify-only types listed on
``PiiEntity``). The canonical draft is :mod:`replace_pii`, which uses a single
:class:`~replace_pii.ColumnKind` vocabulary plus capability tags.

Kept for comparison only; not imported by the package.
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
    "AUTO_DISCOVERY",
    "LLMConfig",
    "PiiEntity",
    "ConditioningColumn",
    "ConditioningColumnType",
    "PiiColumnPlan",
    "PiiPersonBackend",
    "PiiPersonConfig",
    "PiiReplacementPlan",
    "PiiReplacementScope",
    "PiiReplacementSettings",
    "ReplacePiiConfig",
]

# Sentinel value for ``ReplacePiiConfig.replacement_plan`` requesting automatic
# entity discovery instead of an explicit plan.
AUTO_DISCOVERY = "auto_discovery"


class PiiEntity(StrEnum):
    """Closed vocabulary of entity types a replacement plan may name.

    Most members are replaceable. ``date`` is identify-only: discovery uses it
    to keep a generic temporal column out of free text, and a plan that lists it
    is refused. ``free_text`` marks a column for value propagation rather than
    whole-column replacement.
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

    # Column is free text; replaced values are propagated into it rather than
    # replacing the cell as a single structured value.
    free_text = "free_text"

    # Identify-only: discovery excludes these from
    # the replacement plan; naming one in a plan is an error.
    date = "date"  # a generic (non-birth) date.
    sex = "sex"
    ethnic_background = "ethnic_background"
    city = "city"
    state = "state"
    zip_code = "zip_code"
    country = "country"
    organization = "organization"


class ConditioningColumnType(StrEnum):
    """Type of column that conditions the replacement of another column."""

    # These are also replaceable entities. We will use the replacement values to
    # condition the new column.
    first_name = "first_name"
    middle_name = "middle_name"
    last_name = "last_name"

    # These columns are not replaceable. They are read-only for the purpose of
    # conditioning the replacement of another column.
    sex = "sex"
    ethnic_background = "ethnic_background"
    city = "city"
    state = "state"
    zip_code = "zip_code"
    country = "country"
    organization = "organization"


class ConditioningColumn(NSSBaseModel):
    """An existing column that conditions the replacement of another column.

    ``column_type: sex`` with ``column_name: sex`` conditions the replacement of
    another column to agree with each row's original ``sex`` value.
    """

    column_name: str = Field(description="Existing dataframe column that supplies the demographic value.")
    column_type: ConditioningColumnType = Field(
        description="Type of column that conditions the replacement of another column, eg. sex, ethnic_background, zip_code."
    )


class PiiColumnPlan(NSSBaseModel):
    """Replacement spec for one named column.

    Set ``entity_type: free_text`` for columns where replaced values are
    propagated into the cell text rather than replacing the cell as a single
    structured value. Free-text columns cannot declare ``depends_on``.
    """

    column_name: str = Field(description="Name of the dataframe column to replace or scan.")
    entity_type: PiiEntity | None = Field(
        default=None,
        description="Entity type this column holds (required for a usable plan entry).",
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
            "Not allowed when entity_type is free_text."
        ),
    )

    @model_validator(mode="after")
    def _reject_depends_on_for_free_text(self) -> Self:
        if self.entity_type == PiiEntity.free_text and self.depends_on:
            raise ParameterError(
                f"column {self.column_name!r}: depends_on is not allowed when entity_type is free_text"
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
    via ``depends_on`` edges (a DAG).
    """

    scope: PiiReplacementScope = Field(
        default=PiiReplacementScope.dataframe,
        description="How widely one original value keeps the same synthetic value: record, group, or dataframe.",
    )
    columns_to_replace: list[PiiColumnPlan] = Field(
        default_factory=list,
        description="Columns to replace.",
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
        description=("Persona sampler backend: managed assets or Faker."),
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
