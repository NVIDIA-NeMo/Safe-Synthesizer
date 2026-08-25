# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-specific PII detection/replacement plan.

This is the declarative, column-oriented plan a user (or an upstream detector)
provides to describe *how* to replace PII in a dataset. A plan has two halves:

* ``persona_backed_columns`` -- columns that describe a person. Each entry is one
  persona (patient, provider, ...) whose columns are filled from a single
  synthetic identity, so first name, last name, and email stay consistent with
  each other.
* ``standalone_columns_to_replace`` -- columns replaced on their own, with no
  persona behind them (record IDs, free-text notes, ...).
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
    "KeyDomain",
    "LLMConfig",
    "PiiEntity",
    "PersonaMatchColumn",
    "PiiColumnPlan",
    "PersonaColumnSet",
    "PiiPersonBackend",
    "PiiPersonConfig",
    "PiiReplacementPlan",
    "PiiReplacementScope",
    "PiiReplacementSettings",
    "ReplacePiiConfig",
    "PolymorphicForeignKeyPlan",
    "PolymorphicFkTarget",
    "TableReplacementPlan",
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

    # Identify-only: a generic (non-birth) date. Discovery excludes these from
    # the replacement plan; naming one in a plan is an error.
    date = "date"
    # Column is free text; replaced values are propagated into it rather than
    # replacing the cell as a single structured value.
    free_text = "free_text"


class PersonaMatchColumn(NSSBaseModel):
    """An existing column that decides which kind of persona to generate.

    ``persona_attribute: sex`` with ``column_name: sex`` picks synthetic names
    that agree with each row's original ``sex`` value. ``ethnic_background`` is
    supported under the ``managed`` and ``pgm`` backends only; with ``faker`` it
    is omitted from auto-discovered plans and ignored if present in a hand-written
    plan. The named column is read, never replaced.
    """

    persona_attribute: Literal["sex", "ethnic_background"] = Field(
        description="Demographic attribute used to condition the synthetic persona (sex or ethnic_background)."
    )
    column_name: str = Field(description="Existing dataframe column that supplies the demographic value.")


class PiiColumnPlan(NSSBaseModel):
    """Replacement spec for one named column.

    Used in both plan sections. Set ``entity_type: free_text`` for columns where
    replaced values are propagated into the cell text rather than replacing the
    cell as a single structured value.
    """

    column_name: str = Field(description="Name of the dataframe column to replace or scan.")
    entity_type: PiiEntity | None = Field(
        default=None,
        description="Entity type this column holds (required for a usable plan entry).",
    )
    patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Formats this column writes the entity in, most common first: strftime for birth "
            "dates (%m/%d/%Y), character templates for identifiers/phones (pmc-######, "
            "+1-###-555-####), or person-part placeholders for names/emails ({LAST}, {First}, "
            "{f}.{last}@{domain}). A value uses the first pattern that describes it; "
            "otherwise it keeps its own shape."
        ),
    )


class PiiReplacementScope(StrEnum):
    """Unit at which original→synthetic mappings stay consistent.

    ``record``, ``group``, and ``dataframe`` apply within a single table.
    ``database`` applies across a folder of related tables (schema + shared
    runtime store); use ``MultiTablePiiReplacer``, not the full Safe Synthesizer
    pipeline.
    """

    record = "record"
    group = "group"
    dataframe = "dataframe"
    database = "database"


class PersonaColumnSet(NSSBaseModel):
    """Columns grouped under one named persona in the plan.

    Persona-sourced entities listed here share one synthetic identity (e.g.
    patient, doctor). Entity-driven types (IDs, cards, DOB, non-``pgm`` phones)
    still replace from their own values even if listed here; see placement
    advisories. ``match_persona_by`` lists existing columns that constrain which
    persona is drawn.
    """

    persona: str = Field(description="Label for this persona set (e.g. patient, provider).")
    columns_to_replace: list[PiiColumnPlan] = Field(
        default_factory=list,
        description="Columns replaced from (or associated with) this persona.",
    )
    match_persona_by: list[PersonaMatchColumn] = Field(
        default_factory=list,
        description="Existing columns that constrain which synthetic persona is drawn (read, never replaced).",
    )
    person_key_domain: str | None = Field(
        default=None,
        description=(
            "For database scope: key-domain id that identities this persona bundle hang off "
            "(e.g. patients.id). Defaults to the table's PK domain when applicable."
        ),
    )


class KeyDomain(NSSBaseModel):
    """Database scope only: equivalence class of columns that share one map.

    Built from schema PK/FK links (and optional value-overlap bundling). When
    ``person_reference`` is true, person attribute bundles hang off keys in this
    domain. Ordinary FK child columns appear here; polymorphic Id columns do not
    (they route per row into parent domains).
    """

    id: str = Field(description="Stable domain id; prefer the PK table.column (e.g. patients.id).")
    person_reference: bool = Field(
        default=False,
        description="When true, person bundles hang off keys in this domain.",
    )
    columns: list[str] = Field(
        default_factory=list,
        description="Table-qualified columns in this domain (e.g. patients.id, events.patient_id).",
    )


class PolymorphicFkTarget(NSSBaseModel):
    """Database scope only: one parent domain a polymorphic Id may route into."""

    type_value: str = Field(description="Value of the type discriminator (e.g. patient).")
    domain: str = Field(description="Key-domain id for that parent (e.g. patients.id).")


class PolymorphicForeignKeyPlan(NSSBaseModel):
    """Database scope only: per-row router from a polymorphic Id into parent domains."""

    column: str = Field(description="Table-qualified polymorphic Id column (e.g. notes.subject_id).")
    type_column: str = Field(
        description="Table-qualified type discriminator (e.g. notes.subject_type); not replaced."
    )
    targets: list[PolymorphicFkTarget] = Field(
        description="type_value → parent domain mappings.",
    )


class TableReplacementPlan(NSSBaseModel):
    """Database scope only: per-table persona + standalone sections of the plan."""

    persona_backed_columns: list[PersonaColumnSet] = Field(
        default_factory=list,
        description="Persona sets whose columns share one synthetic identity.",
    )
    standalone_columns_to_replace: list[PiiColumnPlan] = Field(
        default_factory=list,
        description="Columns replaced independently of any persona (IDs, free text, entity-driven values).",
    )


class PiiReplacementPlan(Parameters):
    """Dataset-specific detection/replacement plan (column-oriented).

    For single-table scopes (``record`` / ``group`` / ``dataframe``), columns under
    ``persona_backed_columns`` are replaced together as one synthetic person and
    ``standalone_columns_to_replace`` are replaced independently.

    For ``scope: database``, use ``key_domains`` plus ``tables`` (per-table plan
    bodies with table-qualified column names). Top-level persona/standalone lists
    must be empty. ``polymorphic_foreign_keys`` is omitted when the schema has none.
    """

    scope: PiiReplacementScope = Field(
        default=PiiReplacementScope.dataframe,
        description=(
            "How widely one original value keeps the same synthetic value: record, group, "
            "dataframe (single table), or database (multi-table folder + schema)."
        ),
    )
    persona_backed_columns: list[PersonaColumnSet] = Field(
        default_factory=list,
        description="Persona sets whose columns share one synthetic identity (single-table scopes).",
    )
    standalone_columns_to_replace: list[PiiColumnPlan] = Field(
        default_factory=list,
        description="Columns replaced independently of any persona (single-table scopes).",
    )
    key_domains: list[KeyDomain] = Field(
        default_factory=list,
        description="Cross-table key domains for database scope (PK/FK equivalence classes).",
    )
    polymorphic_foreign_keys: list[PolymorphicForeignKeyPlan] | None = Field(
        default=None,
        description=(
            "Database scope only: polymorphic Id routers "
            "(e.g. notes.subject_id + notes.subject_type → patients.id or providers.id). "
            "Omit entirely when the schema has no polymorphic FKs."
        ),
    )
    tables: dict[str, TableReplacementPlan] = Field(
        default_factory=dict,
        description="Per-table plan bodies for database scope (table-qualified column names).",
    )

    @model_validator(mode="after")
    def _validate_scope_shape(self) -> Self:
        if self.scope == PiiReplacementScope.database:
            if not self.tables:
                raise ValueError("database-scope plan requires a non-empty tables mapping")
            if self.persona_backed_columns or self.standalone_columns_to_replace:
                raise ValueError(
                    "database-scope plan must put persona/standalone columns under tables.<Name>, "
                    "not at the top level"
                )
            if self.polymorphic_foreign_keys is not None and len(self.polymorphic_foreign_keys) == 0:
                # Normalize empty list → omitted
                self.polymorphic_foreign_keys = None
            self._validate_key_domain_membership()
        else:
            if self.key_domains or self.tables or self.polymorphic_foreign_keys:
                raise ValueError(
                    f"key_domains, tables, and polymorphic_foreign_keys are only valid when "
                    f"scope is 'database' (got scope={self.scope.value!r})"
                )
        return self

    def _validate_key_domain_membership(self) -> None:
        """Key domains partition columns; polymorphic Ids route instead of joining one."""
        polymorphic_columns = {entry.column for entry in self.polymorphic_foreign_keys or []}
        domain_by_column: dict[str, str] = {}
        for domain in self.key_domains:
            for column in domain.columns:
                owner = domain_by_column.setdefault(column, domain.id)
                if owner != domain.id:
                    raise ValueError(
                        f"column {column!r} appears in key domains {owner!r} and {domain.id!r}; "
                        "each column belongs to exactly one key domain"
                    )
                if column in polymorphic_columns:
                    raise ValueError(
                        f"polymorphic column {column!r} must not be listed in key domain "
                        f"{domain.id!r}; it is routed per row into a parent domain"
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
    # 'pgm' is an internal generator: it is not distributed with Safe Synthesizer and
    # the run fails if its source tree is absent (see replacement.personas._load_pgm_generator).
    pgm = "pgm"
    managed = "managed"
    faker = "faker"


class PiiPersonConfig(NSSBaseModel):
    """Synthetic-person generation settings."""

    backend: PiiPersonBackend = Field(
        default=PiiPersonBackend.managed,
        description=(
            "Persona sampler backend: managed assets or Faker. 'pgm' is internal-only and needs a local "
            "sdg-pgms checkout; it is the only backend that supplies a phone number."
        ),
    )
    sdg_pgms_src: str | None = Field(
        default=None,
        description=(
            "Source tree for sdg-pgms when backend is 'pgm'. Required for that backend; ignored otherwise. "
            "The 'pgm' backend never falls back if this path is missing or unusable."
        ),
    )
    managed_assets_path: str | None = Field(
        default=None,
        description=(
            "Root directory containing a datasets/ folder of locale parquet files. "
            f"Defaults to {NSS_MANAGED_ASSETS_PATH_ENV} or ~/.data-designer/managed-assets."
        ),
    )

    @model_validator(mode="after")
    def _require_sdg_pgms_src_for_pgm(self) -> Self:
        if self.backend == PiiPersonBackend.pgm and self.sdg_pgms_src is None:
            raise ParameterError(
                "replace_pii.person.sdg_pgms_src is required when replace_pii.person.backend is 'pgm'."
            )
        return self

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
    schema_path: str | None = Field(
        default=None,
        description=(
            "Path to a multi-table schema YAML (PK/FK). Required for database-scope "
            "replacement via MultiTablePiiReplacer; ignored for single-table scopes."
        ),
    )

    @field_validator("schema_path", mode="before")
    @classmethod
    def _coerce_schema_path(cls, value: object) -> object:
        if isinstance(value, Path):
            return str(value)
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
