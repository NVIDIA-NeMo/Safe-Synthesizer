# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-specific PII detection/replacement plan.

This is the declarative, column-oriented plan a user (or an upstream detector)
provides to describe *how* to replace PII in a dataset: which columns belong to
which role/identity, how names are conditioned on demographics, and how
standalone (unassociated) columns are treated.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field

from ..configurator.parameters import Parameters
from ..defaults import NSS_MANAGED_ASSETS_PATH_ENV, default_managed_assets_path
from .base import NSSBaseModel

__all__ = [
    "AUTO_DISCOVERY",
    "LLMConfig",
    "PiiDiscoveryConfig",
    "PiiEntity",
    "PiiConditioningColumns",
    "PiiColumnPlan",
    "AssociatedColumnSet",
    "PiiPersonBackend",
    "PiiPersonConfig",
    "PiiReplacementPlan",
    "PiiReplacementSettings",
    "ReplacePiiConfig",
]

# Sentinel value for ``ReplacePiiConfig.replacement_plan`` requesting automatic
# entity discovery instead of an explicit plan.
AUTO_DISCOVERY = "auto_discovery"


class PiiEntity(StrEnum):
    """Closed vocabulary of replaceable entity types."""

    first_name = "first_name"
    last_name = "last_name"
    full_name = "full_name"
    email = "email"
    phone_number = "phone_number"
    date_of_birth = "date_of_birth"
    street_address = "street_address"
    ssn = "ssn"
    national_id = "national_id"
    credit_debit_card = "credit_debit_card"
    api_key = "api_key"
    ipv4 = "ipv4"
    ipv6 = "ipv6"
    unique_identifier = "unique_identifier"

    # Special: generic date column (not a birth date). Used only to mark a column as
    # Special: structured (so it is not treated as free text); generic dates and other
    # temporal types (datetime/time/duration) are NOT replaced and are excluded
    # from the replacement plan.
    date = "date"
    # Special: column is free text; internally re-scanned for entities in a
    # second detection pass rather than replaced as a single structured value.
    free_text = "free_text"


class PiiConditioningColumns(NSSBaseModel):
    """Columns whose values condition synthetic-name generation for a set.

    Each field is the name of a dataframe column to condition on, or ``None``
    when no such column exists. Whether the LLM infers missing demographics is
    controlled by ``llm_enhancement`` on the enclosing ``ReplacePiiConfig``.
    """

    gender: str | None = None
    ethnic_background: str | None = None


class PiiColumnPlan(NSSBaseModel):
    """Replacement spec for one column (keyed by column name).

    Used for both associated (role) and unassociated columns. Set
    ``entity_type: free_text`` for columns handled by a second entity-detection
    pass rather than replaced as a single structured value.
    """

    entity_type: PiiEntity | None = None
    pattern: str | None = None  # dominant concrete format/template (e.g. %m/%d/%Y)
    dominant_pattern_coverage: float | None = Field(
        default=None,
        description="Percent of non-null values matching ``pattern`` (e.g. 99.6).",
    )


class AssociatedColumnSet(NSSBaseModel):
    """A set of columns tied to one identity/role (keyed by role name).

    Groups the columns belonging to a single synthetic identity (e.g. patient,
    doctor, emergency_contact) together with the columns used to condition their
    generated names.
    """

    columns_to_replace: dict[str, PiiColumnPlan] = Field(default_factory=dict)
    conditioning_columns: PiiConditioningColumns | None = None


class PiiReplacementPlan(Parameters):
    """Dataset-specific detection/replacement plan (column-oriented)."""

    group_key: str | None = None
    # role name -> its associated columns; arbitrary number of roles.
    associated_column_sets: dict[str, AssociatedColumnSet] = Field(default_factory=dict)
    unassociated_columns_to_replace: dict[str, PiiColumnPlan] = Field(default_factory=dict)


class LLMConfig(NSSBaseModel):
    """Shared LLM settings for discovery and free-text replacement (placeholder for MVP)."""

    model_provider: str | None = None
    max_workers: int = 64


class PiiDiscoveryConfig(NSSBaseModel):
    """Discovery-only settings; ignored when a plan is supplied."""

    replace_group_key: bool = True


class PiiReplacementSettings(NSSBaseModel):
    """Apply-time replacement basics."""

    locale: str = "en_US"
    seed: int | None = None


class PiiPersonBackend(StrEnum):
    pgm = "pgm"
    managed = "managed"
    faker = "faker"


class PiiPersonConfig(NSSBaseModel):
    """Synthetic-person generation settings."""

    backend: PiiPersonBackend = Field(
        default=PiiPersonBackend.managed,
        description="Persona sampler backend: managed assets, PGM, or Faker.",
    )
    sdg_pgms_src: str = Field(
        default="/root/sdg-pgms/src",
        description="Source tree for sdg-pgms when backend is PGM.",
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

    schema_version: int = 1
    llm_enhancement: bool = False  # MVP mode (False) vs. LLM mode (True)
    replacement_plan: PiiReplacementPlan | str = AUTO_DISCOVERY
    llm: LLMConfig = Field(default_factory=LLMConfig)
    discovery: PiiDiscoveryConfig = Field(default_factory=PiiDiscoveryConfig)
    replacement: PiiReplacementSettings = Field(default_factory=PiiReplacementSettings)
    person: PiiPersonConfig = Field(default_factory=PiiPersonConfig)

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
