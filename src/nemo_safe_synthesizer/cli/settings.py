# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified CLI settings for Safe Synthesizer.

This module provides a unified settings model that composes all sub-settings
(observability, wandb, etc.) into a single pydantic-settings class.

The CLISettings class:

- Automatically loads from environment variables
- Composes existing settings classes as nested fields
- Provides a single source of truth for all CLI configuration
- Can be instantiated from Click kwargs with CLI precedence

Usage:
    # From environment variables only
    settings = CLISettings()

    # From Click kwargs (CLI takes precedence over env vars)
    settings = CLISettings.from_cli_kwargs(url="data.csv", artifact_path="/tmp")

    # Access composed settings
    log_format = settings.observability.nss_log_format
    wandb_mode = settings.wandb.wandb_mode
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..defaults import DEFAULT_ARTIFACTS_PATH
from ..observability import NSSObservabilitySettings
from .wandb_setup import WandbMode, WandbSettings

__all__ = ["CLISettings"]


class CLISettings(BaseSettings):
    """Unified CLI settings composing all sub-settings.

    Consolidates environment variables (automatic via pydantic-settings),
    CLI arguments (passed via `from_cli_kwargs`), and composed sub-settings
    (observability, wandb).
    """

    model_config = SettingsConfigDict(
        # Note: We don't use env_prefix because:
        # 1. Composed settings (observability, wandb) load their own env vars
        # 2. Individual fields use explicit AliasChoices for env var mapping
        # 3. Using env_prefix="NSS_" would conflict with composed settings
        env_nested_delimiter="__",
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    observability: NSSObservabilitySettings = Field(
        default_factory=NSSObservabilitySettings, description="Observability sub-settings (log level, format, color)."
    )
    """Observability sub-settings (log level, format, color).

    Loaded from its own environment variables; not populated by ``CLISettings``.
    """

    wandb: WandbSettings = Field(default_factory=WandbSettings, description="WandB settings (mode, project, phase).")
    """WandB settings (mode, project, phase).

    Loaded from its own environment variables; not populated by ``CLISettings``.
    """

    url: str | None = Field(default=None, description="Dataset URL, name, or path to CSV")
    """Dataset URL, name, or path to CSV."""

    config_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("config_path", "NSS_CONFIG"),
        description="Path to YAML config file",
    )
    """Path to YAML config file (env variable: ``NSS_CONFIG``)."""

    artifact_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("artifact_path", "NSS_ARTIFACTS_PATH"),
        description="Base directory for all runs",
    )
    """Base directory for all runs (env variable: ``NSS_ARTIFACTS_PATH``)."""

    run_path: str | None = Field(
        default=None,
        description="Explicit path for this run's output directory",
    )
    """Explicit path for this run's output directory.

    When specified, overrides ``artifact_path`` and skips the
    ``<project>/<timestamp>`` directory layout.
    """

    output_file: str | None = Field(
        default=None,
        description="Path to output CSV file",
    )
    """Path to output CSV file, overriding the default workdir location."""

    log_format: Literal["json", "plain"] | None = Field(
        default=None,
        validation_alias=AliasChoices("log_format", "NSS_LOG_FORMAT"),
        description="Log format for console output",
    )
    """Log format for console output (env variable: ``NSS_LOG_FORMAT``).

    File logging is always JSON regardless of this setting.
    """

    log_color: bool | None = Field(
        default=None,
        description="Whether to colorize console output",
    )
    """Whether to colorize console output."""

    log_file: str | None = Field(
        default=None,
        validation_alias=AliasChoices("log_file", "NSS_LOG_FILE"),
        description="Path to log file",
    )
    """Path to log file (env variable: ``NSS_LOG_FILE``)."""

    verbose: int | str = Field(
        default=0,
        description="Verbosity level (0=INFO, 1=DEBUG, 2=DEBUG_DEPENDENCIES)",
    )
    """Verbosity level (0=INFO, 1=DEBUG, 2=DEBUG_DEPENDENCIES)."""

    wandb_mode: WandbMode | None = Field(
        default=None,
        description="WandB mode override",
    )
    """WandB mode override (online, offline, or disabled)."""

    wandb_project: str | None = Field(
        default=None,
        description="WandB project override",
    )
    """WandB project name override."""

    synthesis_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Nested dict of SafeSynthesizerParameters overrides from CLI",
    )
    """Nested dict of ``SafeSynthesizerParameters`` overrides from CLI.

    Populated from ``--data__*``, ``--training__*``, etc. options via
    ``parse_overrides``.
    """

    dataset_registry: str | None = Field(
        default=None,
        validation_alias=AliasChoices("dataset_registry", "NSS_DATASET_REGISTRY"),
        description="URL or path to a dataset registry YAML file",
    )
    """URL or path to a dataset registry YAML file (env: ``NSS_DATASET_REGISTRY``)."""

    @field_validator("wandb_mode", mode="before")
    @classmethod
    def validate_wandb_mode(cls, v: str | WandbMode | None) -> WandbMode | None:
        """Coerce string or None to ``WandbMode`` enum, passing through enum values unchanged."""
        if v is None:
            return None
        if isinstance(v, WandbMode):
            return v
        return WandbMode(v)

    @field_validator("verbose", mode="before")
    @classmethod
    def validate_verbose(cls, v: int | str | None) -> int:
        """Coerce string or None to int, defaulting to 0."""
        if v is None:
            return 0
        if isinstance(v, str):
            return int(v)
        return v

    @classmethod
    def from_cli_kwargs(cls, **kwargs: Any) -> CLISettings:
        """Create settings from Click kwargs, filtering None values.

        CLI arguments take precedence over environment variables.
        None values are filtered out so env vars can fill in.

        Args:
            **kwargs: Keyword arguments from Click command

        Returns:
            CLISettings instance with CLI values merged over env vars
        """
        # Filter out None values so env vars can provide defaults
        filtered = {k: v for k, v in kwargs.items() if v is not None}
        return cls(**filtered)

    @property
    def effective_artifact_path(self) -> Path:
        """Effective artifact path, falling back to ``DEFAULT_ARTIFACTS_PATH``."""
        if self.artifact_path:
            return Path(self.artifact_path)
        return DEFAULT_ARTIFACTS_PATH

    @property
    def effective_log_format(self) -> Literal["json", "plain"]:
        """Effective log format, falling back to observability settings."""
        if self.log_format is not None:
            return self.log_format
        return self.observability.nss_log_format or "plain"

    @property
    def effective_log_color(self) -> bool:
        """Effective log color setting, falling back to observability settings."""
        if self.log_color is not None:
            return self.log_color
        return self.observability.nss_log_color

    @property
    def effective_wandb_mode(self) -> WandbMode:
        """Effective wandb mode, falling back to wandb settings."""
        if self.wandb_mode is not None:
            return self.wandb_mode
        return self.wandb.wandb_mode

    @property
    def effective_wandb_project(self) -> str | None:
        """Effective wandb project, falling back to wandb settings."""
        if self.wandb_project is not None:
            return self.wandb_project
        return self.wandb.wandb_project
