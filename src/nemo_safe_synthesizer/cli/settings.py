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

    # Compose existing settings (automatically populated from env vars)
    # These are NOT loaded from env vars by CLISettings - they load their own
    # We use default_factory to create fresh instances that read env vars
    observability: NSSObservabilitySettings = Field(default_factory=NSSObservabilitySettings)
    wandb: WandbSettings = Field(default_factory=WandbSettings)

    # CLI-specific settings (paths)
    # Note: AliasChoices allows both the field name (for CLI kwargs) and env var name to work
    url: str | None = Field(default=None, description="Dataset URL, name, or path to CSV")
    config_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("config_path", "NSS_CONFIG"),
        description="Path to YAML config file",
    )
    artifact_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("artifact_path", "NSS_ARTIFACTS_PATH"),
        description="Base directory for all runs",
    )
    run_path: str | None = Field(
        default=None,
        description="Explicit path for this run's output directory",
    )
    output_file: str | None = Field(
        default=None,
        description="Path to output CSV file",
    )

    # Logging settings (can override observability defaults from CLI)
    log_format: Literal["json", "plain"] | None = Field(
        default=None,
        validation_alias=AliasChoices("log_format", "NSS_LOG_FORMAT"),
        description="Log format for console output",
    )
    log_color: bool | None = Field(
        default=None,
        description="Whether to colorize console output",
    )
    log_file: str | None = Field(
        default=None,
        validation_alias=AliasChoices("log_file", "NSS_LOG_FILE"),
        description="Path to log file",
    )
    verbose: int = Field(
        default=0,
        description="Verbosity level (0=INFO, 1=DEBUG, 2=DEBUG_DEPENDENCIES)",
    )

    # WandB settings (can override wandb defaults from CLI)
    wandb_mode: WandbMode | None = Field(
        default=None,
        description="WandB mode override",
    )
    wandb_project: str | None = Field(
        default=None,
        description="WandB project override",
    )

    # Synthesis parameter overrides (populated from --data__*, --training__*, etc.)
    synthesis_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Nested dict of SafeSynthesizerParameters overrides from CLI",
    )

    dataset_registry: str | None = Field(
        default=None,
        validation_alias=AliasChoices("dataset_registry", "NSS_DATASET_REGISTRY"),
        description="URL or path to a dataset registry YAML file",
    )

    @field_validator("wandb_mode", mode="before")
    @classmethod
    def validate_wandb_mode(cls, v: str | WandbMode | None) -> WandbMode | None:
        """Coerce string or None to ``WandbMode``, passing through enum values unchanged."""
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
