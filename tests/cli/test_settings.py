# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for unified CLI settings."""

from __future__ import annotations

from nemo_safe_synthesizer.cli.settings import CLISettings
from nemo_safe_synthesizer.cli.wandb_setup import WandbMode


class TestCLISettings:
    """Tests for CLISettings unified settings model."""

    def test_default_values(self):
        """Test that CLISettings has sensible defaults."""
        settings = CLISettings()

        # CLI-specific defaults
        assert settings.data_source is None
        assert settings.config_path is None
        assert settings.artifact_path is None
        assert settings.run_path is None
        assert settings.output_file is None
        assert settings.verbose == 0
        assert settings.synthesis_overrides == {}

    def test_env_var_loading_artifact_path(self, monkeypatch):
        """Test that artifact_path loads from NSS_ARTIFACTS_PATH env var."""
        monkeypatch.setenv("NSS_ARTIFACTS_PATH", "/tmp/test-artifacts")

        settings = CLISettings()
        assert settings.artifact_path == "/tmp/test-artifacts"

    def test_env_var_loading_config(self, monkeypatch):
        """Test that config_path loads from NSS_CONFIG env var."""
        monkeypatch.setenv("NSS_CONFIG", "/tmp/config.yaml")

        settings = CLISettings()
        assert settings.config_path == "/tmp/config.yaml"

    def test_env_var_loading_log_format(self, monkeypatch):
        """Test that log_format loads from NSS_LOG_FORMAT env var."""
        monkeypatch.setenv("NSS_LOG_FORMAT", "json")

        settings = CLISettings()
        assert settings.log_format == "json"

    def test_env_var_loading_log_file(self, monkeypatch):
        """Test that log_file loads from NSS_LOG_FILE env var."""
        monkeypatch.setenv("NSS_LOG_FILE", "/tmp/nss.log")

        settings = CLISettings()
        assert settings.log_file == "/tmp/nss.log"

    def test_from_cli_kwargs_filters_none(self):
        """Test that from_cli_kwargs filters out None values and passes through non-None."""
        settings = CLISettings.from_cli_kwargs(
            data_source="data.csv",
            config_path=None,  # Should be filtered
            run_path="/tmp/run1",  # Should be passed through (no env var alias)
            output_file=None,  # Should be filtered
        )

        assert settings.data_source == "data.csv"
        assert settings.run_path == "/tmp/run1"
        # None values should be filtered, so these come from defaults
        assert settings.config_path is None
        assert settings.output_file is None

    def test_from_cli_kwargs_all_values(self):
        """Test from_cli_kwargs with values that don't have env var aliases."""
        settings = CLISettings.from_cli_kwargs(
            data_source="data.csv",
            run_path="/tmp/run1",  # No env var alias
            output_file="output.csv",  # No env var alias
            log_color=False,  # No env var alias
            verbose=2,
            wandb_mode="online",
            wandb_project="test-project",
            synthesis_overrides={"training": {"epochs": 5}},
            dataset_registry="path/to/registry.yaml",
        )

        assert settings.data_source == "data.csv"
        assert settings.run_path == "/tmp/run1"
        assert settings.output_file == "output.csv"
        assert settings.log_color is False
        assert settings.verbose == 2
        assert settings.wandb_mode == WandbMode.ONLINE
        assert settings.wandb_project == "test-project"
        assert settings.synthesis_overrides == {"training": {"epochs": 5}}
        assert settings.dataset_registry == "path/to/registry.yaml"

    def test_env_vars_provide_defaults_for_unset_cli_args(self, monkeypatch):
        """Test that env vars provide defaults when CLI args are not provided."""
        # Set env vars
        monkeypatch.setenv("NSS_ARTIFACTS_PATH", "/env/artifacts")
        monkeypatch.setenv("NSS_LOG_FORMAT", "json")
        monkeypatch.setenv("NSS_DATASET_REGISTRY", "path/to/registry.yaml")

        # Create settings without providing those CLI kwargs
        settings = CLISettings.from_cli_kwargs(
            data_source="data.csv",  # Only provide data_source
        )

        # Env vars provide values for fields with AliasChoices
        assert settings.artifact_path == "/env/artifacts"
        assert settings.log_format == "json"
        assert settings.dataset_registry == "path/to/registry.yaml"
        # data_source is set from CLI
        assert settings.data_source == "data.csv"

    def test_composed_settings_accessible(self):
        """Test that composed settings are accessible."""
        settings = CLISettings()

        # Check that composed settings exist and are the right type
        assert hasattr(settings, "observability")
        assert hasattr(settings, "wandb")

        # Check that they're instances of the right classes
        from nemo_safe_synthesizer.cli.wandb_setup import WandbSettings
        from nemo_safe_synthesizer.observability import NSSObservabilitySettings

        assert isinstance(settings.observability, NSSObservabilitySettings)
        assert isinstance(settings.wandb, WandbSettings)

    def test_effective_artifact_path_default(self):
        """Test effective_artifact_path returns default when not set."""
        from nemo_safe_synthesizer.defaults import DEFAULT_ARTIFACTS_PATH

        settings = CLISettings()
        assert settings.effective_artifact_path == DEFAULT_ARTIFACTS_PATH

    def test_effective_artifact_path_custom(self, monkeypatch):
        """Test effective_artifact_path returns custom path when set via env var."""
        from pathlib import Path

        # Set via env var (the supported way to configure artifact_path)
        monkeypatch.setenv("NSS_ARTIFACTS_PATH", "/custom/path")
        settings = CLISettings()
        assert settings.effective_artifact_path == Path("/custom/path")

    def test_effective_log_format_from_cli(self):
        """Test effective_log_format uses CLI value when set."""
        settings = CLISettings.from_cli_kwargs(log_format="json")
        assert settings.effective_log_format == "json"

    def test_effective_log_format_from_observability(self):
        """Test effective_log_format falls back to observability settings."""
        settings = CLISettings()
        # Should fall back to observability.nss_log_format
        assert settings.effective_log_format in ["json", "plain"]

    def test_effective_wandb_mode_from_cli(self):
        """Test effective_wandb_mode uses CLI value when set."""
        settings = CLISettings.from_cli_kwargs(wandb_mode="online")
        assert settings.effective_wandb_mode == WandbMode.ONLINE

    def test_effective_wandb_mode_from_settings(self):
        """Test effective_wandb_mode falls back to wandb settings."""
        settings = CLISettings()
        # Should fall back to wandb.wandb_mode (default is DISABLED)
        assert settings.effective_wandb_mode == WandbMode.DISABLED

    def test_effective_wandb_project_from_cli(self):
        """Test effective_wandb_project uses CLI value when set."""
        settings = CLISettings.from_cli_kwargs(wandb_project="my-project")
        assert settings.effective_wandb_project == "my-project"

    def test_wandb_mode_string_conversion(self):
        """Test that wandb_mode accepts string values."""
        settings = CLISettings.from_cli_kwargs(wandb_mode="offline")
        assert settings.wandb_mode == WandbMode.OFFLINE

        settings = CLISettings.from_cli_kwargs(wandb_mode="online")
        assert settings.wandb_mode == WandbMode.ONLINE

        settings = CLISettings.from_cli_kwargs(wandb_mode="disabled")
        assert settings.wandb_mode == WandbMode.DISABLED

    def test_verbose_string_conversion(self):
        """Test that verbose accepts string values (for env var loading)."""
        settings = CLISettings(verbose="2")  # ty: ignore[invalid-argument-type]
        assert settings.verbose == 2

    def test_synthesis_overrides_default(self):
        """Test that synthesis_overrides defaults to empty dict."""
        settings = CLISettings()
        assert settings.synthesis_overrides == {}
        assert isinstance(settings.synthesis_overrides, dict)

    def test_synthesis_overrides_populated(self):
        """Test that synthesis_overrides can be populated."""
        overrides = {
            "training": {"epochs": 10, "batch_size": 32},
            "generation": {"num_samples": 1000},
        }
        settings = CLISettings.from_cli_kwargs(synthesis_overrides=overrides)
        assert settings.synthesis_overrides == overrides

    def test_dataset_registry_default(self):
        """Test that dataset_registry defaults to None."""
        settings = CLISettings()
        assert settings.dataset_registry is None

    def test_dataset_registry_from_env(self, monkeypatch):
        """Test that dataset_registry returns custom path when set via env var."""
        monkeypatch.setenv("NSS_DATASET_REGISTRY", "/custom/path/registry.yaml")
        settings = CLISettings()
        assert settings.dataset_registry == "/custom/path/registry.yaml"

    def test_dataset_registry_from_cli(self, monkeypatch):
        """Test that dataset_registry via CLI takes precedence over env var"""
        monkeypatch.setenv("NSS_DATASET_REGISTRY", "/other/registry.yaml")
        settings = CLISettings.from_cli_kwargs(dataset_registry="path/to/registry.yaml")
        assert settings.dataset_registry == "path/to/registry.yaml"


class TestCLISettingsIntegration:
    """Integration tests for CLISettings with env vars."""

    def test_wandb_env_vars_propagate(self, monkeypatch):
        """Test that WANDB env vars are accessible via composed settings."""
        monkeypatch.setenv("WANDB_MODE", "offline")
        monkeypatch.setenv("WANDB_PROJECT", "test-project")

        settings = CLISettings()
        assert settings.wandb.wandb_mode == WandbMode.OFFLINE
        assert settings.wandb.wandb_project == "test-project"

    def test_extra_env_vars_ignored(self, monkeypatch):
        """Test that unknown env vars are ignored (extra='ignore')."""
        monkeypatch.setenv("NSS_UNKNOWN_SETTING", "value")

        # Should not raise an error
        settings = CLISettings()
        assert not hasattr(settings, "unknown_setting")
