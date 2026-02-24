# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CLI utility functions, specifically the common_setup function."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nemo_safe_synthesizer.cli.settings import CLISettings
from nemo_safe_synthesizer.cli.utils import common_setup


@pytest.fixture
def registry_with_dataset(tmp_path: Path, dummy_csv: Path) -> Path:
    """Create a dataset registry YAML file that references a CSV."""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(f"""
datasets:
  - name: my-dataset
    url: {dummy_csv}
    overrides:
      training:
        batch_size: 16
        learning_rate: 0.01
      enable_replace_pii: false
""")
    return registry_path


@pytest.fixture
def registry_with_base_url(tmp_path: Path) -> Path:
    """Create a dataset registry YAML with base_url and a data file."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_file = data_dir / "test_data.csv"
    csv_file.write_text("x,y,z\na,b,c\nd,e,f\n")

    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(f"""
base_url: {data_dir}
datasets:
  - name: test-data
    url: test_data.csv
    overrides:
      generation:
        num_records: 500
""")
    return registry_path


@pytest.fixture
def patched_common_setup_dependencies(mock_workdir: MagicMock, mock_logger: MagicMock):
    """Patch dependencies needed to test common_setup.

    This fixture patches:
    - _create_workdir: returns mock workdir
    - initialize_wandb_run: mocked to avoid wandb dependency
    - _initialize_logging_for_cli_from_settings: returns mock logger

    Yields a dict with all mocks for assertions.
    """
    with (
        patch("nemo_safe_synthesizer.cli.utils._create_workdir") as mock_create_workdir,
        patch("nemo_safe_synthesizer.cli.utils.initialize_wandb_run") as mock_init_wandb,
        patch("nemo_safe_synthesizer.cli.utils._initialize_logging_for_cli_from_settings") as mock_init_logging,
    ):
        mock_create_workdir.return_value = mock_workdir
        mock_init_logging.return_value = mock_logger

        yield {
            "create_workdir": mock_create_workdir,
            "init_wandb": mock_init_wandb,
            "init_logging": mock_init_logging,
            "workdir": mock_workdir,
            "logger": mock_logger,
        }


class TestCommonSetupDatasetRegistry:
    """Tests for dataset registry logic in common_setup (steps 3, 4, 5)."""

    def test_load_dataset_without_registry(
        self,
        dummy_csv: Path,
        patched_common_setup_dependencies: dict,
    ):
        """Test step 3: An empty DatasetRegistry is created when not specified."""
        settings = CLISettings.from_cli_kwargs(
            url=str(dummy_csv),
            dataset_registry=None,
        )

        _, _, df, _ = common_setup(settings)

        # Verify the common_setup completed successfully
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        # Dataset should be loaded directly from the url
        assert list(df.columns) == ["col1", "col2"]
        assert len(df) == 2

    def test_load_other_url_with_registry(
        self,
        dummy_csv: Path,
        registry_with_base_url: Path,
        patched_common_setup_dependencies: dict,
    ):
        """Test step 3: DatasetRegistry is created from YAML file when specified."""
        settings = CLISettings.from_cli_kwargs(
            url=str(dummy_csv),
            dataset_registry=str(registry_with_base_url),
        )

        _, _, df, _ = common_setup(settings)

        # Verify the common_setup completed successfully
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["col1", "col2"]
        assert len(df) == 2

    def test_load_dataset_by_name_from_registry(
        self,
        registry_with_dataset: Path,
        patched_common_setup_dependencies: dict,
    ):
        """Test step 4: Dataset is loaded by name when it exists in registry."""
        settings = CLISettings.from_cli_kwargs(
            url="my-dataset",  # Name in registry, not a file path
            dataset_registry=str(registry_with_dataset),
        )

        _, _, df, _ = common_setup(settings)

        # Verify dataset was loaded from registry
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["col1", "col2"]
        assert len(df) == 2

    def test_load_dataset_with_base_url(
        self,
        registry_with_base_url: Path,
        patched_common_setup_dependencies: dict,
    ):
        """Test step 4: Dataset URL is resolved relative to base_url."""
        settings = CLISettings.from_cli_kwargs(
            url="test-data",  # Name in registry
            dataset_registry=str(registry_with_base_url),
        )

        _, _, df, _ = common_setup(settings)

        # Verify dataset was loaded with base_url resolution
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["x", "y", "z"]
        assert len(df) == 2

    def test_apply_dataset_overrides_to_config(
        self,
        registry_with_dataset: Path,
        patched_common_setup_dependencies: dict,
    ):
        """Test step 5: Dataset overrides from registry are applied to config."""
        settings = CLISettings.from_cli_kwargs(
            url="my-dataset",
            dataset_registry=str(registry_with_dataset),
        )

        _, config, _, _ = common_setup(settings)

        # Verify overrides from registry are applied to config
        assert config.training.batch_size == 16
        assert config.training.learning_rate == 0.01

    def test_cli_overrides_take_precedence(
        self,
        registry_with_dataset: Path,
        patched_common_setup_dependencies: dict,
    ):
        """Test step 5: CLI overrides take precedence over dataset overrides."""
        settings = CLISettings.from_cli_kwargs(
            url="my-dataset",
            dataset_registry=str(registry_with_dataset),
            synthesis_overrides={
                "training": {
                    "batch_size": 32,  # Should override registry's 16
                }
            },
        )

        _, config, _, _ = common_setup(settings)

        # CLI override should take precedence
        assert config.training.batch_size == 32
        # Dataset override should still be applied for learning_rate
        assert config.training.learning_rate == 0.01

    def test_merge_dataset_and_cli_overrides(
        self,
        registry_with_base_url: Path,
        patched_common_setup_dependencies: dict,
    ):
        """Test step 5: Dataset and CLI overrides are merged correctly."""
        settings = CLISettings.from_cli_kwargs(
            url="test-data",
            dataset_registry=str(registry_with_base_url),
            synthesis_overrides={
                "training": {
                    "batch_size": 8,
                },
                "data": {
                    "holdout": 0.1,
                },
            },
        )

        _, config, _, _ = common_setup(settings)

        # Dataset override from registry
        assert config.generation.num_records == 500
        # CLI overrides
        assert config.training.batch_size == 8
        assert config.data.holdout == 0.1

    def test_overrides_config_registry_and_cli(
        self,
        registry_with_dataset: Path,
        patched_common_setup_dependencies: dict,
        tmp_path: Path,
    ):
        """Test that dataset overrides from config file, registry and CLI are merged correctly."""
        # Precedence is (1) CLI, (2) registry, (3) config file

        # Config file with base values overriding SafeSynthesizerParameters defaults
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
training:
  num_input_records_to_sample: 150
  learning_rate: 0.05
  batch_size: 8
generation:
  use_structured_generation: true
""")

        # Overrides from registry for reference (see registry_with_dataset fixture)
        # training:
        #   batch_size: 16
        #   learning_rate: 0.01
        # enable_replace_pii: false

        # CLI overrides in synthesis_overrides
        settings = CLISettings.from_cli_kwargs(
            url="my-dataset",
            dataset_registry=str(registry_with_dataset),
            synthesis_overrides={
                "training": {
                    "batch_size": 32,
                },
                "generation": {
                    "num_records": 496,
                },
            },
            config_path=str(config_file),
        )

        _, config, _, _ = common_setup(settings)

        # Only given in config file
        assert config.training.num_input_records_to_sample == 150
        assert config.generation.use_structured_generation
        # Only given in registry
        assert not config.enable_replace_pii
        # Only given in CLI
        assert config.generation.num_records == 496
        # Present in config file and registry, registry takes precedence
        assert config.training.learning_rate == 0.01
        # Present in config file, registry, and CLI, CLI takes precedence
        assert config.training.batch_size == 32


class TestCommonSetupWithoutRegistry:
    """Tests for common_setup when no dataset registry is used."""

    def test_load_csv_directly_from_url(
        self,
        dummy_csv: Path,
        patched_common_setup_dependencies: dict,
    ):
        """Test that CSV is loaded directly when no registry is specified."""
        settings = CLISettings.from_cli_kwargs(
            url=str(dummy_csv),
        )

        _, _, df, _ = common_setup(settings)

        assert df is not None
        assert list(df.columns) == ["col1", "col2"]
        assert len(df) == 2

    def test_apply_cli_overrides_without_registry(
        self,
        dummy_csv: Path,
        patched_common_setup_dependencies: dict,
    ):
        """Test that CLI overrides are applied when no registry is used."""
        settings = CLISettings.from_cli_kwargs(
            url=str(dummy_csv),
            synthesis_overrides={
                "generation": {
                    "num_records": 100,
                    "temperature": 0.7,
                },
            },
        )

        _, config, _, _ = common_setup(settings)

        assert config.generation.num_records == 100
        assert config.generation.temperature == 0.7


class TestCommonSetupReturnValues:
    """Tests for common_setup return values."""

    def test_returns_correct_types(
        self,
        dummy_csv: Path,
        patched_common_setup_dependencies: dict,
    ):
        """Test that common_setup returns correct types."""
        from nemo_safe_synthesizer.config import SafeSynthesizerParameters

        settings = CLISettings.from_cli_kwargs(
            url=str(dummy_csv),
        )

        result = common_setup(settings)
        assert isinstance(result, tuple)
        assert len(result) == 4
        logger, config, df, workdir = result

        # Logger is mocked
        assert logger is patched_common_setup_dependencies["logger"]
        # Config should be SafeSynthesizerParameters
        assert isinstance(config, SafeSynthesizerParameters)
        # DataFrame loaded from CSV
        assert isinstance(df, pd.DataFrame)
        # Workdir is mocked
        assert workdir is patched_common_setup_dependencies["workdir"]
