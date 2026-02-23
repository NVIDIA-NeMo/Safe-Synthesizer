# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CLI run command and its options."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

import nemo_safe_synthesizer.sdk.library_builder  # noqa: F401 - ensure submodule is loaded for mock.patch
from nemo_safe_synthesizer.cli.run import run
from nemo_safe_synthesizer.cli.settings import CLISettings

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock SafeSynthesizerParameters config."""
    config = MagicMock()
    config.model_dump.return_value = {}
    return config


@pytest.fixture
def mock_dataframe() -> MagicMock:
    """Create a mock DataFrame."""
    return MagicMock()


@pytest.fixture
def mock_safe_synthesizer() -> MagicMock:
    """Create a mock SafeSynthesizer with all necessary method stubs."""
    ss = MagicMock()
    ss.with_data_source.return_value = ss
    ss.process_data.return_value = ss
    ss.train.return_value = ss
    ss.generate.return_value = ss
    ss.evaluate.return_value = ss
    ss.load_from_save_path.return_value = ss
    ss.run.return_value = ss
    ss.save_results.return_value = ss  # Return self for method chaining
    ss.generator.teardown.return_value = None
    ss.results.summary.log_summary = MagicMock()
    ss.results.summary.timing.log_timing = MagicMock()
    ss.results.summary.log_wandb = MagicMock()
    return ss


@pytest.fixture
def mock_common_setup_return(
    mock_logger: MagicMock,
    mock_config: MagicMock,
    mock_dataframe: MagicMock,
    mock_workdir: MagicMock,
) -> tuple:
    """Create the return value tuple for common_setup mock."""
    return (mock_logger, mock_config, mock_dataframe, mock_workdir)


@pytest.fixture
def patched_run_dependencies(mock_common_setup_return: tuple, mock_safe_synthesizer: MagicMock):
    """Patch all dependencies needed to test the run command.

    This fixture patches:
    - common_setup: returns mock logger, config, df, workdir
    - traced_user: mock context manager
    - SafeSynthesizer: returns mock synthesizer (with save_results as a method)

    Note: load_dataset and merge_overrides are called inside common_setup (in utils.py),
    not directly in run.py, so they don't need to be patched here since common_setup
    is already mocked.

    Yields a dict with all mocks for assertions.
    """
    with (
        patch("nemo_safe_synthesizer.cli.run.common_setup") as mock_common_setup,
        patch("nemo_safe_synthesizer.cli.run.traced_user") as mock_traced_user,
        patch(
            "nemo_safe_synthesizer.sdk.library_builder.SafeSynthesizer",
            return_value=mock_safe_synthesizer,
        ),
    ):
        mock_common_setup.return_value = mock_common_setup_return

        # Mock the traced_user context manager
        mock_traced_user.return_value.__enter__ = MagicMock()
        mock_traced_user.return_value.__exit__ = MagicMock(return_value=False)

        yield {
            "common_setup": mock_common_setup,
            "traced_user": mock_traced_user,
            "safe_synthesizer": mock_safe_synthesizer,
        }


# =============================================================================
# Tests
# =============================================================================


class TestRunCommandOptions:
    """Tests for run command CLI options."""

    def test_generate_subcommand_has_output_file_option(self, cli_runner: CliRunner):
        """Verify --output-file option appears in generate subcommand help."""
        result = cli_runner.invoke(run, ["generate", "--help"])

        assert result.exit_code == 0
        assert "--output-file" in result.output


class TestOutputFileOverride:
    """Tests for --output-file override behavior."""

    def test_run_uses_custom_output_file(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        tmp_path: Path,
        fixture_session_cache_dir: Path,
        patched_run_dependencies: dict,
    ):
        """Verify that --output-file overrides default workdir output."""
        custom_output = tmp_path / "custom_output.csv"

        result = cli_runner.invoke(
            run,
            [
                "--url",
                str(dummy_csv),
                "--output-file",
                str(custom_output),
                "--artifact-path",
                str(fixture_session_cache_dir),
            ],
            catch_exceptions=False,
        )

        # Verify save_results was called with the custom output file
        assert result.exit_code == 0
        mock_ss = patched_run_dependencies["safe_synthesizer"]
        mock_ss.save_results.assert_called_once()
        actual_output_path = mock_ss.save_results.call_args.kwargs.get("output_file")
        assert str(actual_output_path) == str(custom_output)

    def test_run_uses_workdir_output_when_no_override(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        fixture_session_cache_dir: Path,
        mock_workdir: MagicMock,
        patched_run_dependencies: dict,
    ):
        """Verify that workdir.output_file is used when --output-file is not provided."""
        result = cli_runner.invoke(
            run,
            [
                "--url",
                str(dummy_csv),
                "--artifact-path",
                str(fixture_session_cache_dir),
            ],
            catch_exceptions=False,
        )

        # Verify save_results was called with the workdir's default output file
        assert result.exit_code == 0
        mock_ss = patched_run_dependencies["safe_synthesizer"]
        mock_ss.save_results.assert_called_once()
        actual_output_path = mock_ss.save_results.call_args.kwargs.get("output_file")
        assert str(actual_output_path) == str(mock_workdir.output_file)


class TestPathOptions:
    """Tests for --artifacts-path and --run-path options."""

    def test_run_help_shows_artifact_path_option(self, cli_runner: CliRunner):
        """Verify --artifact-path option appears in run command help."""
        result = cli_runner.invoke(run, ["--help"])

        assert result.exit_code == 0
        assert "--artifact-path" in result.output
        assert "Base directory for all runs" in result.output

    def test_run_help_shows_run_path_option(self, cli_runner: CliRunner):
        """Verify --run-path option appears in run command help."""
        result = cli_runner.invoke(run, ["--help"])

        assert result.exit_code == 0
        assert "--run-path" in result.output
        assert "Explicit path for this run" in result.output

    def test_run_with_artifact_path_only(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        tmp_path: Path,
        patched_run_dependencies: dict,
    ):
        """Verify run works with only --artifact-path specified."""
        artifacts_dir = tmp_path / "my-artifacts"

        result = cli_runner.invoke(
            run,
            [
                "--url",
                str(dummy_csv),
                "--artifact-path",
                str(artifacts_dir),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        # common_setup should have been called with settings containing artifact_path
        mock_common_setup = patched_run_dependencies["common_setup"]
        mock_common_setup.assert_called_once()
        call_kwargs = mock_common_setup.call_args.kwargs
        settings: CLISettings = call_kwargs["settings"]
        assert settings.artifact_path == str(artifacts_dir)
        assert settings.run_path is None

    def test_run_with_run_path_only(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        tmp_path: Path,
        patched_run_dependencies: dict,
    ):
        """Verify run works with only --run-path specified."""
        run_dir = tmp_path / "my-explicit-run"

        result = cli_runner.invoke(
            run,
            [
                "--url",
                str(dummy_csv),
                "--run-path",
                str(run_dir),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        # common_setup should have been called with settings containing run_path
        mock_common_setup = patched_run_dependencies["common_setup"]
        mock_common_setup.assert_called_once()
        call_kwargs = mock_common_setup.call_args.kwargs
        settings: CLISettings = call_kwargs["settings"]
        assert settings.artifact_path is None
        assert settings.run_path == str(run_dir)

    def test_run_with_both_paths_uses_run_path(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        tmp_path: Path,
        patched_run_dependencies: dict,
    ):
        """Verify that --run-path takes precedence when both options are specified."""
        artifacts_dir = tmp_path / "artifacts"
        run_dir = tmp_path / "explicit-run"

        result = cli_runner.invoke(
            run,
            [
                "--url",
                str(dummy_csv),
                "--artifact-path",
                str(artifacts_dir),
                "--run-path",
                str(run_dir),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        # common_setup should have been called with both, but _create_workdir handles precedence
        mock_common_setup = patched_run_dependencies["common_setup"]
        mock_common_setup.assert_called_once()
        call_kwargs = mock_common_setup.call_args.kwargs
        settings: CLISettings = call_kwargs["settings"]
        assert settings.artifact_path == str(artifacts_dir)
        assert settings.run_path == str(run_dir)

    def test_run_with_dataset_registry_calls_common_setup(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        patched_run_dependencies: dict,
    ):
        """Verify run with --dataset-registry calls common_setup correctly."""
        # common_setup() is mocked, so no actual file is needed, only
        # checking that common_setup is called with expected argument
        result = cli_runner.invoke(
            run,
            [
                "--url",
                str(dummy_csv),
                "--dataset-registry",
                "./registry.yaml",
            ],
        )

        assert result.exit_code == 0
        mock_common_setup = patched_run_dependencies["common_setup"]
        mock_common_setup.assert_called_once()
        call_kwargs = mock_common_setup.call_args.kwargs
        settings: CLISettings = call_kwargs["settings"]
        assert settings.dataset_registry == "./registry.yaml"


class TestRunTrainOptions:
    """Tests for run train command options."""

    def test_train_help_shows_options(self, cli_runner: CliRunner):
        """Verify train subcommand help shows expected options."""
        result = cli_runner.invoke(run, ["train", "--help"])

        assert result.exit_code == 0
        assert "--url" in result.output
        assert "--config" in result.output
        assert "--run-path" in result.output

    def test_train_with_run_path_calls_common_setup(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        tmp_path: Path,
        patched_run_dependencies: dict,
    ):
        """Verify train with --run-path calls common_setup correctly."""
        run_dir = tmp_path / "new-training-run"

        result = cli_runner.invoke(
            run,
            [
                "train",
                "--url",
                str(dummy_csv),
                "--run-path",
                str(run_dir),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        mock_common_setup = patched_run_dependencies["common_setup"]
        mock_common_setup.assert_called_once()
        # Check that phase="train" is passed and settings has run_path
        call_kwargs = mock_common_setup.call_args.kwargs
        assert call_kwargs.get("phase") == "train"
        settings: CLISettings = call_kwargs["settings"]
        assert settings.run_path == str(run_dir)

    def test_train_does_not_call_load_from_save_path(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        tmp_path: Path,
        patched_run_dependencies: dict,
    ):
        """Verify train command does NOT call load_from_save_path (fresh training)."""
        run_dir = tmp_path / "fresh-training-run"
        mock_ss = patched_run_dependencies["safe_synthesizer"]

        result = cli_runner.invoke(
            run,
            [
                "train",
                "--url",
                str(dummy_csv),
                "--run-path",
                str(run_dir),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        # Verify load_from_save_path was NOT called - fresh training doesn't resume
        mock_ss.load_from_save_path.assert_not_called()

    def test_train_with_dataset_registry_calls_common_setup(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        patched_run_dependencies: dict,
    ):
        """Verify train with --dataset-registry calls common_setup correctly."""
        # common_setup() is mocked, so no actual file is needed, only
        # checking that common_setup is called with expected argument
        result = cli_runner.invoke(
            run,
            [
                "train",
                "--url",
                str(dummy_csv),
                "--dataset-registry",
                "./registry.yaml",
            ],
        )

        assert result.exit_code == 0
        mock_common_setup = patched_run_dependencies["common_setup"]
        mock_common_setup.assert_called_once()
        call_kwargs = mock_common_setup.call_args.kwargs
        settings: CLISettings = call_kwargs["settings"]
        assert settings.dataset_registry == "./registry.yaml"


class TestRunGenerateOptions:
    """Tests for run generate command options."""

    def test_generate_help_shows_auto_discover_flag(self, cli_runner: CliRunner):
        """Verify --auto-discover-adapter flag appears in generate help."""
        result = cli_runner.invoke(run, ["generate", "--help"])

        assert result.exit_code == 0
        assert "--auto-discover-adapter" in result.output
        assert "Automatically find the latest trained adapter" in result.output

    def test_generate_help_shows_run_path_option(self, cli_runner: CliRunner):
        """Verify --run-path option appears in generate help."""
        result = cli_runner.invoke(run, ["generate", "--help"])

        assert result.exit_code == 0
        assert "--run-path" in result.output

    def test_generate_without_run_path_or_auto_discover_errors(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
    ):
        """Verify generate errors when neither --run-path nor --auto-discover-adapter is provided."""
        result = cli_runner.invoke(
            run,
            [
                "generate",
                "--url",
                str(dummy_csv),
            ],
        )

        assert result.exit_code != 0
        assert "--run-path is required" in result.output
        assert "--auto-discover-adapter" in result.output

    def test_generate_with_run_path_calls_common_setup(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        tmp_path: Path,
        patched_run_dependencies: dict,
    ):
        """Verify generate with --run-path calls common_setup correctly."""
        run_dir = tmp_path / "trained-run"

        result = cli_runner.invoke(
            run,
            [
                "generate",
                "--url",
                str(dummy_csv),
                "--run-path",
                str(run_dir),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        mock_common_setup = patched_run_dependencies["common_setup"]
        mock_common_setup.assert_called_once()
        # Check that resume=True and auto_discover_adapter=False, with settings containing run_path
        call_kwargs = mock_common_setup.call_args.kwargs
        assert call_kwargs.get("resume") is True
        assert call_kwargs.get("auto_discover_adapter") is False
        settings: CLISettings = call_kwargs["settings"]
        assert settings.run_path == str(run_dir)

    def test_generate_with_auto_discover_calls_common_setup(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        tmp_path: Path,
        patched_run_dependencies: dict,
    ):
        """Verify generate with --auto-discover-adapter calls common_setup correctly."""
        artifacts_dir = tmp_path / "artifacts"

        result = cli_runner.invoke(
            run,
            [
                "generate",
                "--url",
                str(dummy_csv),
                "--artifact-path",
                str(artifacts_dir),
                "--auto-discover-adapter",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        mock_common_setup = patched_run_dependencies["common_setup"]
        mock_common_setup.assert_called_once()
        # Check that resume=True and auto_discover_adapter=True, with settings containing artifact_path
        call_kwargs = mock_common_setup.call_args.kwargs
        assert call_kwargs.get("resume") is True
        assert call_kwargs.get("auto_discover_adapter") is True
        settings: CLISettings = call_kwargs["settings"]
        assert settings.artifact_path == str(artifacts_dir)

    def test_generate_with_wandb_resume_job_id_calls_common_setup(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        tmp_path: Path,
        patched_run_dependencies: dict,
    ):
        """Verify generate with --wandb-resume-job-id passes it to common_setup."""
        run_dir = tmp_path / "trained-run"
        wandb_run_id = "abc123xyz"

        result = cli_runner.invoke(
            run,
            [
                "generate",
                "--url",
                str(dummy_csv),
                "--run-path",
                str(run_dir),
                "--wandb-resume-job-id",
                wandb_run_id,
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        mock_common_setup = patched_run_dependencies["common_setup"]
        mock_common_setup.assert_called_once()
        call_kwargs = mock_common_setup.call_args.kwargs
        assert call_kwargs.get("wandb_resume_job_id") == wandb_run_id

    def test_generate_with_wandb_resume_job_id_file_calls_common_setup(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        tmp_path: Path,
        patched_run_dependencies: dict,
    ):
        """Verify generate with --wandb-resume-job-id pointing to a file passes it to common_setup."""
        run_dir = tmp_path / "trained-run"
        wandb_id_file = tmp_path / "wandb_run_id.txt"
        wandb_id_file.write_text("file_based_run_id_456")

        result = cli_runner.invoke(
            run,
            [
                "generate",
                "--url",
                str(dummy_csv),
                "--run-path",
                str(run_dir),
                "--wandb-resume-job-id",
                str(wandb_id_file),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        mock_common_setup = patched_run_dependencies["common_setup"]
        mock_common_setup.assert_called_once()
        call_kwargs = mock_common_setup.call_args.kwargs
        # The file path is passed to common_setup; resolution happens in wandb_setup
        assert call_kwargs.get("wandb_resume_job_id") == str(wandb_id_file)

    def test_generate_with_dataset_registry_calls_common_setup(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        patched_run_dependencies: dict,
    ):
        """Verify train with --dataset-registry calls common_setup correctly."""
        # common_setup() is mocked, so no actual file is needed, only
        # checking that common_setup is called with expected argument
        result = cli_runner.invoke(
            run,
            [
                "generate",
                "--url",
                str(dummy_csv),
                "--dataset-registry",
                "./registry.yaml",
            ],
        )

        assert result.exit_code == 0
        mock_common_setup = patched_run_dependencies["common_setup"]
        mock_common_setup.assert_called_once()
        call_kwargs = mock_common_setup.call_args.kwargs
        settings: CLISettings = call_kwargs["settings"]
        assert settings.dataset_registry == "./registry.yaml"
