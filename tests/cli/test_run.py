# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CLI run command and its options."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

import nemo_safe_synthesizer.observability as obs
import nemo_safe_synthesizer.sdk.library_builder  # noqa: F401 - ensure submodule is loaded for mock.patch
from nemo_safe_synthesizer.cli.run import run
from nemo_safe_synthesizer.cli.settings import CLISettings
from nemo_safe_synthesizer.cli.utils import merge_overrides
from nemo_safe_synthesizer.telemetry import DeploymentTypeEnum, TaskStatusEnum
from nemo_safe_synthesizer.tooling import PreflightRenderContext

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
    config.emit_telemetry = True
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
        ) as mock_safe_synthesizer_cls,
        patch("nemo_safe_synthesizer.sdk.library_builder._emit_nss_telemetry") as mock_emit_telemetry,
    ):

        def fake_common_setup(**kwargs):
            settings = kwargs["settings"]
            mock_common_setup_return[1].emit_telemetry = settings.synthesis_overrides.get("emit_telemetry", True)
            return mock_common_setup_return

        mock_common_setup.side_effect = fake_common_setup

        # Mock the traced_user context manager
        mock_traced_user.return_value.__enter__ = MagicMock()
        mock_traced_user.return_value.__exit__ = MagicMock(return_value=False)

        yield {
            "common_setup": mock_common_setup,
            "traced_user": mock_traced_user,
            "safe_synthesizer_cls": mock_safe_synthesizer_cls,
            "safe_synthesizer": mock_safe_synthesizer,
            "emit_telemetry": mock_emit_telemetry,
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

    def test_run_help_shows_emit_telemetry_config_option(self, cli_runner: CliRunner):
        """Verify the generated telemetry config option appears in run command help."""
        result = cli_runner.invoke(run, ["--help"])

        assert result.exit_code == 0
        assert "--emit_telemetry" in result.output

    def test_run_help_explains_wandb_report_opt_out(self, cli_runner: CliRunner):
        """The W&B report opt-out describes the output that remains enabled."""
        result = cli_runner.invoke(run, ["--help"])

        assert result.exit_code == 0
        assert "--no-wandb-upload-evaluation-report" in result.output
        assert "summary metrics and the evaluation scorecard" in " ".join(result.output.split())

    @pytest.mark.parametrize(
        ("upload_args", "expected_upload"),
        [
            ([], True),
            (["--no-wandb-upload-evaluation-report"], False),
        ],
    )
    def test_run_evaluation_report_upload_default_and_opt_out(
        self,
        upload_args: list[str],
        expected_upload: bool,
        cli_runner: CliRunner,
        dummy_csv: Path,
        fixture_session_cache_dir: Path,
        mock_workdir: MagicMock,
        patched_run_dependencies: dict,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Evaluation report publishing defaults on and supports explicit opt-out."""
        monkeypatch.delenv("NSS_WANDB_UPLOAD_EVALUATION_REPORT", raising=False)

        with patch("nemo_safe_synthesizer.cli.run.publish_evaluation_report") as mock_publish:
            result = cli_runner.invoke(
                run,
                [
                    "--data-source",
                    str(dummy_csv),
                    "--artifact-path",
                    str(fixture_session_cache_dir),
                    *upload_args,
                ],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        mock_publish.assert_called_once_with(
            mock_workdir,
            patched_run_dependencies["safe_synthesizer"].results.summary,
            expected_upload,
        )

    def test_run_defaults_to_emit_telemetry(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        fixture_session_cache_dir: Path,
        patched_run_dependencies: dict,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Telemetry is enabled by default."""
        monkeypatch.delenv("NEMO_DEPLOYMENT_TYPE", raising=False)
        monkeypatch.delenv("NEMO_TELEMETRY_ENABLED", raising=False)

        result = cli_runner.invoke(
            run,
            [
                "--data-source",
                str(dummy_csv),
                "--artifact-path",
                str(fixture_session_cache_dir),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        mock_safe_synthesizer_cls = patched_run_dependencies["safe_synthesizer_cls"]
        assert mock_safe_synthesizer_cls.call_args.kwargs["emit_telemetry"] is True
        assert "deployment_type" not in mock_safe_synthesizer_cls.call_args.kwargs
        assert os.environ["NEMO_DEPLOYMENT_TYPE"] == DeploymentTypeEnum.CLI.value

    def test_run_preserves_existing_deployment_type(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        fixture_session_cache_dir: Path,
        patched_run_dependencies: dict,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """An existing deployment type, such as Slurm, wins over the CLI default."""
        monkeypatch.setenv("NEMO_DEPLOYMENT_TYPE", DeploymentTypeEnum.SLURM.value)

        result = cli_runner.invoke(
            run,
            [
                "--data-source",
                str(dummy_csv),
                "--artifact-path",
                str(fixture_session_cache_dir),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert os.environ["NEMO_DEPLOYMENT_TYPE"] == DeploymentTypeEnum.SLURM.value

    def test_run_emit_telemetry_config_override_disables_sdk_telemetry(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        fixture_session_cache_dir: Path,
        patched_run_dependencies: dict,
    ):
        """The autogenerated --emit_telemetry option flows through config."""
        result = cli_runner.invoke(
            run,
            [
                "--data-source",
                str(dummy_csv),
                "--artifact-path",
                str(fixture_session_cache_dir),
                "--emit_telemetry",
                "false",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        mock_safe_synthesizer_cls = patched_run_dependencies["safe_synthesizer_cls"]
        assert mock_safe_synthesizer_cls.call_args.kwargs["emit_telemetry"] is False

    def test_merge_overrides_uses_env_telemetry_when_unset(self, monkeypatch: pytest.MonkeyPatch):
        """Omitting --emit_telemetry allows NEMO_TELEMETRY_ENABLED to provide the default."""
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "false")

        config = merge_overrides(None, {})

        assert config.emit_telemetry is False


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
        """Verify that --output-file is forwarded to run()."""
        custom_output = tmp_path / "custom_output.csv"

        result = cli_runner.invoke(
            run,
            [
                "--data-source",
                str(dummy_csv),
                "--output-file",
                str(custom_output),
                "--artifact-path",
                str(fixture_session_cache_dir),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        mock_ss = patched_run_dependencies["safe_synthesizer"]
        mock_ss.run.assert_called_once_with(output_file=str(custom_output))

    def test_run_without_output_file_passes_none(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        fixture_session_cache_dir: Path,
        patched_run_dependencies: dict,
    ):
        """Without --output-file, run() is called with output_file=None."""
        result = cli_runner.invoke(
            run,
            [
                "--data-source",
                str(dummy_csv),
                "--artifact-path",
                str(fixture_session_cache_dir),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        mock_ss = patched_run_dependencies["safe_synthesizer"]
        # Default output path is used if no --output-file is provided
        mock_ss.run.assert_called_once_with(output_file=None)


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

    def test_run_help_shows_runtime_settings_options(self, cli_runner: CliRunner):
        """Verify runtime PII/NER settings appear in run command help."""
        result = cli_runner.invoke(run, ["--help"])

        assert result.exit_code == 0
        assert "--inference-endpoint-url" in result.output
        assert "--inference-api-key" in result.output
        assert "--inference-model-id" in result.output
        assert "--disable-huggingface-remote" in result.output
        assert "--cpu-count" in result.output
        assert "NSS_INFERENCE_ENDPOINT" in result.output
        assert "NSS_INFERENCE_KEY" in result.output

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
                "--data-source",
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
                "--data-source",
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
                "--data-source",
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
                "--data-source",
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
        assert "--data-source" in result.output
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
                "--data-source",
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
                "--data-source",
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
                "--data-source",
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


class TestValidateMode:
    """Tests for `--validate` execution paths."""

    @pytest.mark.parametrize(
        "cli_args, skipped_attr",
        [
            pytest.param(["--validate"], "run", id="run"),
            pytest.param(["train", "--validate"], "train", id="run-train"),
        ],
    )
    def test_validate_renders_preflight_and_skips_execution(
        self,
        cli_args: list[str],
        skipped_attr: str,
        cli_runner: CliRunner,
        dummy_csv: Path,
        mock_config: MagicMock,
        mock_dataframe: MagicMock,
        mock_workdir: MagicMock,
        patched_run_dependencies: dict,
    ):
        """``--validate`` runs preflight through ``process_data(check_only=True)``,
        renders the report against the run's artifact locations, and skips
        ``run``/``train``.
        """
        mock_config.training.pretrained_model = "stub-model"
        mock_dataframe.columns = ["col1", "col2"]
        mock_dataframe.__len__.return_value = 2

        mock_ss = patched_run_dependencies["safe_synthesizer"]
        mock_ss.preflight_report = MagicMock()
        mock_ss._preflight_config_path = mock_workdir.run_dir / "safe-synthesizer-config.yaml"

        with patch("nemo_safe_synthesizer.cli.run.render_preflight_report") as mock_render:
            result = cli_runner.invoke(
                run,
                [*cli_args, "--data-source", str(dummy_csv)],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        mock_ss.process_data.assert_called_once_with(check_only=True)
        getattr(mock_ss, skipped_attr).assert_not_called()

        mock_render.assert_called_once()
        render_args = mock_render.call_args
        assert render_args.args[0] is mock_ss.preflight_report

        render_context = render_args.kwargs["context"]
        assert isinstance(render_context, PreflightRenderContext)
        assert render_context.config_path == mock_ss._preflight_config_path
        assert render_context.data_source == str(dummy_csv)
        assert render_context.artifact_dir == mock_workdir.run_dir


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
                "--data-source",
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
        mock_common_setup_return: tuple,
    ):
        """Verify generate with --run-path calls common_setup correctly."""
        run_dir = tmp_path / "trained-run"

        result = cli_runner.invoke(
            run,
            [
                "generate",
                "--data-source",
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
        mock_ss = patched_run_dependencies["safe_synthesizer"]
        mock_ss.load_from_save_path.assert_called_once_with(runtime_config=mock_common_setup_return[1])
        mock_ss.evaluate.assert_called_once_with()
        patched_run_dependencies["emit_telemetry"].assert_called_once_with(mock_ss, TaskStatusEnum.COMPLETED)

    def test_generate_emits_error_when_save_results_fails(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        tmp_path: Path,
        patched_run_dependencies: dict,
    ):
        """Verify generate reports an error status if saving results fails."""
        run_dir = tmp_path / "trained-run"
        mock_ss = patched_run_dependencies["safe_synthesizer"]
        mock_ss.save_results.side_effect = RuntimeError("save failed")

        with pytest.raises(RuntimeError, match="save failed"):
            cli_runner.invoke(
                run,
                [
                    "generate",
                    "--data-source",
                    str(dummy_csv),
                    "--run-path",
                    str(run_dir),
                ],
                catch_exceptions=False,
            )

        patched_run_dependencies["emit_telemetry"].assert_called_once_with(mock_ss, TaskStatusEnum.ERROR)
        mock_ss.generator.teardown.assert_called_once_with()

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
                "--data-source",
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
                "--data-source",
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
                "--data-source",
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
                "--data-source",
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


class TestAutoParamCliOverrides:
    """End-to-end tests for ``Auto*Param`` field CLI overrides (issue #159).

    These tests drive the real ``run`` Click command and capture the parsed
    ``synthesis_overrides`` reaching ``common_setup`` to verify each CLI value
    is parsed correctly. A separate test checks that the value also lands on
    the resolved ``SafeSynthesizerParameters`` object (using fields that pass
    through Pydantic validation unchanged -- some ``Auto*Param`` fields have
    model validators that resolve ``"auto"`` to a concrete value).
    """

    # Note: ``AutoBoolParam`` is defined in ``config.types`` but no field of
    # ``SafeSynthesizerParameters`` currently uses it (the only such field,
    # ``training.use_unsloth``, was removed when the Unsloth backend was
    # dropped). Bool conversion is exercised at the unit level by
    # ``tests/configurator/test_pydantic_click_options.py``.
    @pytest.mark.parametrize(
        "flag,raw_value,nested_path,expected",
        [
            # AutoIntParam / OptionalAutoInt
            ("--training__rope_scaling_factor", "auto", ("training", "rope_scaling_factor"), "auto"),
            ("--training__rope_scaling_factor", "2", ("training", "rope_scaling_factor"), 2),
            ("--training__num_input_records_to_sample", "auto", ("training", "num_input_records_to_sample"), "auto"),
            ("--training__num_input_records_to_sample", "100", ("training", "num_input_records_to_sample"), 100),
            ("--data__max_sequences_per_example", "auto", ("data", "max_sequences_per_example"), "auto"),
            ("--data__max_sequences_per_example", "5", ("data", "max_sequences_per_example"), 5),
            # AutoFloatParam
            ("--privacy__delta", "auto", ("privacy", "delta"), "auto"),
            ("--privacy__delta", "0.001", ("privacy", "delta"), 0.001),
        ],
    )
    def test_auto_param_override_is_parsed_into_synthesis_overrides(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        patched_run_dependencies: dict,
        flag: str,
        raw_value: str,
        nested_path: tuple[str, ...],
        expected: object,
    ):
        """Auto*Param CLI flags accept ``"auto"`` and typed values, and reach ``common_setup``."""
        result = cli_runner.invoke(
            run,
            ["--data-source", str(dummy_csv), flag, raw_value],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output

        mock_common_setup = patched_run_dependencies["common_setup"]
        mock_common_setup.assert_called_once()
        settings: CLISettings = mock_common_setup.call_args.kwargs["settings"]

        # Click parses the raw CLI string (``"auto"`` stays a string, numbers
        # and bools are coerced) and ``parse_overrides`` reshapes the flat
        # kwargs into the nested overrides dict before it reaches settings.
        node: object = settings.synthesis_overrides
        for key in nested_path:
            assert isinstance(node, dict) and key in node, (
                f"missing {'.'.join(nested_path)} in synthesis_overrides: {settings.synthesis_overrides}"
            )
            node = node[key]
        assert node == expected
        assert type(node) is type(expected)

    @pytest.mark.parametrize(
        "flag,raw_value,nested_path,expected",
        [
            # rope_scaling_factor, num_input_records_to_sample, and
            # privacy.delta pass through Pydantic validation unchanged for both
            # 'auto' and explicit values. max_sequences_per_example is excluded
            # because its model validator rewrites 'auto' to a concrete default
            # (10 with DP disabled, 1 with DP enabled).
            ("--training__rope_scaling_factor", "auto", ("training", "rope_scaling_factor"), "auto"),
            ("--training__rope_scaling_factor", "2", ("training", "rope_scaling_factor"), 2),
            ("--training__num_input_records_to_sample", "auto", ("training", "num_input_records_to_sample"), "auto"),
            ("--training__num_input_records_to_sample", "100", ("training", "num_input_records_to_sample"), 100),
            ("--privacy__delta", "auto", ("privacy", "delta"), "auto"),
            ("--privacy__delta", "0.001", ("privacy", "delta"), 0.001),
        ],
    )
    def test_auto_param_override_reaches_params_object(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        patched_run_dependencies: dict,
        flag: str,
        raw_value: str,
        nested_path: tuple[str, ...],
        expected: object,
    ):
        """The parsed CLI value also lands on the validated ``SafeSynthesizerParameters`` object."""
        result = cli_runner.invoke(
            run,
            ["--data-source", str(dummy_csv), flag, raw_value],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        settings: CLISettings = patched_run_dependencies["common_setup"].call_args.kwargs["settings"]

        params = merge_overrides(None, settings.synthesis_overrides)
        resolved: object = params
        for key in nested_path:
            resolved = getattr(resolved, key)
        assert resolved == expected
        assert type(resolved) is type(expected)


class TestRunErrorPathExitCodes:
    """Tests that run command error paths exit with non-zero status."""

    @pytest.fixture(autouse=True)
    def _reset_observability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # These tests invoke the real common_setup, which calls
        # initialize_observability() and flips the module-level
        # _INITIALIZED_OBSERVABILITY flag. Without this reset, get_logger()
        # in subsequent tests on the same xdist worker returns a
        # CategoryLogger wrapping a structlog BoundLogger, and stdlib
        # LoggerAdapter.isEnabledFor() then fails with AttributeError.
        #
        # common_setup also writes NSS_LOG_* via configure_logging_from_workdir;
        # clear those so default-value tests on the same worker are not polluted.
        monkeypatch.setattr(obs, "_INITIALIZED_OBSERVABILITY", False)
        monkeypatch.delenv("NSS_PHASE", raising=False)
        for name in ("NSS_LOG_LEVEL", "NSS_LOG_FORMAT", "NSS_LOG_FILE", "NSS_LOG_COLOR"):
            monkeypatch.delenv(name, raising=False)

    def test_run_with_no_data_source_exits_nonzero(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ):
        """`run` without --data-source must fail with a ClickException."""
        result = cli_runner.invoke(
            run,
            [
                "--artifact-path",
                str(tmp_path / "artifacts"),
            ],
        )

        assert result.exit_code != 0

    def test_run_with_nonexistent_data_source_exits_nonzero(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ):
        """`run --data-source missing.csv` must fail when the file doesn't exist."""
        missing_csv = tmp_path / "does_not_exist.csv"

        result = cli_runner.invoke(
            run,
            [
                "--data-source",
                str(missing_csv),
                "--artifact-path",
                str(tmp_path / "artifacts"),
            ],
        )

        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)

    def test_run_with_unsupported_data_source_extension_exits_nonzero(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ):
        """`run --data-source bad.xyz` must fail for an unsupported file extension."""
        bad_source = tmp_path / "bad_source.xyz"
        bad_source.write_text("irrelevant contents")

        result = cli_runner.invoke(
            run,
            [
                "--data-source",
                str(bad_source),
                "--artifact-path",
                str(tmp_path / "artifacts"),
            ],
        )

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)

    def test_generate_with_nonexistent_run_path_exits_nonzero(
        self,
        cli_runner: CliRunner,
        dummy_csv: Path,
        tmp_path: Path,
    ):
        """`run generate --run-path /nonexistent` must fail with a ClickException."""
        missing_run = tmp_path / "no_such_run"

        result = cli_runner.invoke(
            run,
            [
                "generate",
                "--data-source",
                str(dummy_csv),
                "--run-path",
                str(missing_run),
            ],
        )

        assert result.exit_code != 0


def test_common_run_options_map_to_settings_fields() -> None:
    """Every shared run flag must be backed by a CLISettings field.

    ``_settings_from_run_kwargs`` splits a command's kwargs by matching names
    against ``CLISettings.model_fields``; anything unmatched is routed to
    synthesis overrides. A shared flag whose name is not a settings field would
    therefore be silently misrouted instead of populating settings.
    """
    from nemo_safe_synthesizer.cli.run import common_run_options

    def _target(**kwargs: object) -> None: ...

    decorated = common_run_options(_target)
    option_names = {param.name for param in getattr(decorated, "__click_params__", [])}
    assert option_names, "common_run_options registered no Click options"

    unmapped = option_names - set(CLISettings.model_fields)
    assert not unmapped, f"common_run_options flags not backed by CLISettings fields: {sorted(unmapped)}"
