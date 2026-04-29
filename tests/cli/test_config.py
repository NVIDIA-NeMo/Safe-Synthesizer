# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``safe-synthesizer config`` command group.

Focuses on error-path exit codes: every failure mode must produce a non-zero
exit code so automation and scripts that check ``$?`` can detect problems.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from nemo_safe_synthesizer.cli.config import config


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


class TestConfigValidateErrorPathExitCodes:
    """Ensure ``config validate`` returns a non-zero exit code on failures.

    Each test asserts ``exit_code != 0`` rather than a specific value because
    Click's exit code varies by failure mode: ``FileNotFoundError`` bubbles up
    as an unhandled exception (exit 1), while ``ValidationError`` is caught
    in ``merge_overrides`` and re-raised via ``sys.exit(1)``.
    """

    def test_validate_nonexistent_config_exits_nonzero(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ):
        """`config validate --config nonexistent.yaml` must fail when the file doesn't exist."""
        missing_config = tmp_path / "does_not_exist.yaml"

        result = cli_runner.invoke(
            config,
            [
                "validate",
                "--config",
                str(missing_config),
            ],
        )

        assert result.exit_code != 0
        assert result.exception is not None
        assert isinstance(result.exception, FileNotFoundError)

    def test_validate_malformed_config_exits_nonzero(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ):
        """`config validate --config malformed.yaml` must fail when the YAML fails schema validation."""
        malformed_config = tmp_path / "malformed.yaml"
        # YAML parses cleanly, but the field values violate SafeSynthesizerParameters:
        # ``training.batch_size`` must be a positive integer, and ``data.holdout`` must
        # be a number in [0, 1]. Either one alone is enough to trigger a ValidationError.
        malformed_config.write_text(
            "training:\n"
            "  batch_size: -1\n"
            "data:\n"
            "  holdout: not_a_number\n"
        )

        result = cli_runner.invoke(
            config,
            [
                "validate",
                "--config",
                str(malformed_config),
            ],
        )

        assert result.exit_code != 0

    def test_validate_unparseable_yaml_exits_nonzero(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ):
        """`config validate --config garbage.yaml` must fail when the file is not valid YAML."""
        broken_yaml = tmp_path / "broken.yaml"
        # Unbalanced brackets trigger yaml.YAMLError during safe_load.
        broken_yaml.write_text("training: {batch_size: 1\n")

        result = cli_runner.invoke(
            config,
            [
                "validate",
                "--config",
                str(broken_yaml),
            ],
        )

        assert result.exit_code != 0

    def test_validate_missing_required_config_flag_exits_nonzero(
        self,
        cli_runner: CliRunner,
    ):
        """`config validate` without `--config` must fail because the option is required."""
        result = cli_runner.invoke(config, ["validate"])

        assert result.exit_code != 0
        assert "--config" in result.output

    def test_validate_valid_config_exits_zero(
        self,
        cli_runner: CliRunner,
        tmp_path_factory,
        fixture_yaml_config_str: str,
    ):
        """Positive control: a valid YAML config must exit 0.

        Guards against a regression where ``validate`` starts returning a
        non-zero exit code for correct input, which would invert the contract
        these error-path tests protect.
        """
        tmp_path = tmp_path_factory.mktemp("valid_config", numbered=True)
        valid_config = tmp_path / "valid.yaml"
        valid_config.write_text(fixture_yaml_config_str)

        result = cli_runner.invoke(
            config,
            [
                "validate",
                "--config",
                str(valid_config),
            ],
        )

        assert result.exit_code == 0
        assert "is valid" in result.output
