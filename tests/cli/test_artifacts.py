# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CLI artifacts command exit codes."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from nemo_safe_synthesizer.cli.artifacts import artifacts

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


# =============================================================================
# Tests
# =============================================================================


class TestCleanExitCodes:
    """Verify that 'artifacts clean' uses non-zero exit codes on failure."""

    def test_clean_invalid_workdir_exits_nonzero(self, cli_runner: CliRunner, tmp_path: Path):
        """A directory that exists but isn't a valid workdir should exit 1."""
        result = cli_runner.invoke(artifacts, ["clean", "--artifact-path", str(tmp_path)])
        assert result.exit_code != 0

    def test_clean_nonexistent_path_exits_nonzero(self, cli_runner: CliRunner):
        """A path that doesn't exist is rejected by Click (exit 2)."""
        result = cli_runner.invoke(artifacts, ["clean", "--artifact-path", "/no/such/path"])
        assert result.exit_code != 0

    def test_clean_deletion_failure_exits_nonzero(self, cli_runner: CliRunner, tmp_path: Path):
        """A permission/OS error during deletion should exit 1."""
        with patch("nemo_safe_synthesizer.cli.artifacts.Workdir") as mock_workdir_cls:
            mock_workdir = mock_workdir_cls.from_path.return_value
            mock_workdir.run_dir = tmp_path / "run"
            mock_workdir.run_dir.mkdir()

            with patch("shutil.rmtree", side_effect=PermissionError("denied")):
                result = cli_runner.invoke(
                    artifacts, ["clean", "--artifact-path", str(tmp_path), "--force"]
                )

        assert result.exit_code != 0
        assert "denied" in result.output

    def test_clean_successful_exits_zero(self, cli_runner: CliRunner, tmp_path: Path):
        """A successful clean should exit 0."""
        with patch("nemo_safe_synthesizer.cli.artifacts.Workdir") as mock_workdir_cls:
            mock_workdir = mock_workdir_cls.from_path.return_value
            mock_workdir.run_dir = tmp_path / "run"
            mock_workdir.run_dir.mkdir()

            result = cli_runner.invoke(
                artifacts, ["clean", "--artifact-path", str(tmp_path), "--force"]
            )

        assert result.exit_code == 0
