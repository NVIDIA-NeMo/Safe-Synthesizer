# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the wandb_setup module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from nemo_safe_synthesizer.cli import wandb_setup
from nemo_safe_synthesizer.cli.wandb_setup import resolve_wandb_run_id


class TestResolveWandbRunId:
    """Tests for resolve_wandb_run_id function."""

    def test_resolve_direct_id(self):
        """resolve_wandb_run_id returns the ID directly when it's not a file path."""
        run_id = "abc123xyz"
        result = resolve_wandb_run_id(run_id)
        assert result == run_id

    def test_resolve_from_file(self, tmp_path: Path):
        """resolve_wandb_run_id reads ID from file when path exists."""
        run_id = "file_based_run_id_456"
        id_file = tmp_path / "wandb_run_id.txt"
        id_file.write_text(run_id)

        result = resolve_wandb_run_id(str(id_file))
        assert result == run_id

    def test_resolve_from_file_strips_whitespace(self, tmp_path: Path):
        """resolve_wandb_run_id strips whitespace from file content."""
        run_id = "run_id_with_newline"
        id_file = tmp_path / "wandb_run_id.txt"
        id_file.write_text(f"  {run_id}  \n")

        result = resolve_wandb_run_id(str(id_file))
        assert result == run_id

    def test_resolve_nonexistent_path_treated_as_id(self):
        """resolve_wandb_run_id treats nonexistent paths as direct IDs."""
        # This could be a valid wandb run ID that happens to look like a path
        run_id = "/nonexistent/path/that/is/actually/an/id"
        result = resolve_wandb_run_id(run_id)
        assert result == run_id

    def test_resolve_directory_treated_as_id(self, tmp_path: Path):
        """resolve_wandb_run_id treats directories as direct IDs (not files)."""
        # Create a directory (not a file)
        dir_path = tmp_path / "some_directory"
        dir_path.mkdir()

        result = resolve_wandb_run_id(str(dir_path))
        # Since it's a directory, not a file, it's returned as-is
        assert result == str(dir_path)


def test_failure_reporting_updates_summary_without_history() -> None:
    """Failure status is final run state, not an additional W&B history point."""
    fake_run = MagicMock()
    with patch.object(wandb_setup.wandb, "run", fake_run), patch.object(wandb_setup.wandb, "log") as mock_log:
        wandb_setup.log_failure_to_wandb(ValueError("bad input"), "generate")

    mock_log.assert_not_called()
    assert fake_run.summary.update.call_args.args[0] == {
        "eval/success": 0,
        "generate/error_type": "ValueError",
        "generate/error_message": "bad input",
    }
