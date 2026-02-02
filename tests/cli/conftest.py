# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_workdir(tmp_path: Path) -> MagicMock:
    """Create a mock Workdir that uses tmp_path for directories."""
    workdir = MagicMock()
    workdir.run_dir = tmp_path / "run"
    workdir.project_dir = tmp_path / "project"
    workdir.log_file = tmp_path / "run.log"
    workdir.output_file = tmp_path / "output.csv"
    workdir.source_dataset.path = tmp_path / "source"
    workdir.source_dataset.training = tmp_path / "source" / "training.csv"
    workdir.source_dataset.test = tmp_path / "source" / "test.csv"
    workdir.ensure_directories = MagicMock()
    workdir.phase_dir = MagicMock(return_value=tmp_path / "phase")
    return workdir


@pytest.fixture
def mock_logger() -> MagicMock:
    """Create a mock CategoryLogger."""
    logger = MagicMock()
    logger.system = MagicMock()
    logger.info = MagicMock()
    return logger


@pytest.fixture
def dummy_csv(tmp_path: Path) -> Path:
    """Create a dummy CSV file for testing."""
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("col1,col2\n1,2\n3,4\n")
    return csv_path
