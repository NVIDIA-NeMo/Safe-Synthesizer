# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from click.testing import CliRunner


@pytest.mark.unit
def test_validate_help():
    """--validate appears in help text."""
    from nemo_safe_synthesizer.cli.run import run

    runner = CliRunner()
    result = runner.invoke(run, ["--help"])
    assert "--validate" in result.output


@pytest.mark.unit
def test_validate_train_help():
    """--validate appears in run train help text."""
    from nemo_safe_synthesizer.cli.run import run

    runner = CliRunner()
    result = runner.invoke(run, ["train", "--help"])
    assert "--validate" in result.output
