# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from click.testing import CliRunner


@pytest.mark.unit
@pytest.mark.parametrize("args", [["--help"], ["train", "--help"]], ids=["run", "run-train"])
def test_validate_help(args):
    """``--validate`` appears in both ``run --help`` and ``run train --help``."""
    from nemo_safe_synthesizer.cli.run import run

    result = CliRunner().invoke(run, args)
    assert "--validate" in result.output
