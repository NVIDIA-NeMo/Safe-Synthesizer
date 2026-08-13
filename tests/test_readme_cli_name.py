# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_dataset_registry_uses_cli_name():
    readme = README.read_text(encoding="utf-8")
    # Positive check: the documented dataset-registry example must use the CLI
    # entry point name, not the Python package name.
    assert "safe-synthesizer run --dataset-registry" in readme, (
        "README dataset-registry example should show the CLI entry point "
        "'safe-synthesizer', not the Python package name"
    )
    assert "nemo-safe-synthesizer run" not in readme, (
        "README dataset-registry example uses package name 'nemo-safe-synthesizer' "
        "instead of CLI entry point 'safe-synthesizer'"
    )
