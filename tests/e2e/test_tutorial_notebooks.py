# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute tutorial notebooks top to bottom against the current environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
import pytest
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.manager import KernelManager
from nbclient import NotebookClient

TUTORIALS = Path(__file__).resolve().parents[2] / "docs" / "tutorials"


@pytest.mark.e2e
@pytest.mark.requires_gpu
@pytest.mark.timeout(5400)
@pytest.mark.skipif(sys.platform == "darwin", reason="Not applicable on macOS")
def test_safe_synthesizer_101_notebook_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An empty value skips the notebook's getpass prompt, which cannot be answered here.
    monkeypatch.setenv("NSS_INFERENCE_KEY", os.environ.get("NSS_INFERENCE_KEY", ""))
    notebook = nbformat.read(TUTORIALS / "safe-synthesizer-101.ipynb", as_version=4)
    # Without kernel dirs, jupyter_client falls back to ipykernel's built-in python3 spec, which
    # runs sys.executable. A user-level python3 kernelspec could point at another environment.
    kernel_manager = KernelManager(kernel_name="python3", kernel_spec_manager=KernelSpecManager(kernel_dirs=[]))
    client = NotebookClient(
        notebook,
        km=kernel_manager,
        timeout=3600,
        # Run from tmp_path so safe-synthesizer-artifacts/ lands there.
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()
