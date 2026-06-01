# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the hub-free CLI import invariant.

``huggingface_hub`` caches ``HF_HUB_OFFLINE`` at import time. The CLI propagates
the ``--(enable|disable)-huggingface-remote`` flag to that env var inside
``common_setup``; for the propagation to take effect, ``huggingface_hub`` must
not be imported during the ``cli.cli`` import chain. Run in a subprocess because
``sys.modules`` is process-global and other tests import ``huggingface_hub``.
"""

from __future__ import annotations

import subprocess
import sys


def test_importing_cli_does_not_import_huggingface_hub():
    code = (
        "import sys;"
        "import nemo_safe_synthesizer.cli.cli;"
        "loaded = 'huggingface_hub' in sys.modules;"
        "print('LOADED' if loaded else 'CLEAN');"
        "sys.exit(1 if loaded else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "Importing nemo_safe_synthesizer.cli.cli pulled in huggingface_hub, which "
        "caches HF_HUB_OFFLINE at import time and breaks --(enable|disable)-"
        f"huggingface-remote propagation.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
