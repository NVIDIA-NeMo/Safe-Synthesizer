# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal tooling: rendering, reporting, and output-mode surfaces.

This package collects the "display" side of Safe Synthesizer that is
decoupled from core logic. Today it hosts the pre-flight report
renderer; over time it is expected to absorb the evaluation-report
rendering and alternative output modes (agent-friendly markdown, plain
text, JSON) behind a common [`RenderMode`][nemo_safe_synthesizer.tooling.modes.RenderMode] dispatcher.

Contract:

- Every renderer is a *function* that consumes a structured value object
  from elsewhere in the codebase. No value object should depend on this
  package.
- Rich (or any other presentation library) imports live under
  [`preflight`][nemo_safe_synthesizer.tooling.preflight] and its private
  ``_rich`` siblings, never leaking into the structured types.
- New output modes extend [`RenderMode`][nemo_safe_synthesizer.tooling.modes.RenderMode]
  and add a case in the corresponding renderer's dispatcher.
"""

from __future__ import annotations

from .modes import RenderMode
from .preflight import PreflightRenderContext, render_preflight_report

__all__ = [
    "PreflightRenderContext",
    "RenderMode",
    "render_preflight_report",
]
