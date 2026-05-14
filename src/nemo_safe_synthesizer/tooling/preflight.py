# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rendering surface for [`PreflightReport`][nemo_safe_synthesizer.preflight.PreflightReport].

The preflight package produces structured values; this module turns
them into human- (and, eventually, agent-) readable output. Callers
should use [`render_preflight_report`][nemo_safe_synthesizer.tooling.preflight.render_preflight_report]
and select the output mode via [`RenderMode`][nemo_safe_synthesizer.tooling.modes.RenderMode].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .modes import RenderMode

if TYPE_CHECKING:
    from rich.console import Console

    from ..preflight import PreflightRegistry, PreflightReport

__all__ = [
    "PreflightRenderContext",
    "render_preflight_report",
]


@dataclass(frozen=True)
class PreflightRenderContext:
    """Display-only context threaded into a preflight-report render call.

    None of these fields are part of ``PreflightReport`` itself --
    they are caller-supplied extras (usually from the CLI) that shape
    what auxiliary sections the renderer draws.
    """

    config_path: Path | None = None
    data_source: str | None = None
    artifact_dir: Path | None = None
    log_file: Path | None = None
    run_info: dict[str, str] | None = None


def render_preflight_report(
    report: PreflightReport,
    *,
    registry: PreflightRegistry,
    context: PreflightRenderContext | None = None,
    mode: RenderMode = RenderMode.RICH,
    console: Console | None = None,
) -> None:
    """Render ``report`` to the given output ``mode``.

    Args:
        report: The structured preflight report to render.
        registry: The registry used to produce ``report``. Supplies the
            per-check display metadata (label, category) and the panel
            ordering; the report itself only carries raw issues and
            statuses.
        context: Optional display-only context (paths, run info) that
            tells the renderer which auxiliary sections to emit.
        mode: Output format. Currently only ``RenderMode.RICH`` is
            implemented.
        console: Only consulted for Rich output. Defaults to a new
            ``rich.console.Console``.

    Raises:
        NotImplementedError: If ``mode`` is a ``RenderMode`` value
            whose backend has not yet been implemented.
    """
    if context is None:
        context = PreflightRenderContext()

    if mode is RenderMode.RICH:
        from ._rich.preflight import render_rich

        render_rich(report, registry=registry, context=context, console=console)
        return

    raise NotImplementedError(f"Preflight render mode {mode!r} is not implemented yet.")
