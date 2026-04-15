# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rich-based ``PreflightReport`` renderer.

Called by [`render_preflight_report`][nemo_safe_synthesizer.tooling.preflight.render_preflight_report]
when ``mode=RenderMode.RICH``. Kept private so the exact composition
of panels/trees/rules can evolve without breaking callers; the stable
entry point is the ``render_preflight_report`` free function.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

if TYPE_CHECKING:
    from ...preflight import PreflightRegistry, PreflightReport
    from ..preflight import PreflightRenderContext

__all__ = ["render_rich"]


def render_rich(
    report: PreflightReport,
    *,
    registry: PreflightRegistry,
    context: PreflightRenderContext,
    console: Console | None = None,
) -> None:
    """Print a Rich-formatted validation report to the console."""
    if console is None:
        console = Console()

    n_errors = len(report.errors)
    n_warns = len(report.warnings)
    has_run_info = bool(context.run_info)
    has_output_locations = context.artifact_dir is not None or context.config_path is not None
    has_followup = not n_errors and context.config_path is not None and context.data_source is not None
    header = _build_header(n_errors, n_warns)

    console.print()
    if has_run_info:
        console.rule("[dim]runtime info[/]", style="dim")
        _print_run_info(console, context.run_info)
    console.rule(header, style="dim")
    for panel in _build_category_panels(report, registry, n_errors, n_warns):
        console.print(panel)
    if has_output_locations:
        console.rule("[dim]output locations[/]", style="dim")
        console.print()
        _print_output_locations(
            console,
            context.artifact_dir,
            context.config_path,
            context.log_file,
        )

    if has_followup:
        console.rule("[dim]next steps[/]", style="dim")
        console.print()
        _print_followup_command(
            console,
            data_source=context.data_source,
            config_path=context.config_path,
        )


def _build_header(n_errors: int, n_warns: int) -> Text:
    """Build the summary heading for the pre-flight output."""
    if n_errors:
        return Text(
            f"Pre-flight validation failed with {n_errors} error(s), {n_warns} warning(s)",
            style="bold red",
        )
    if n_warns:
        return Text(f"Pre-flight validation passed with {n_warns} warning(s)", style="bold yellow")
    return Text("Pre-flight validation passed with 0 warning(s)", style="bold green")


def _build_category_panels(
    report: PreflightReport,
    registry: PreflightRegistry,
    n_errors: int,
    n_warns: int,
) -> list[Panel]:
    """Build one panel per check category, preserving first-seen registry order.

    The report carries per-check results (name, status, issues). Display
    metadata (``label``, ``category``) lives on the check classes and is
    looked up from the registry by check name at render time.
    """
    # Aggregate by category preserving first-seen registry order from the
    # report. Using itertools.groupby here would drop non-contiguous
    # categories (e.g. plugin checks interleaved with core ones) into
    # multiple panels for the same category.
    categories: dict[str, list] = {}
    for result in report.checks:
        if result.name not in registry:
            continue  # orphan result (registry mutated between run and render)
        categories.setdefault(registry[result.name].category, []).append(result)

    panels: list[Panel] = []
    for category, group in categories.items():
        lines: list[Text] = []
        for result in group:
            check = registry[result.name]
            if result.status == "skipped":
                lines.append(
                    Text.assemble(
                        ("- ", "dim"),
                        check.label,
                        (": ", "dim"),
                        ("⊘ skipped (blocked by a prior failed/disabled check)", "dim"),
                    )
                )
                continue

            if not result.issues:
                lines.append(Text.assemble(("- ", "dim"), check.label, (": ", "dim"), ("✓", "green")))
                continue

            has_errors = any(i.severity == "error" for i in result.issues)
            icon, style = ("✗", "bold red") if has_errors else ("⚠", "yellow")
            lines.append(Text.assemble(("- ", "dim"), check.label, (": ", "dim"), (icon, style)))
            for issue in result.issues:
                lines.append(
                    Text.assemble(
                        ("     - ", "dim"),
                        (f"{issue.code}", "dim"),
                        (": ", "dim"),
                        issue.message,
                    )
                )

        panels.append(
            Panel(
                Group(*lines),
                title=f"[dim]{category}[/]",
                title_align="left",
                border_style="dim",
                padding=(0, 1),
            )
        )

    if n_errors and panels:
        panels[-1] = Panel(
            panels[-1].renderable,
            title=panels[-1].title,
            title_align="left",
            border_style="dim",
            subtitle=f"[bold red]{n_errors} error(s)[/], {n_warns} warning(s) -- fix errors before running",
            subtitle_align="left",
            padding=(0, 1),
        )

    return panels


def _print_run_info(console: Console, run_info: dict[str, str] | None) -> None:
    """Print run metadata as a compact key/value list."""
    if not run_info:
        return

    for key, value in run_info.items():
        console.print(Text.assemble((f"{key}:  ", "dim"), (value, "")))
    console.print()


def _print_output_locations(
    console: Console,
    artifact_dir: Path | None,
    config_path: Path | None,
    log_file: Path | None,
) -> None:
    """Print artifact/config/log locations."""
    output_width = max(console.width, 200)

    if artifact_dir is not None:
        tree = Tree(Text("artifact dir", style="bold"), guide_style="dim")
        wrapped_root_path = _wrapped_path(artifact_dir, max_width=max(20, console.width - 8))
        tree.add(Text(wrapped_root_path, overflow="ignore"))
        if config_path is not None:
            tree.add(
                _location_label(
                    "resolved config",
                    _relative_if_possible(config_path, artifact_dir),
                )
            )
        if log_file is not None:
            tree.add(
                _location_label(
                    "log file",
                    _relative_if_possible(log_file, artifact_dir),
                )
            )
        console.print(tree, soft_wrap=True, crop=False, overflow="fold", width=output_width)
        console.print()
        return

    if config_path is not None:
        tree = Tree(Text("resolved config", style="bold"), guide_style="dim")
        tree.add(Text(str(config_path), overflow="fold"))
        console.print(tree, soft_wrap=True, crop=False, overflow="fold", width=output_width)
        console.print()


def _wrapped_path(path: Path, max_width: int) -> str:
    """Greedily wrap ``path`` at separator boundaries to fit ``max_width``.

    Returns the full string if it already fits or if the path has no
    parts to split on. Continuation lines start with ``os.sep`` so the
    break is visually obvious.
    """
    path_text = str(path)
    if len(path_text) <= max_width or not path.parts:
        return path_text

    lines: list[str] = []
    line = path.parts[0]
    for part in path.parts[1:]:
        candidate = str(Path(line) / part)
        if len(candidate) <= max_width:
            line = candidate
        else:
            lines.append(line)
            line = os.sep + part
    lines.append(line)
    return "\n".join(lines)


def _location_label(label: str, value: Path | str) -> Text:
    return Text.assemble(
        (f"{label}: ", "bold"),
        Text(str(value), overflow="fold"),
    )


def _relative_if_possible(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _print_followup_command(
    console: Console,
    *,
    data_source: str | None,
    config_path: Path | None,
) -> None:
    """Print a copy/paste follow-up command when there are no errors."""
    if data_source is None or config_path is None:
        return

    quoted_data_source = shlex.quote(data_source)
    quoted_config_path = shlex.quote(str(config_path))
    console.print(Text("Run with the resolved configuration:", style="dim"))
    console.print(
        Text(
            f"  safe-synthesizer run --data-source {quoted_data_source} \\",
            no_wrap=True,
            overflow="ignore",
        ),
        soft_wrap=True,
    )
    console.print(
        Text(f"    --config {quoted_config_path}", no_wrap=True, overflow="ignore"),
        soft_wrap=True,
    )
    console.print()
    console.print(
        Text(
            "Note: the full run will create a new timestamped directory for its artifacts unless you specify a path.",
            style="dim italic",
        )
    )
