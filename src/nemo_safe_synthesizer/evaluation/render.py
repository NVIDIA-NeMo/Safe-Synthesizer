# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jinja2 report rendering for evaluation results.

Loads HTML/Jinja templates from the ``assets/`` directory and renders
``EvaluationReportData`` into a self-contained HTML file.
"""

from __future__ import annotations

import datetime
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FunctionLoader, select_autoescape

from ..config.evaluation_report_data import EvaluationReportData
from ..observability import get_logger
from .report_context import report_context
from .report_data import build_report_data
from .reports.multimodal.multimodal_report import MultimodalReport

if TYPE_CHECKING:
    from ..cli.artifact_structure import Workdir

logger = get_logger(__name__)

_TEMPLATE_PATH = str(Path(__file__).parent / "assets/")


def _get_template(name: str) -> str | None:
    """Load a Jinja template by relative path, falling back to the filesystem.

    Tries ``pkgutil.get_data`` first (works when installed as a package),
    then falls back to a direct file read (useful in the test suite).

    Args:
        name: Relative path to the template within the assets directory.

    Returns:
        The template content as a string, or ``None`` if not found.
    """
    try:
        # If this module is being used as someone else's dependency, this will work
        data = pkgutil.get_data(__name__, _TEMPLATE_PATH + name)
        if data is None:
            return None
        template = data.decode()
    except FileNotFoundError:
        # If we get here, we are probably in the test suite.  Look for the file as if we are.
        try:
            with open(_TEMPLATE_PATH + "/" + name) as f:
                template = f.read()
        except FileNotFoundError:
            template = None
    return template


def maybe_render_report(
    evaluation_report: MultimodalReport | EvaluationReportData | None,
    template_name: str = "multi_modal_report.j2",
    output_path: str | Path | None = None,
    workdir: "Workdir | None" = None,
) -> str | None:
    """Render the evaluation report if one is provided, otherwise return ``None``."""
    if not evaluation_report:
        return None
    return render_report(evaluation_report, template_name=template_name, output_path=output_path, workdir=workdir)


def render_report(
    evaluation_report: MultimodalReport | EvaluationReportData,
    template_name: str = "multi_modal_report.j2",
    output_path: str | Path | None = None,
    workdir: "Workdir | None" = None,
) -> str | None:
    """Render an evaluation report to HTML using a Jinja2 template.

    Args:
        evaluation_report: The report data to render, or a completed report to build it from.
        template_name: Jinja2 template filename relative to ``assets/jinja/reports/``.
        output_path: File path to write the rendered HTML. If ``None`` and
            ``workdir`` is provided, falls back to ``workdir.evaluation_report``.
        workdir: Working directory structure for output path resolution.

    Returns:
        The rendered HTML string, or ``None`` if rendering fails.
    """
    # Resolve output path from workdir if not explicitly provided
    if output_path is None and workdir is not None:
        output_path = workdir.evaluation_report

    env = Environment(
        loader=FunctionLoader(_get_template),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("jinja/reports/" + template_name)
    template.globals["now"] = datetime.datetime.now(datetime.timezone.utc)

    output = None
    try:
        data = (
            evaluation_report
            if isinstance(evaluation_report, EvaluationReportData)
            else build_report_data(evaluation_report)
        )
        output = template.render(ctx=report_context(data))
        if output_path:
            with open(output_path, "w") as f:
                f.write(output)
        else:
            return output
    except Exception:
        logger.exception("Failed to render report.")
    return output
