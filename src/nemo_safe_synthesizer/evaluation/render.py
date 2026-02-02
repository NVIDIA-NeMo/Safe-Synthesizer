# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import datetime
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FunctionLoader, select_autoescape

from nemo_safe_synthesizer.evaluation.data_model.evaluation_report import (
    EvaluationReport,
)
from nemo_safe_synthesizer.observability import get_logger

if TYPE_CHECKING:
    from nemo_safe_synthesizer.cli.artifact_structure import Workdir

logger = get_logger(__name__)

_TEMPLATE_PATH = str(Path(__file__).parent / "assets/")


def _get_template(name):
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
    evaluation_report: EvaluationReport | None,
    template_name: str = "multi_modal_report.j2",
    output_path: str | Path | None = None,
    workdir: "Workdir | None" = None,
) -> str | None:
    if not evaluation_report:
        return None
    return render_report(evaluation_report, template_name=template_name, output_path=output_path, workdir=workdir)


def render_report(
    evaluation_report: EvaluationReport,
    template_name: str = "multi_modal_report.j2",
    output_path: str | Path | None = None,
    workdir: "Workdir | None" = None,
) -> str | None:
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
        ctx = evaluation_report.jinja_context
        output = template.render(ctx=ctx)
        if output_path:
            with open(output_path, "w") as f:
                f.write(output)
        else:
            return output
    except Exception:
        logger.exception("Failed to render report.")
    return output
