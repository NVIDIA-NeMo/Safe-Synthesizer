# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan YAML persistence and source resolution (discovery / file / inline)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from pydantic import ValidationError

from ...config.replace_pii import PiiReplacementPlan, ReplacePiiConfig
from ...errors import ParameterError
from .. import entities

PII_REPLACEMENT_PLAN_FILENAME = "pii_replacement_plan.yaml"

PLAN_YAML_HEADER = """\
# PII replacement plan for this dataset. Edit and pass back via
# replace_pii.replacement_plan to override auto-discovery.

"""

# Explanations emitted above each top-level section so a plan file can be read
# without the docs. Keyed by field name of ``PiiReplacementPlan``.
PLAN_YAML_SECTION_COMMENTS: dict[str, str] = {
    "scope": (
        "# How widely one original value keeps the same synthetic value:\n"
        "# record (per row), group (per training group), or dataframe (whole dataset).\n"
    ),
    "columns_to_replace": (
        "# Flat columns_to_replace with optional depends_on / pattern.\n"
        "# Identify-only conditioners (gender, …) appear only as depends_on.\n"
    ),
}


def load_plan_from_path(path: str) -> PiiReplacementPlan:
    """Load and validate a PII replacement plan from a YAML file.

    Args:
        path: Filesystem path to the plan YAML file.

    Returns:
        Validated ``PiiReplacementPlan`` instance.

    Raises:
        ParameterError: When the file cannot be read, parsed, or validated.
    """
    p = Path(path)
    try:
        text = p.read_text()
    except OSError as exc:
        raise ParameterError(f"Could not read PII replacement plan file {path!r}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ParameterError(f"Invalid YAML in PII replacement plan file {path!r}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ParameterError(f"plan file {path!r} must contain a mapping")
    try:
        return PiiReplacementPlan.model_validate(data)
    except ValidationError as exc:
        raise ParameterError(f"Invalid PII replacement plan in {path!r}: {exc}") from exc


def plan_to_commented_yaml(plan: PiiReplacementPlan) -> str:
    """Serialize a plan to YAML with a header and per-section explanations.

    Args:
        plan: Replacement plan to serialize.

    Returns:
        YAML string with header comments and section explanations.
    """
    data = json.loads(plan.model_dump_json(exclude_none=True, exclude_defaults=True))
    dumped = yaml.safe_dump(data, sort_keys=False)

    lines: list[str] = []
    for line in dumped.splitlines():
        key = line.split(":", 1)[0]
        # Only top-level keys start at column 0, so nested keys never match.
        if line and not line[0].isspace() and (comment := PLAN_YAML_SECTION_COMMENTS.get(key)):
            if lines:
                lines.append("")
            lines.append(comment.rstrip("\n"))
        lines.append(line)
    return PLAN_YAML_HEADER + "\n".join(lines) + "\n"


def save_plan_to_path(plan: PiiReplacementPlan, path: str | Path) -> Path:
    """Write a replacement plan as YAML, omitting null and default values.

    Args:
        plan: Replacement plan to persist.
        path: Output file path; parent directories are created as needed.

    Returns:
        Resolved output ``Path``.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(plan_to_commented_yaml(plan))
    return out


def resolve_plan(
    config: ReplacePiiConfig,
    df: pd.DataFrame,
    *,
    group_key: str | None,
) -> PiiReplacementPlan:
    """Resolve a replacement plan from auto-discovery, file, or inline config.

    Validation and protected-column stripping land in follow-up PRs. This path
    only materializes the plan that ``process_data`` writes to the run directory.

    Args:
        config: User-facing PII replacement configuration.
        df: Input dataframe used for auto-discovery.
        group_key: Training group column name, or ``None`` for dataframe scope.

    Returns:
        ``PiiReplacementPlan`` from discovery, a plan file, or an inline plan.

    Raises:
        ParameterError: When the plan source is missing or invalid.
    """
    if config.is_auto_discovery:
        from .discovery import discover_plan

        return discover_plan(
            df,
            group_key,
            entities.config_from_replace_pii(config),
            config,
        )
    if config.plan_path is not None:
        return load_plan_from_path(config.plan_path)
    if config.inline_plan is not None:
        return config.inline_plan
    raise ParameterError(
        "replace_pii.replacement_plan must be 'auto_discovery', a path to a plan file, or an inline plan"
    )
