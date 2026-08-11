# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan resolution and YAML persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from pydantic import ValidationError

from ...config.data import DataParameters
from ...config.replace_pii import PiiReplacementPlan, PiiReplacerConfig
from ...config.time_series import TimeSeriesParameters
from ...errors import ParameterError
from ...observability import get_logger
from .. import entities
from ..llm import PiiEnhancer
from .validation import protected_columns, strip_protected_columns_from_plan, validate_plan

logger = get_logger(__name__)


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
    if not isinstance(data, dict):
        raise ParameterError(f"plan file {path!r} must contain a mapping")
    try:
        return PiiReplacementPlan.model_validate(data)
    except ValidationError as exc:
        raise ParameterError(f"Invalid PII replacement plan in {path!r}: {exc}") from exc


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
    "persona_backed_columns": (
        "# Columns that describe a person. Each entry is one persona whose\n"
        "# columns_to_replace are filled from a single synthetic identity, so the\n"
        "# values stay consistent with each other. match_persona_by lists existing\n"
        "# columns that constrain which persona is drawn (those columns are read,\n"
        "# never replaced). Put free_text and entity-driven columns\n"
        "# (IDs, cards, IPs, date_of_birth; phone when not using pgm) in\n"
        "# standalone_columns_to_replace instead.\n"
    ),
    "standalone_columns_to_replace": (
        "# Columns replaced on their own, with no persona behind them: record IDs,\n"
        "# free-text notes, and other values that need no cross-column consistency.\n"
        "# Person-identifying columns listed only here do not share a synthetic\n"
        "# person with other columns.\n"
    ),
}


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
    config: PiiReplacerConfig,
    df: pd.DataFrame,
    *,
    data_config: DataParameters,
    cfg: entities.Config,
    time_series: TimeSeriesParameters | None = None,
    enhancer: PiiEnhancer | None = None,
) -> PiiReplacementPlan:
    """Resolve a replacement plan from auto-discovery, file, or inline config.

    Auto-discovered plans strip protected structural columns before validation.
    User-supplied plans that list protected columns fail validation instead of
    being silently rewritten.

    Args:
        config: User-facing PII replacement configuration.
        df: Input dataframe the plan will be applied to.
        data_config: Data parameters with group and order keys.
        cfg: Engine configuration for discovery thresholds.
        time_series: Optional time-series parameters for protected-column rules.
        enhancer: Optional LLM enhancer injected for discovery review.

    Returns:
        Validated ``PiiReplacementPlan`` ready for replacement.

    Raises:
        ParameterError: When the plan source is missing, invalid, or fails user/config validation.
        InternalError: When auto-discovery emits an unexpected invalid plan.
    """
    # ``llm_enhancement`` is refused by the enhancer discovery/apply select; do not
    # gate here so the failure site matches the future stacking call path.

    group_key = data_config.group_training_examples_by
    protected = protected_columns(data_config, time_series=time_series)

    if config.is_auto_discovery:
        from .discovery import discover_plan

        plan = discover_plan(
            df,
            group_key=group_key,
            cfg=cfg,
            config=config,
            enhancer=enhancer,
        )
        # Auto-discovery may still classify structural columns as PII; drop them
        # before validation so the emitted plan is runnable.
        removed = strip_protected_columns_from_plan(plan, protected)
        if removed:
            logger.user.warning(
                "[PII Replacement] Leaving structural columns unchanged (not replaced): "
                + ", ".join(repr(c) for c in removed)
                + ". Group keys are preserved in time-series mode; order-by / timestamp columns "
                "are preserved when configured."
            )
    elif config.plan_path:
        plan = load_plan_from_path(config.plan_path)
    else:
        inline = config.inline_plan
        if inline is None:
            raise ParameterError("replacement_plan must be auto_discovery, a path, or an inline plan")
        plan = inline

    # User-supplied plans that list protected columns fail validation instead of
    # being silently rewritten. Auto-discovered plans are validated too (defense
    # in depth); unexpected failures there are discovery bugs.
    validate_plan(
        df,
        plan,
        data_config=data_config,
        time_series=time_series,
        plan_origin="auto_discovery" if config.is_auto_discovery else "user",
    )
    return plan
