# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and save dataset-specific PII replacement plans."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from ...config.replace_pii import PiiReplacementPlan
from ...errors import ParameterError

__all__ = ["load_plan", "save_plan"]

_CURRENT_PLAN_SCHEMA_VERSION = 3


def _plan_body(raw: dict[object, object], plan_path: Path) -> dict[object, object]:
    """Validate document metadata and return the unversioned runtime plan body."""
    body = dict(raw)
    version = body.pop("schema_version", _CURRENT_PLAN_SCHEMA_VERSION)
    if type(version) is not int:
        raise ParameterError(f"PII replacement plan file {str(plan_path)!r} schema_version must be an integer")
    if version != _CURRENT_PLAN_SCHEMA_VERSION:
        raise ParameterError(
            f"PII replacement plan file {str(plan_path)!r} uses unsupported schema version {version}; "
            f"this NSS release supports version {_CURRENT_PLAN_SCHEMA_VERSION}"
        )
    return body


def load_plan(path: str | Path) -> PiiReplacementPlan:
    """Load a replacement plan from a standalone YAML file.

    A plan file contains ``schema_version`` metadata followed by the same fields
    accepted as an inline ``replace_pii.replacement_plan`` value. A missing
    version is interpreted as version 3.

    Args:
        path: YAML file containing a replacement-plan mapping.

    Returns:
        The parsed, context-free validated plan.

    Raises:
        ParameterError: If the file cannot be read, parsed, or validated.
    """
    plan_path = Path(path)
    try:
        raw = yaml.safe_load(plan_path.read_text())
    except OSError as exc:
        raise ParameterError(f"Could not read PII replacement plan file {str(plan_path)!r}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ParameterError(f"Invalid YAML in PII replacement plan file {str(plan_path)!r}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ParameterError(f"PII replacement plan file {str(plan_path)!r} must contain a mapping")

    try:
        return PiiReplacementPlan.model_validate(_plan_body(raw, plan_path))
    except ValidationError as exc:
        raise ParameterError(f"Invalid PII replacement plan in {str(plan_path)!r}: {exc}") from exc


def save_plan(plan: PiiReplacementPlan, path: str | Path) -> Path:
    """Save a replacement plan as a versioned reusable standalone YAML document."""
    plan_path = Path(path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": _CURRENT_PLAN_SCHEMA_VERSION,
        **plan.model_dump(mode="json", exclude_unset=False),
    }
    plan_path.write_text(yaml.safe_dump(document, sort_keys=False))
    return plan_path
