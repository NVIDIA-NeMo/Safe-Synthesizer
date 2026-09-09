# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and save dataset-specific PII replacement plans."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from ...config.replace_pii import PiiReplacementPlan
from ...config.validation import format_pydantic_validation_error
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
        details = format_pydantic_validation_error(exc)
        raise ParameterError(f"Invalid PII replacement plan in {str(plan_path)!r} ({details})") from exc


def _plan_document(plan: PiiReplacementPlan) -> dict[str, object]:
    """Return canonical sparse YAML data while preserving inferred-type omission."""
    columns: list[dict[str, object]] = []
    for spec in plan.columns_to_replace:
        serialized_spec: dict[str, object] = {
            "column_name": spec.column_name,
            "entity_type": spec.entity_type.value,
        }
        if spec.pattern is not None:
            serialized_spec["pattern"] = spec.pattern
        if spec.depends_on:
            dependencies: list[dict[str, object]] = []
            for dependency in spec.depends_on:
                serialized_dependency: dict[str, object] = {"column_name": dependency.column_name}
                if "entity_type" in dependency.model_fields_set and dependency.entity_type is not None:
                    serialized_dependency["entity_type"] = dependency.entity_type.value
                dependencies.append(serialized_dependency)
            serialized_spec["depends_on"] = dependencies
        columns.append(serialized_spec)
    return {
        "schema_version": _CURRENT_PLAN_SCHEMA_VERSION,
        "scope": plan.scope.value,
        "columns_to_replace": columns,
    }


def save_plan(plan: PiiReplacementPlan, path: str | Path) -> Path:
    """Save a replacement plan as a versioned reusable standalone YAML document."""
    plan_path = Path(path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(yaml.safe_dump(_plan_document(plan), sort_keys=False))
    return plan_path
