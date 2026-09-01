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


def load_plan(path: str | Path) -> PiiReplacementPlan:
    """Load a replacement plan from a standalone YAML file.

    A plan file contains the same mapping accepted as an embedded
    ``replace_pii.replacement_plan`` value in the main NSS configuration.

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
        return PiiReplacementPlan.model_validate(raw)
    except ValidationError as exc:
        raise ParameterError(f"Invalid PII replacement plan in {str(plan_path)!r}: {exc}") from exc


def save_plan(plan: PiiReplacementPlan, path: str | Path) -> Path:
    """Save a replacement plan as reusable standalone YAML."""
    plan_path = Path(path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_yaml(plan_path, exclude_unset=False)
    return plan_path
