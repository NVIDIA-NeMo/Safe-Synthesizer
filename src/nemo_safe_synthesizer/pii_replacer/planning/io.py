# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and save dataset-specific PII replacement plans."""

from __future__ import annotations

import json
import re
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml
from pydantic import ValidationError

from ...config.replace_pii import PiiReplacementPlan
from ...config.unknown_fields import normalize_unknown_fields
from ...config.validation import format_pydantic_validation_error
from ...errors import ParameterError

if TYPE_CHECKING:
    from ...config.parameters import SafeSynthesizerParameters

__all__ = ["load_plan", "save_config", "save_plan"]

# Unversioned documents are permanently interpreted as v3. A new schema must
# opt into a new output version without changing how existing documents load.
_IMPLICIT_PLAN_SCHEMA_VERSION = 3
_OUTPUT_PLAN_SCHEMA_VERSION = 3
_SUPPORTED_PLAN_SCHEMA_VERSIONS = frozenset({3})

_PLAN_FILE_HEADER = (
    "# PII replacement plan for this dataset. Edit it and pass it back through\n"
    "# replace_pii.replacement_plan to skip automatic discovery.\n\n"
)
# Extra guidance appended to a field's description in written YAML comments.
_PLAN_FIELD_NOTES = {
    "data_to_sampler_value_mapping": "You can usually leave this section as is.",
}
_COMMENT_WIDTH = 100
_YAML_KEY_LINE = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?: |$)")


def _plan_body(raw: dict[object, object], plan_path: Path) -> dict[object, object]:
    """Validate document metadata and return the unversioned runtime plan body."""
    body = dict(raw)
    version = body.pop("schema_version", _IMPLICIT_PLAN_SCHEMA_VERSION)
    if type(version) is not int:
        raise ParameterError(f"PII replacement plan file {str(plan_path)!r} schema_version must be an integer")
    if version not in _SUPPORTED_PLAN_SCHEMA_VERSIONS:
        supported = ", ".join(str(item) for item in sorted(_SUPPORTED_PLAN_SCHEMA_VERSIONS))
        raise ParameterError(
            f"PII replacement plan file {str(plan_path)!r} uses unsupported schema version {version}; "
            f"this NSS release supports version {supported}"
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

    body = _plan_body(raw, plan_path)
    try:
        normalized = normalize_unknown_fields(
            PiiReplacementPlan,
            cast(dict[str, object], body),
            "reject",
        )
        return PiiReplacementPlan.model_validate(normalized)
    except ValidationError as exc:
        details = format_pydantic_validation_error(exc)
        raise ParameterError(f"Invalid PII replacement plan in {str(plan_path)!r} ({details})") from exc
    except ParameterError as exc:
        raise ParameterError(f"Invalid PII replacement plan in {str(plan_path)!r} ({exc})") from exc


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
        "schema_version": _OUTPUT_PLAN_SCHEMA_VERSION,
        "columns_to_replace": columns,
        "data_to_sampler_value_mapping": plan.data_to_sampler_value_mapping,
    }


def save_plan(plan: PiiReplacementPlan, path: str | Path) -> Path:
    """Save a replacement plan as a versioned reusable standalone YAML document.

    Each plan section is preceded by a comment copied from its field description.
    """
    plan_path = Path(path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(_plan_document(plan), sort_keys=False)
    plan_path.write_text(_PLAN_FILE_HEADER + _with_key_comments(text, _plan_field_comments(())))
    return plan_path


def save_config(config: SafeSynthesizerParameters, path: str | Path) -> Path:
    """Save a complete NSS configuration with comments on its replacement plan sections.

    The output matches ``config.to_yaml(path, exclude_unset=False)`` apart from comments copied from the
    ``PiiReplacementPlan`` field descriptions above ``replace_pii.replacement_plan`` sections.
    """
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(json.loads(config.model_dump_json(exclude_unset=False)))
    config_path.write_text(_with_key_comments(text, _plan_field_comments(("replace_pii", "replacement_plan"))))
    return config_path


def _plan_field_comments(prefix: tuple[str, ...]) -> dict[tuple[str, ...], str]:
    """Return comment text for each described plan field, keyed by its YAML key path under ``prefix``."""
    comments: dict[tuple[str, ...], str] = {}
    for name, field in PiiReplacementPlan.model_fields.items():
        if field.description:
            note = _PLAN_FIELD_NOTES.get(name)
            comments[(*prefix, name)] = f"{field.description} {note}" if note else field.description
    return comments


def _with_key_comments(text: str, comments: Mapping[tuple[str, ...], str]) -> str:
    """Insert wrapped ``#`` comments above block-style YAML mapping keys whose path appears in ``comments``.

    Paths follow nested mapping keys by indentation; keys inside sequence items never match a path that does
    not pass through them, so only the intended sections receive comments.
    """
    lines: list[str] = []
    path: list[tuple[int, str]] = []
    for line in text.splitlines():
        if match := _YAML_KEY_LINE.match(line):
            indent = len(match["indent"])
            while path and path[-1][0] >= indent:
                path.pop()
            path.append((indent, match["key"]))
            comment = comments.get(tuple(key for _, key in path))
            if comment:
                width = max(_COMMENT_WIDTH - indent - 2, 40)
                lines.extend(f"{match['indent']}# {part}" for part in textwrap.wrap(comment, width))
        lines.append(line)
    return "\n".join(lines) + "\n"
