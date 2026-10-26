# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Save PII replacement settings and configurations with comments for review."""

from __future__ import annotations

import json
import os
import re
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ...config.replace_pii import PiiReplacementPlan, PiiReplacementSettings

if TYPE_CHECKING:
    from ...config.parameters import SafeSynthesizerParameters
    from .review import PlanReviewNotes

__all__ = ["save_config", "save_replacement"]

_REPLACEMENT_FILE_HEADER = (
    "# PII replacement for this dataset: the plan, and the locale, seed, and sampler it depends on.\n"
    "# Review and edit it, then name this file in replace_pii.replacement to reuse it.\n\n"
)
# Extra guidance appended to a field's description in written YAML comments.
_PLAN_FIELD_NOTES = {
    "data_to_sampler_value_mapping": "You can usually leave this section as is.",
}
_COMMENT_WIDTH = 100
# Review notes list at most this many dataset values or columns, then a total.
_MAX_LISTED_VALUES = 20
_MAX_LISTED_COLUMNS = 50
_YAML_KEY_LINE = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?: |$)")


def save_replacement(
    settings: PiiReplacementSettings, path: str | Path, *, review_notes: PlanReviewNotes | None = None
) -> Path:
    """Save ``replace_pii.replacement`` as a standalone YAML file for review and reuse.

    Fields keep their model order, so the plan follows the locale, seed, and
    sampler it depends on. Each plan section is preceded by a comment copied
    from its field description, then by ``review_notes`` when given.

    Args:
        settings: Replacement settings to save.
        path: Destination YAML file; parent directories are created.
        review_notes: Dataset context to add as comments for a person reviewing a drafted plan.

    Returns:
        The path written.
    """
    replacement_path = Path(path)
    replacement_path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(_replacement_document(settings), sort_keys=False)
    comments = _plan_field_comments(("plan",), review_notes)
    replacement_path.write_text(_REPLACEMENT_FILE_HEADER + _with_key_comments(text, comments))
    return replacement_path


def save_config(
    config: SafeSynthesizerParameters,
    path: str | Path,
    *,
    review_notes: PlanReviewNotes | None = None,
    replacement_path: str | Path | None = None,
) -> Path:
    """Save a complete NSS configuration.

    With ``replacement_path``, ``replace_pii.replacement`` is written to that
    file by ``save_replacement``, and the configuration names it by a path
    relative to the configuration's directory, so loading the configuration
    reads the file back. Otherwise the section is written inline, and the output
    matches ``config.to_yaml(path, exclude_unset=False)`` apart from the same
    plan comments.

    Args:
        config: Configuration to save.
        path: Destination YAML file; parent directories are created.
        review_notes: Dataset context to add as comments for a person reviewing a drafted plan.
        replacement_path: Separate file for ``replace_pii.replacement``.

    Returns:
        The path written.
    """
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    document = json.loads(config.model_dump_json(exclude_unset=False))
    comments: dict[tuple[str, ...], list[str]] = {}
    if replacement_path is not None and config.replace_pii is not None:
        written = save_replacement(config.replace_pii.replacement, replacement_path, review_notes=review_notes)
        document["replace_pii"]["replacement"] = Path(os.path.relpath(written, config_path.parent)).as_posix()
    else:
        comments = _plan_field_comments(("replace_pii", "replacement", "plan"), review_notes)
    config_path.write_text(_with_key_comments(yaml.safe_dump(document), comments))
    return config_path


def _replacement_document(settings: PiiReplacementSettings) -> dict[str, object]:
    """Return the settings as YAML data with a sparse plan, in field order."""
    document = settings.model_dump(mode="json", exclude_unset=False)
    if (plan := settings.inline_plan) is not None:
        document["plan"] = _plan_document(plan)
    return document


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
                if dependency.entity_type is not None:
                    serialized_dependency["entity_type"] = dependency.entity_type.value
                dependencies.append(serialized_dependency)
            serialized_spec["depends_on"] = dependencies
        columns.append(serialized_spec)
    return {
        "columns_to_replace": columns,
        "data_to_sampler_value_mapping": plan.data_to_sampler_value_mapping,
    }


def _plan_field_comments(
    prefix: tuple[str, ...], review_notes: PlanReviewNotes | None = None
) -> dict[tuple[str, ...], list[str]]:
    """Return comment paragraphs for each described plan field, keyed by its YAML key path under ``prefix``."""
    comments: dict[tuple[str, ...], list[str]] = {}
    for name, field in PiiReplacementPlan.model_fields.items():
        if field.description:
            note = _PLAN_FIELD_NOTES.get(name)
            comments[(*prefix, name)] = [f"{field.description} {note}" if note else field.description]
    if review_notes is not None:
        for name, paragraphs in _review_comments(review_notes).items():
            comments.setdefault((*prefix, name), []).extend(paragraphs)
    return comments


def _review_comments(notes: PlanReviewNotes) -> dict[str, list[str]]:
    """Return review paragraphs for the ``columns_to_replace`` and ``data_to_sampler_value_mapping`` sections.

    For example, a plan that leaves ``admission_id`` unchanged and conditions on ``sex`` gives
    ``Draft review: not in this plan, so kept unchanged: admission_id. ...`` and
    ``Draft review: sex (gender): Female, Male are used as is. Sampler values: female, male.``
    """
    unchanged = (
        f"not in this plan, so kept unchanged: {_listed(notes.unchanged_columns, _MAX_LISTED_COLUMNS)}"
        if notes.unchanged_columns
        else "every dataset column is in this plan"
    )
    comments = {
        "columns_to_replace": [
            f"Draft review: {unchanged}. Review notes describe the plan as drafted and are not updated "
            "when you edit it."
        ]
    }
    if notes.dependency_values:
        mapping_notes: list[str] = []
        for review in notes.dependency_values:
            if review.used_as_is:
                verb = "is" if len(review.used_as_is) == 1 else "are"
                used = f"{_listed(review.used_as_is, _MAX_LISTED_VALUES)} {verb} used as is"
            else:
                used = "no value is used as is"
            mapping_notes.append(
                f"Draft review: {review.column_name} ({review.entity_type.value}): {used}. "
                f"Sampler values: {_listed(review.sampler_values, _MAX_LISTED_VALUES)}."
            )
        comments["data_to_sampler_value_mapping"] = mapping_notes
    return comments


def _listed(values: Sequence[str], limit: int) -> str:
    """Join up to ``limit`` values with commas, noting the total when some are left out."""
    if len(values) <= limit:
        return ", ".join(values)
    return f"{', '.join(values[:limit])}, ... ({len(values)} in total)"


def _with_key_comments(text: str, comments: Mapping[tuple[str, ...], Sequence[str]]) -> str:
    """Insert wrapped ``#`` comments above block-style YAML mapping keys whose path appears in ``comments``.

    Each paragraph is wrapped separately. Paths follow nested mapping keys by indentation; keys inside sequence
    items never match a path that does not pass through them, so only the intended sections receive comments.
    """
    lines: list[str] = []
    path: list[tuple[int, str]] = []
    for line in text.splitlines():
        if match := _YAML_KEY_LINE.match(line):
            indent = len(match["indent"])
            while path and path[-1][0] >= indent:
                path.pop()
            path.append((indent, match["key"]))
            width = max(_COMMENT_WIDTH - indent - 2, 40)
            for paragraph in comments.get(tuple(key for _, key in path), ()):
                lines.extend(f"{match['indent']}# {part}" for part in textwrap.wrap(paragraph, width))
        lines.append(line)
    return "\n".join(lines) + "\n"
