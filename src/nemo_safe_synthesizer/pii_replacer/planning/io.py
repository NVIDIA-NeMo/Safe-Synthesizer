# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Save PII replacement plans and configurations with comments for review."""

from __future__ import annotations

import json
import re
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from ...config.parameters import SafeSynthesizerParameters
    from ...config.replace_pii import PiiReplacementPlan, PiiSamplerConfig

__all__ = ["save_config", "save_plan"]

# Comments written above plan sections, for a person reviewing or editing the plan.
_PLAN_SECTION_COMMENTS = {
    "columns_to_replace": (
        "Columns whose values are replaced. pattern sets the format of the new values, such as pmc-###### "
        "for an ID or %m/%d/%Y for a date. depends_on makes a new value fit other columns in the same row, "
        "such as a first name that matches the row's sex, or an email built from the row's new name."
    ),
    "data_to_sampler_value_mapping": (
        "THE NEXT SECTION IS SAFE TO IGNORE unless a run reports an error about it. It translates values in "
        "depends_on columns, such as Latino in a race column, to values the sampler understands. "
        "Safe Synthesizer fills it in, and it can get long."
    ),
}
# The published PII replacement guide, linked from the plan file header.
_PLAN_DOCS_URL = "https://nvidia-nemo.github.io/Safe-Synthesizer/latest/product-overview/pii_replacement/"
_COMMENT_WIDTH = 100
_YAML_KEY_LINE = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?: |$)")


def save_plan(plan: PiiReplacementPlan, path: str | Path, *, sampler: PiiSamplerConfig, locale: str) -> Path:
    """Save ``replace_pii.replacement_plan`` as a standalone YAML file for review and reuse.

    The file starts with a header naming the sampler and locale the plan was
    written for, and each section is preceded by a short comment. The header
    only informs the reader: each run checks the plan against its own
    ``replace_pii.sampler`` and ``replace_pii.locale``.

    Args:
        plan: Replacement plan to save.
        path: Destination YAML file; parent directories are created.
        sampler: Sampler the plan was written for.
        locale: Locale the plan was written for.

    Returns:
        The path written.
    """
    plan_path = Path(path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# PII replacement plan for this dataset, written for the {sampler.backend.value} sampler in locale "
        f"{locale}.\n# Review and edit it, then name this file in replace_pii.replacement_plan to reuse it.\n"
        f"# Plan format: {_PLAN_DOCS_URL}#discovery\n"
        f"# Patterns: {_PLAN_DOCS_URL}#patterns\n\n"
    )
    text = yaml.safe_dump(_plan_document(plan), sort_keys=False)
    plan_path.write_text(header + _with_key_comments(text, _plan_section_comments(())))
    return plan_path


def save_config(
    config: SafeSynthesizerParameters,
    path: str | Path,
    *,
    plan_path: str | Path | None = None,
) -> Path:
    """Save a complete NSS configuration.

    With ``plan_path`` and an inline plan, ``replace_pii.replacement_plan`` is
    written to that file by ``save_plan``, and the configuration names it by a
    path relative to the configuration's directory when the file is inside it,
    or by its absolute path otherwise, so loading the configuration reads the
    file back. Otherwise the plan is written inline, and the output matches
    ``config.to_yaml(path, exclude_unset=False)`` apart from the same plan
    comments.

    Args:
        config: Configuration to save.
        path: Destination YAML file; parent directories are created.
        plan_path: Separate file for ``replace_pii.replacement_plan``.

    Returns:
        The path written.
    """
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    document = json.loads(config.model_dump_json(exclude_unset=False))
    comments: dict[tuple[str, ...], list[str]] = {}
    replace_pii = config.replace_pii
    plan = replace_pii.inline_plan if replace_pii is not None else None
    if replace_pii is not None and plan is not None and plan_path is not None:
        written = save_plan(plan, plan_path, sampler=replace_pii.sampler, locale=replace_pii.locale)
        document["replace_pii"]["replacement_plan"] = _path_from(config_path.parent, written)
    else:
        comments = _plan_section_comments(("replace_pii", "replacement_plan"))
    config_path.write_text(_with_key_comments(yaml.safe_dump(document), comments))
    return config_path


def _path_from(directory: Path, path: Path) -> str:
    """Return ``path`` relative to ``directory`` when it is inside it, else absolute, in POSIX form."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(directory.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _plan_document(plan: PiiReplacementPlan) -> dict[str, object]:
    """Return the plan as YAML data in field order, with sparse column specs that omit inferred types."""
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


def _plan_section_comments(prefix: tuple[str, ...]) -> dict[tuple[str, ...], list[str]]:
    """Return the comment for each plan section, keyed by its YAML key path under ``prefix``."""
    return {(*prefix, name): [comment] for name, comment in _PLAN_SECTION_COMMENTS.items()}


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
