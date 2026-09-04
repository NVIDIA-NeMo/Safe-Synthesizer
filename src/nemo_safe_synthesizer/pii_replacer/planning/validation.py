# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataframe-aware validation for final PII replacement plans."""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import datetime, timezone

import pandas as pd

from ...config.data import DataParameters
from ...config.replace_pii import (
    ENTITY_BY_TYPE,
    EntityType,
    PatternSyntax,
    PiiReplacementPlan,
    PiiReplacementScope,
)
from ...config.time_series import TimeSeriesParameters
from ...errors import ParameterError

__all__ = ["protected_columns", "validate_plan"]

MIN_PATTERN_COVERAGE = 0.85
_NAME_PART_PATTERN = re.compile(r"\{([^{}]+)\}")
_NAME_PART_TOKENS = frozenset({"first", "middle", "last", "domain", "f", "m", "l"})


def protected_columns(
    data_config: DataParameters,
    time_series: TimeSeriesParameters | None = None,
) -> frozenset[str]:
    """Return ordering columns that automatic replacement must preserve."""
    candidates = {
        data_config.order_training_examples_by,
        time_series.timestamp_column if time_series is not None else None,
    }
    return frozenset(column for column in candidates if column is not None)


def _template_regex(pattern: str) -> tuple[re.Pattern[str] | None, str | None]:
    parts: list[str] = []
    has_variable = False
    index = 0
    token_regex = {
        "#": r"\d",
        "^": "[A-Z]",
        "@": "[a-z]",
        "&": "[A-Z0-9]",
        "%": "[a-z0-9]",
        "*": "[A-Za-z0-9]",
    }

    while index < len(pattern):
        char = pattern[index]
        if char == "\\" and index + 1 < len(pattern):
            index += 1
            parts.append(re.escape(pattern[index]))
        elif char in token_regex:
            parts.append(token_regex[char])
            has_variable = True
        elif char == "[":
            end = pattern.find("]", index + 1)
            if end < 0:
                return None, "has an unclosed '[' character class"
            choices = pattern[index + 1 : end]
            if not choices:
                return None, "has an empty '[]' character class"
            parts.append("[" + re.escape(choices) + "]")
            has_variable = True
            index = end
        else:
            parts.append(re.escape(char))
        index += 1

    if not has_variable:
        return None, "has no variable placeholder"
    return re.compile("".join(parts)), None


def _name_parts_regex(
    entity_type: EntityType,
    pattern: str,
) -> tuple[re.Pattern[str] | None, str | None]:
    matches = list(_NAME_PART_PATTERN.finditer(pattern))
    if not matches:
        return None, "has no name-part placeholder"
    if "{" in _NAME_PART_PATTERN.sub("", pattern) or "}" in _NAME_PART_PATTERN.sub("", pattern):
        return None, "has an unmatched '{' or '}'"

    parts: list[str] = []
    cursor = 0
    for match in matches:
        literal = pattern[cursor : match.start()]
        parts.append(_name_parts_literal_regex(literal, entity_type))
        token = match.group(1).lower()
        if token not in _NAME_PART_TOKENS:
            return None, f"uses unknown placeholder {match.group(0)!r}"
        if token == "domain" and entity_type is not EntityType.EMAIL:
            return None, "uses {domain} outside an email pattern"
        if token == "domain":
            parts.append(r"[^@\s]+")
        elif token in {"f", "m", "l"}:
            parts.append(r"[^\W\d_]")
        elif entity_type is EntityType.EMAIL:
            parts.append(r"[^@\s.]+")
        else:
            parts.append(r"[^@\s]+")
        cursor = match.end()
    parts.append(_name_parts_literal_regex(pattern[cursor:], entity_type))

    if entity_type is EntityType.EMAIL and "@" not in pattern:
        return None, "does not contain '@'"
    return re.compile("".join(parts), re.UNICODE), None


def _name_parts_literal_regex(literal: str, entity_type: EntityType) -> str:
    if entity_type is not EntityType.EMAIL:
        return re.escape(literal)
    return "".join(r"\d" if character == "#" else re.escape(character) for character in literal)


def _strftime_pattern_error(pattern: str) -> str | None:
    try:
        # Use an aware probe so timezone directives such as ``%z`` and ``%Z``
        # render values that can be parsed back with the same format.
        rendered = datetime(2001, 2, 3, 4, 5, 6, tzinfo=timezone.utc).strftime(pattern)
        datetime.strptime(rendered, pattern)
    except ValueError as exc:
        return str(exc)
    return None


def _pattern_matcher(
    entity_type: EntityType,
    pattern: str,
) -> tuple[re.Pattern[str] | None, str | None]:
    pattern_syntax = ENTITY_BY_TYPE[entity_type].pattern_syntax
    if pattern_syntax is PatternSyntax.CHARACTER_MASK:
        return _template_regex(pattern)
    if pattern_syntax is PatternSyntax.NAME_PARTS:
        return _name_parts_regex(entity_type, pattern)
    return None, None


def _iter_pattern_issues(df: pd.DataFrame, plan: PiiReplacementPlan) -> Iterator[str]:
    for spec in plan.columns_to_replace:
        pattern = spec.pattern
        if pattern is None or spec.column_name not in df.columns:
            continue

        values = df[spec.column_name].dropna().astype(str).tolist()
        pattern_syntax = ENTITY_BY_TYPE[spec.entity_type].pattern_syntax
        if pattern_syntax is PatternSyntax.STRFTIME:
            if error := _strftime_pattern_error(pattern):
                yield f"column {spec.column_name!r}: pattern {pattern!r} is not valid strftime ({error})"
                continue
            matches = sum(_parses_datetime(value, pattern) for value in values)
        else:
            matcher, error = _pattern_matcher(spec.entity_type, pattern)
            if error is not None:
                yield f"column {spec.column_name!r}: pattern {pattern!r} {error}"
                continue
            if matcher is None:
                continue
            matches = sum(matcher.fullmatch(value) is not None for value in values)

        if values and matches / len(values) < MIN_PATTERN_COVERAGE:
            coverage = matches / len(values)
            yield (
                f"column {spec.column_name!r}: pattern {pattern!r} covers {coverage:.1%} of non-null values; "
                f"at least {MIN_PATTERN_COVERAGE:.0%} is required"
            )


def _parses_datetime(value: str, pattern: str) -> bool:
    try:
        datetime.strptime(value, pattern)
    except ValueError:
        return False
    return True


def _iter_reference_issues(
    df: pd.DataFrame,
    plan: PiiReplacementPlan,
    structural_columns: frozenset[str],
) -> Iterator[str]:
    dataframe_columns = set(df.columns)
    for spec in plan.columns_to_replace:
        if spec.column_name not in dataframe_columns:
            yield f"replacement column {spec.column_name!r} is not present in the dataframe"
        if spec.column_name in structural_columns:
            yield f"structural column {spec.column_name!r} cannot be replaced"
        for dependency in spec.depends_on:
            if dependency.column_name not in dataframe_columns:
                yield (
                    f"column {spec.column_name!r}: depends_on column "
                    f"{dependency.column_name!r} is not present in the dataframe"
                )
            if dependency.column_name == spec.column_name:
                yield f"column {spec.column_name!r} cannot depend on itself"


def _cycle_columns(plan: PiiReplacementPlan) -> list[str]:
    """Return replacement columns involved in a dependency cycle.

    The current allowed dependency matrix is acyclic, so normally validated
    plans cannot contain a cycle. Keep this dataframe-aware gate as a defensive
    check in case that matrix evolves or model validation is bypassed.
    """
    targets = {spec.column_name for spec in plan.columns_to_replace}
    adjacency: dict[str, set[str]] = {column: set() for column in targets}
    indegree = dict.fromkeys(targets, 0)

    for spec in plan.columns_to_replace:
        for dependency in spec.depends_on:
            source = dependency.column_name
            if source not in targets or source == spec.column_name or spec.column_name in adjacency[source]:
                continue
            adjacency[source].add(spec.column_name)
            indegree[spec.column_name] += 1

    ready = [column for column, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        source = ready.pop()
        visited += 1
        for target in adjacency[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)

    if visited == len(targets):
        return []
    return sorted(column for column, degree in indegree.items() if degree > 0)


def validate_plan(
    df: pd.DataFrame,
    plan: PiiReplacementPlan,
    *,
    data_config: DataParameters,
    time_series: TimeSeriesParameters | None = None,
) -> None:
    """Validate the selected final plan against the dataframe and structural config.

    This is the resolver's single dataframe-aware validation gate. Pydantic
    model construction and LLM response parsing are separate, context-free
    checks performed at their respective seams.
    """
    issues: list[str] = []
    group_column = data_config.group_training_examples_by
    if plan.scope is PiiReplacementScope.GROUP:
        if group_column is None:
            issues.append("plan scope is 'group' but data.group_training_examples_by is not configured")
        elif group_column not in df.columns:
            issues.append(f"group column {group_column!r} is not present in the dataframe")

    issues.extend(_iter_reference_issues(df, plan, protected_columns(data_config, time_series)))
    if cycles := _cycle_columns(plan):
        issues.append("replacement dependencies contain a cycle involving: " + ", ".join(repr(name) for name in cycles))
    issues.extend(_iter_pattern_issues(df, plan))

    if issues:
        details = "\n".join(f"  - {issue}" for issue in issues)
        raise ParameterError(f"Invalid PII replacement plan for this dataframe:\n{details}")
