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
)
from ...config.time_series import TimeSeriesParameters
from ...errors import InternalError, ParameterError
from .patterns import CHARACTER_MASK_ESCAPABLE_CHARACTERS, CHARACTER_MASK_TOKENS, NAME_PART_PLACEHOLDERS

__all__ = ["get_protected_columns", "validate_plan"]

MIN_PATTERN_COVERAGE = 0.85
_NAME_PART_PATTERN = re.compile(r"\{([^{}]+)\}")


def get_protected_columns(
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

    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            if index + 1 >= len(pattern):
                return None, "ends with a trailing '\\'"
            index += 1
            escaped = pattern[index]
            if escaped not in CHARACTER_MASK_ESCAPABLE_CHARACTERS:
                allowed = " ".join(sorted(CHARACTER_MASK_ESCAPABLE_CHARACTERS))
                return None, f"escapes unsupported character {escaped!r}; only {allowed} may be escaped"
            parts.append(re.escape(pattern[index]))
        elif char in CHARACTER_MASK_TOKENS:
            parts.append(CHARACTER_MASK_TOKENS[char].regex)
            has_variable = True
        elif char == "[":
            choices, end, error = _character_class(pattern, index)
            if error is not None:
                return None, error
            assert choices is not None
            parts.append("[" + re.escape(choices) + "]")
            has_variable = True
            index = end
        else:
            parts.append(re.escape(char))
        index += 1

    if not has_variable:
        return None, "has no variable placeholder"
    return re.compile("".join(parts)), None


def _character_class(pattern: str, start: int) -> tuple[str | None, int, str | None]:
    """Parse one literal-choice class, honoring the same restricted escapes."""
    choices: list[str] = []
    index = start + 1
    while index < len(pattern):
        char = pattern[index]
        if char == "]":
            if not choices:
                return None, index, "has an empty '[]' character class"
            return "".join(choices), index, None
        if char == "\\":
            if index + 1 >= len(pattern):
                return None, index, "ends with a trailing '\\'"
            index += 1
            char = pattern[index]
            if char not in CHARACTER_MASK_ESCAPABLE_CHARACTERS:
                allowed = " ".join(sorted(CHARACTER_MASK_ESCAPABLE_CHARACTERS))
                return None, index, f"escapes unsupported character {char!r}; only {allowed} may be escaped"
        choices.append(char)
        index += 1
    return None, index, "has an unclosed '[' character class"


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
        placeholder = NAME_PART_PLACEHOLDERS.get(match.group(0))
        if placeholder is None:
            return None, f"uses unknown placeholder {match.group(0)!r}"
        if placeholder.part == "domain" and entity_type is not EntityType.EMAIL:
            return None, "uses {domain} outside an email pattern"
        if placeholder.part == "domain":
            parts.append(r"[^@\s]+")
        elif placeholder.initial:
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
                raise InternalError(
                    f"Pattern syntax {pattern_syntax!r} for entity_type {spec.entity_type.value!r} has no matcher"
                )
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
    protected_columns: frozenset[str],
) -> Iterator[str]:
    dataframe_columns = set(df.columns)
    for spec in plan.columns_to_replace:
        if spec.column_name not in dataframe_columns:
            yield f"replacement column {spec.column_name!r} is not present in the dataframe"
        if spec.column_name in protected_columns:
            yield f"protected column {spec.column_name!r} cannot be replaced"
        for dependency in spec.depends_on:
            if dependency.column_name not in dataframe_columns:
                yield (
                    f"column {spec.column_name!r}: depends_on column "
                    f"{dependency.column_name!r} is not present in the dataframe"
                )


def validate_plan(
    df: pd.DataFrame,
    plan: PiiReplacementPlan,
    *,
    data_config: DataParameters,
    time_series: TimeSeriesParameters | None = None,
) -> None:
    """Validate the selected final plan against the dataframe and data configuration.

    This is the resolver's single dataframe-aware validation gate. Pydantic
    model construction and LLM response parsing are separate, context-free
    checks performed at their respective seams.
    """
    issues: list[str] = []
    group_column = data_config.group_training_examples_by
    # Replacement consistency is derived from data configuration rather than
    # stored in the reusable plan: configured groups use group consistency;
    # otherwise every record is replaced independently.
    if group_column is not None and group_column not in df.columns:
        issues.append(f"group column {group_column!r} is not present in the dataframe")

    issues.extend(_iter_reference_issues(df, plan, get_protected_columns(data_config, time_series)))
    issues.extend(_iter_pattern_issues(df, plan))

    if issues:
        details = "\n".join(f"  - {issue}" for issue in issues)
        raise ParameterError(f"Invalid PII replacement plan for this dataframe:\n{details}")
