# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan validation: protected columns, issues, and advisories."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd

from ...config.data import DataParameters
from ...config.replace_pii import (
    PiiColumnPlan,
    PiiEntity,
    PiiReplacementPlan,
    PiiReplacementScope,
)
from ...config.time_series import TimeSeriesParameters
from ...errors import ParameterError
from .. import entities, patterns
from ..entity_handlers import get_handler


def _training_group_key(data_config: DataParameters | None) -> str | None:
    return data_config.group_training_examples_by if data_config else None


def protected_columns(
    data_config: DataParameters | None,
    *,
    time_series: TimeSeriesParameters | None = None,
) -> set[str]:
    """Return columns that must not be PII-replaced.

    In time-series mode, the training group key is left unchanged so sequences stay
    grouped. ``data.order_training_examples_by``, when set, is left unchanged so
    row order stays intact. In time-series mode, ``time_series.timestamp_column`` is
    also protected (it becomes the order-by column during timeseries preprocessing).

    Args:
        data_config: Data parameters with group and order keys; ``None`` yields an
            empty set.
        time_series: Optional time-series parameters for timestamp protection.

    Returns:
        Set of column names that define training structure and must not be replaced.
    """
    protected: set[str] = set()
    if data_config is None:
        return protected
    is_ts = bool(time_series and time_series.is_timeseries)
    if is_ts and data_config.group_training_examples_by:
        protected.add(data_config.group_training_examples_by)
    if data_config.order_training_examples_by:
        protected.add(data_config.order_training_examples_by)
    if is_ts and time_series and time_series.timestamp_column:
        protected.add(time_series.timestamp_column)
    return protected


def strip_protected_columns_from_plan(
    plan: PiiReplacementPlan,
    columns: set[str],
) -> list[str]:
    """Remove protected columns from the plan in place.

    Args:
        plan: Replacement plan to mutate.
        columns: Protected column names to strip from ``columns_to_replace``.

    Returns:
        Column names that were removed, in first-seen order and deduplicated.
    """
    if not columns:
        return []
    removed: list[str] = []
    for col_set in plan.persona_backed_columns:
        kept = []
        for spec in col_set.columns_to_replace:
            if spec.column_name in columns:
                removed.append(spec.column_name)
            else:
                kept.append(spec)
        col_set.columns_to_replace = kept
    kept_standalone = []
    for spec in plan.standalone_columns_to_replace:
        if spec.column_name in columns:
            removed.append(spec.column_name)
        else:
            kept_standalone.append(spec)
    plan.standalone_columns_to_replace = kept_standalone
    # Drop persona sets that only had protected columns and no remaining matchers-with-cols.
    plan.persona_backed_columns = [p for p in plan.persona_backed_columns if p.columns_to_replace]
    # Preserve order, unique
    seen: set[str] = set()
    ordered: list[str] = []
    for col in removed:
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return ordered


def _column_specs(plan: PiiReplacementPlan) -> list[tuple[str, PiiColumnPlan]]:
    """Every column spec in the plan, each with its location for error messages."""
    specs: list[tuple[str, PiiColumnPlan]] = []
    for col_set in plan.persona_backed_columns:
        specs.extend((f"persona {col_set.persona!r}", spec) for spec in col_set.columns_to_replace)
    specs.extend(("standalone_columns_to_replace", spec) for spec in plan.standalone_columns_to_replace)
    return specs


def _replacement_columns(plan: PiiReplacementPlan) -> list[str]:
    return [spec.column_name for _, spec in _column_specs(plan)]


def _protected_replacement_columns(plan: PiiReplacementPlan, protected: set[str]) -> list[str]:
    """Protected columns listed under columns_to_replace, in plan order (unique)."""
    if not protected:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for col in _replacement_columns(plan):
        if col in protected and col not in seen:
            seen.add(col)
            ordered.append(col)
    return ordered


def _iter_protected_column_issues(
    plan: PiiReplacementPlan,
    *,
    data_config: DataParameters | None,
    time_series: TimeSeriesParameters | None,
) -> Iterator[PlanIssue]:
    """Reject user plans that ask to replace structural training columns."""
    protected = protected_columns(data_config, time_series=time_series)
    listed = _protected_replacement_columns(plan, protected)
    if not listed:
        return
    yield PlanIssue(
        "pii_plan_protected_column",
        "replacement plan must not replace structural columns "
        + ", ".join(repr(c) for c in listed)
        + ". Time-series group keys, data.order_training_examples_by, and the "
        "time-series timestamp column define training structure and cannot be "
        "PII-replaced; remove them from columns_to_replace "
        "(match_persona_by may still reference them).",
    )


# ``patterns`` carries a different language per entity: birth dates use strftime
# formats to parse and re-format each value, while identifiers and phone numbers use
# value templates such as ``pmc-[68]######-[1234]`` expanded by
# ``patterns.generate_from_pattern``. Every other entity takes its value from the
# synthetic persona and reads its patterns as the parts of a person.
# Pattern families are derived from ``EntitySpec.pattern_kind`` (see entities.py).
DATE_PATTERN_ENTITIES = frozenset(
    PiiEntity(s.label)
    for s in entities.ENTITY_REGISTRY.values()
    if s.pattern_kind == "strftime" and s.label in PiiEntity.__members__
)
TEMPLATE_PATTERN_ENTITIES = frozenset(
    PiiEntity(s.label)
    for s in entities.ENTITY_REGISTRY.values()
    if s.pattern_kind == "template" and s.label in PiiEntity.__members__
)
PERSONA_PATTERN_ENTITIES = frozenset(
    PiiEntity(s.label)
    for s in entities.ENTITY_REGISTRY.values()
    if s.pattern_kind == "persona_placeholder" and s.label in PiiEntity.__members__
)
# Every entity that reads its ``patterns`` list. A plan that sets patterns on
# anything else is refused, since the replacement would ignore them.
PATTERNED_ENTITIES = DATE_PATTERN_ENTITIES | TEMPLATE_PATTERN_ENTITIES | PERSONA_PATTERN_ENTITIES
# How to say what a persona pattern may name, since the parts are a closed set.
_PERSONA_PARTS_HINT = (
    "the parts are '{first}', '{middle}' and '{last}', written in the case you want "
    "('{First}', '{LAST}') or as an initial ('{f}')"
)

# Patterns are checked against the same evidence slice discovery used to infer
# them (``patterns.pattern_evidence_values``), so a secondary format found in the
# seeded row sample cannot fail validation that only saw first-seen distincts.


def _distinct_values(df: pd.DataFrame, col: str) -> list[str]:
    """Values used when checking whether a plan pattern describes the column."""
    return patterns.pattern_evidence_values(df[col])


def _parses_as_date(value: str, pattern: str) -> bool:
    try:
        datetime.strptime(value, pattern)
    except (ValueError, TypeError):
        return False
    return True


@dataclass(frozen=True)
class PlanIssue:
    """A problem or advisory found while checking a plan against a dataframe.

    ``code`` is the stable identifier the preflight check emits; codes are listed
    in ``docs/user-guide/troubleshooting.md``. Errors make the plan unusable;
    warnings flag section placement that the engine still honors by entity type
    rather than by YAML section.
    """

    code: str
    """Stable issue identifier for logs and troubleshooting docs."""
    message: str
    """Human-readable explanation of the problem."""
    severity: Literal["error", "warning"] = "error"
    """``error`` blocks the plan; ``warning`` is advisory only."""


def _iter_date_pattern_issues(col: str, pattern: str, samples: list[str]) -> Iterator[PlanIssue]:
    """Report a date format none of the column's own values parse with.

    A format that parses no value describes nothing the column writes, so it
    would silently do nothing and each value would be re-formatted in its own.
    """
    if not samples or any(_parses_as_date(value, pattern) for value in samples):
        return
    yield PlanIssue(
        "pii_plan_pattern_invalid",
        f"pattern {pattern!r} for column {col!r} does not parse any of its values "
        f"(e.g. {samples[0]!r}); expected a strftime format such as "
        f"{patterns.detect_date_format(samples[0])!r}",
    )


def _iter_unused_template_issues(col: str, pattern: str, samples: list[str]) -> Iterator[PlanIssue]:
    """Report a template that describes none of the column's values.

    A value is written in the first listed template that describes it, so one
    that describes nothing is never reached and the column is replaced as though
    it had never been listed.
    """
    if not samples or any(patterns.value_matches_template(value, pattern) for value in samples):
        return
    yield PlanIssue(
        "pii_plan_pattern_invalid",
        f"pattern {pattern!r} for column {col!r} matches none of its values (e.g. {samples[0]!r}), so it "
        f"would never be used; a value no pattern describes is replaced in its own shape. Write the "
        f"template the values wear, such as {patterns.value_shape_template(samples[0])!r}.",
    )


def _iter_persona_pattern_issues(col: str, entity: PiiEntity, pattern: str) -> Iterator[PlanIssue]:
    """Report a name or address pattern the persona cannot be written into.

    These patterns name the parts of a person rather than characters, so what
    goes wrong is a part that does not exist and a pattern naming none at all,
    which would leave every row of the column reading the same.
    """
    local = pattern
    if entity == PiiEntity.email:
        local, domain = patterns.split_email(pattern)
        if not domain:
            yield PlanIssue(
                "pii_plan_pattern_invalid",
                f"pattern {pattern!r} for column {col!r} has no '@' and its domain; an address pattern says "
                f"what goes before the '@' and ends {patterns.EMAIL_DOMAIN_ONLY_PATTERN!r}, which keeps the "
                "domain each value already had",
            )
            return
        if patterns.DOMAIN_PLACEHOLDER in local:
            yield PlanIssue(
                "pii_plan_pattern_invalid",
                f"pattern {pattern!r} for column {col!r} writes {patterns.DOMAIN_PLACEHOLDER!r} before the '@', "
                "where only the parts of a person belong",
            )
            return
        # A column with no convention before the '@' writes '@{domain}': the
        # domain is kept and each handle keeps the shape it has.
        if not local:
            return
        if not patterns.placeholder_tokens(local):
            # A local part naming no part of a person describes a handle, which
            # has a shape rather than a convention and is read as one.
            if patterns.value_template_is_constant(local):
                yield PlanIssue(
                    "pii_plan_pattern_invalid",
                    f"pattern {pattern!r} for column {col!r} has no variable position before the '@', so "
                    f"every row would read the same; name the parts of a person, where {_PERSONA_PARTS_HINT}, "
                    "or write the shape a handle has, such as 'usr####'",
                )
            return

    tokens = patterns.placeholder_tokens(local)
    unknown = [token for token in tokens if "{" + token + "}" not in patterns.PERSONA_PLACEHOLDERS]
    if unknown:
        yield PlanIssue(
            "pii_plan_pattern_invalid",
            f"pattern {pattern!r} for column {col!r} names no part of a person in '{{{unknown[0]}}}'; "
            f"{_PERSONA_PARTS_HINT}",
        )
    elif not tokens:
        yield PlanIssue(
            "pii_plan_pattern_invalid",
            f"pattern {pattern!r} for column {col!r} names no part of a person, so every row would read the "
            f"same; {_PERSONA_PARTS_HINT}",
        )


def _iter_scope_issues(
    plan: PiiReplacementPlan, df_cols: set[str], data_config: DataParameters | None
) -> Iterator[PlanIssue]:
    if plan.scope != PiiReplacementScope.group:
        return
    group_key = _training_group_key(data_config)
    if not group_key:
        yield PlanIssue(
            "pii_plan_group_scope_invalid",
            "plan.scope is 'group' but data.group_training_examples_by is not set",
        )
    elif group_key not in df_cols:
        yield PlanIssue(
            "pii_plan_group_scope_invalid",
            f"data.group_training_examples_by column {group_key!r} not found in dataframe",
        )


def _iter_column_reference_issues(plan: PiiReplacementPlan, df_cols: set[str]) -> Iterator[PlanIssue]:
    seen: set[str] = set()
    for col in _replacement_columns(plan):
        if col in seen:
            yield PlanIssue("pii_plan_duplicate_entry", f"column {col!r} appears more than once in replacement plan")
            continue
        seen.add(col)
        if col not in df_cols:
            yield PlanIssue("pii_plan_column_not_found", f"replacement plan references missing column {col!r}")


def _iter_column_spec_issues(df: pd.DataFrame, plan: PiiReplacementPlan, df_cols: set[str]) -> Iterator[PlanIssue]:
    """Report column specs that are internally inconsistent or unusable on ``df``."""
    matcher_columns = {cond.column_name for col_set in plan.persona_backed_columns for cond in col_set.match_persona_by}
    for location, spec in _column_specs(plan):
        col = spec.column_name
        if spec.entity_type is None:
            yield PlanIssue(
                "pii_plan_entity_type_invalid",
                f"column {col!r} in {location} has no entity_type, so nothing would be replaced",
            )
            continue
        if spec.entity_type == PiiEntity.date:
            yield PlanIssue(
                "pii_plan_entity_type_invalid",
                f"column {col!r} in {location} uses entity_type 'date'; generic dates are only identified, "
                "never replaced. Use 'date_of_birth' for birth dates, or drop the column from the plan.",
            )
            continue
        if col in matcher_columns:
            yield PlanIssue(
                "pii_plan_column_conflict",
                f"column {col!r} is listed both in match_persona_by and in {location}; match_persona_by "
                "columns are read to pick the persona and are never replaced",
            )
        if not spec.patterns:
            continue
        pattern_rejection = get_handler(spec.entity_type.value).plan_pattern_rejection(col)
        if pattern_rejection is not None:
            yield PlanIssue(
                "pii_plan_pattern_invalid",
                pattern_rejection,
            )
            continue
        if spec.entity_type == PiiEntity.free_text:
            yield PlanIssue(
                "pii_plan_pattern_invalid",
                f"column {col!r} sets patterns, but free_text columns are rewritten by propagating "
                "replaced values into the text, which ignores patterns",
            )
            continue
        if spec.entity_type not in PATTERNED_ENTITIES:
            yield PlanIssue(
                "pii_plan_pattern_invalid",
                f"column {col!r} sets patterns, but a {spec.entity_type.value} value is taken from the "
                "synthetic person (or rebuilt from the original) and does not follow a listed format; "
                "drop the patterns",
            )
            continue
        # Sampling reads the column, so an absent one is left to the
        # missing-column issue rather than reported twice.
        samples = _distinct_values(df, col) if col in df_cols else []
        seen_patterns: set[str] = set()
        for pattern in spec.patterns:
            if not pattern.strip():
                yield PlanIssue(
                    "pii_plan_pattern_invalid",
                    f"column {col!r} lists an empty pattern; every entry has to say how a value is written, "
                    "and a column that writes no format of its own simply lists none",
                )
                continue
            if pattern in seen_patterns:
                yield PlanIssue(
                    "pii_plan_duplicate_entry",
                    f"column {col!r} lists pattern {pattern!r} more than once; only the first of them would "
                    "ever be used",
                )
                continue
            seen_patterns.add(pattern)
            if spec.entity_type in DATE_PATTERN_ENTITIES:
                yield from _iter_date_pattern_issues(col, pattern, samples)
            elif spec.entity_type in TEMPLATE_PATTERN_ENTITIES:
                if patterns.value_template_is_constant(pattern):
                    yield PlanIssue(
                        "pii_plan_pattern_invalid",
                        f"pattern {pattern!r} for column {col!r} has no variable position, so every row "
                        "would get the same value; use '#' (digit), '^' (A-Z), '@' (a-z), '*' (alphanumeric) "
                        "or a class such as '[68]'",
                    )
                else:
                    yield from _iter_unused_template_issues(col, pattern, samples)
            elif spec.entity_type in PERSONA_PATTERN_ENTITIES:
                yield from _iter_persona_pattern_issues(col, spec.entity_type, pattern)


def _iter_persona_issues(plan: PiiReplacementPlan, df_cols: set[str]) -> Iterator[PlanIssue]:
    seen_personas: set[str] = set()
    for col_set in plan.persona_backed_columns:
        if col_set.persona in seen_personas:
            yield PlanIssue(
                "pii_plan_duplicate_entry",
                f"persona {col_set.persona!r} appears more than once in replacement plan",
            )
        seen_personas.add(col_set.persona)
        seen_attributes: set[str] = set()
        for cond in col_set.match_persona_by:
            if cond.persona_attribute in seen_attributes:
                yield PlanIssue(
                    "pii_plan_duplicate_entry",
                    f"persona_attribute {cond.persona_attribute!r} appears more than once "
                    f"for persona {col_set.persona!r}",
                )
            seen_attributes.add(cond.persona_attribute)
            if cond.column_name not in df_cols:
                yield PlanIssue(
                    "pii_plan_column_not_found",
                    f"match_persona_by column {cond.column_name!r} not found in dataframe",
                )


def _iter_section_placement_advisories(
    plan: PiiReplacementPlan,
    *,
    persona_backend: str,
) -> Iterator[PlanIssue]:
    """Warn when a column's YAML section changes replacement behavior.

    Entity type (and persona backend) decide the generation channel. Listing a
    standalone-mapped column under ``persona_backed_columns`` does not change
    that channel, so it is not warned. Listing a persona-sourced column only
    under ``standalone_columns_to_replace`` does change consistency: it will not
    share a synthetic person with other columns.
    """
    persona_expected = {
        label
        for label in entities.ENTITY_REGISTRY
        if entities.effective_apply_path(label, persona_backend) == "persona"
    }

    if persona_backend == "faker":
        for col_set in plan.persona_backed_columns:
            for cond in col_set.match_persona_by:
                if cond.persona_attribute == "ethnic_background":
                    yield PlanIssue(
                        "pii_plan_ethnic_background_ignored_under_faker",
                        f"persona {col_set.persona!r} match_persona_by lists ethnic_background "
                        f"on column {cond.column_name!r}, but person.backend is 'faker', which "
                        "only conditions names on sex; ethnic_background is ignored. Remove it "
                        "from the plan or use the managed/pgm backend.",
                        severity="warning",
                    )

    for col_set in plan.persona_backed_columns:
        loc = f"persona {col_set.persona!r}"
        for spec in col_set.columns_to_replace:
            label = spec.entity_type.value if spec.entity_type is not None else None
            if label is None:
                continue
            if label == PiiEntity.free_text.value:
                yield PlanIssue(
                    "pii_plan_free_text_under_persona",
                    f"{loc} lists {spec.column_name!r} as free_text; free-text columns belong in "
                    "standalone_columns_to_replace (they still use free-text replacement either way).",
                    severity="warning",
                )

    for spec in plan.standalone_columns_to_replace:
        label = spec.entity_type.value if spec.entity_type is not None else None
        if label is None or label not in persona_expected:
            continue
        yield PlanIssue(
            "pii_plan_persona_column_under_standalone",
            f"standalone_columns_to_replace lists {spec.column_name!r} as {label}; "
            "person-identifying columns listed only there are replaced without sharing a "
            "synthetic person with other columns. Move them under persona_backed_columns "
            "when they should stay consistent with each other.",
            severity="warning",
        )


def iter_plan_advisories(
    plan: PiiReplacementPlan,
    *,
    persona_backend: str,
) -> Iterator[PlanIssue]:
    """Yield non-blocking advisories for section placement mismatches.

    Args:
        plan: Replacement plan to inspect.
        persona_backend: Persona sampling backend (``managed``, ``pgm``, ``faker``).

    Yields:
        ``PlanIssue`` instances with ``severity="warning"``.
    """
    yield from _iter_section_placement_advisories(plan, persona_backend=persona_backend)


def iter_plan_issues(
    df: pd.DataFrame,
    plan: PiiReplacementPlan,
    *,
    data_config: DataParameters | None = None,
    time_series: TimeSeriesParameters | None = None,
) -> Iterator[PlanIssue]:
    """Yield every problem that makes ``plan`` unusable on ``df``.

    Used by the preflight check to report a hand-edited plan's mistakes in one
    pass; ``validate_plan`` wraps this and raises on the first error. Section
    placement mismatches are warnings via ``iter_plan_advisories``, not errors.

    Args:
        df: Input dataframe the plan will be applied to.
        plan: Replacement plan to validate.
        data_config: Optional data parameters for group-key and protected-column checks.
        time_series: Optional time-series parameters for timestamp protection.

    Yields:
        ``PlanIssue`` instances with ``severity="error"`` for blocking problems.
    """
    df_cols = set(df.columns)
    yield from _iter_protected_column_issues(plan, data_config=data_config, time_series=time_series)
    yield from _iter_scope_issues(plan, df_cols, data_config)
    yield from _iter_column_reference_issues(plan, df_cols)
    yield from _iter_column_spec_issues(df, plan, df_cols)
    yield from _iter_persona_issues(plan, df_cols)


def validate_plan(
    df: pd.DataFrame,
    plan: PiiReplacementPlan,
    *,
    data_config: DataParameters | None = None,
    time_series: TimeSeriesParameters | None = None,
) -> None:
    """Validate a resolved plan against the input dataframe.

    Args:
        df: Input dataframe the plan will be applied to.
        plan: Replacement plan to validate.
        data_config: Optional data parameters for group-key and protected-column checks.
        time_series: Optional time-series parameters for timestamp protection.

    Raises:
        ParameterError: On the first validation error with ``severity="error"``.
    """
    for issue in iter_plan_issues(df, plan, data_config=data_config, time_series=time_series):
        if issue.severity == "error":
            raise ParameterError(issue.message)
