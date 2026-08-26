# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Programmatic PII column discovery → flat ``PiiReplacementPlan``."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import pandas as pd

from ...config.replace_pii import (
    ALLOWED_DEPENDS_ON,
    ENTITY_BY_TYPE,
    ConditioningColumn,
    EntityType,
    PatternKind,
    PiiColumnPlan,
    PiiReplacementPlan,
    PiiReplacementScope,
    ReplacePiiConfig,
    is_columns_to_replace_type,
)
from ...observability import get_logger
from .. import detection, entities, patterns
from ..models import DiscoveryResult, SamePersonBundle

logger = get_logger(__name__)


def _discovery_exclude_columns(discovery: DiscoveryResult) -> set[str]:
    """Columns already assigned to a same-person column bundle or standalone entity."""
    exclude: set[str] = set()
    for bundle in discovery.same_person_bundles:
        exclude |= {field.column for field in bundle.fields.values()}
        exclude |= set(bundle.demographics.values())
    for ent in discovery.standalone_columns:
        if ent.column:
            exclude.add(ent.column)
    exclude |= set(discovery.conditioning_demographics.values())
    return exclude


def _detect_full_dataframe(
    df: pd.DataFrame,
    cfg: entities.Config,
    *,
    group_key: str | None = None,
) -> DiscoveryResult:
    _ = group_key  # Used only for plan scope in ``discover_plan``, not detection.
    discovery = detection.detect_structured_columns(df, cfg)

    patterns.attach_name_patterns(df, discovery.same_person_bundles, cfg)
    patterns.attach_value_patterns(df, discovery.standalone_columns, cfg)

    exclude = _discovery_exclude_columns(discovery)
    exclude |= set(discovery.identified_not_replaced)

    has_same_person = any(bundle.fields for bundle in discovery.same_person_bundles)
    has_replaceable_standalone = any(
        e.entity and not entities.is_identify_only(e.entity) for e in discovery.standalone_columns
    )
    has_structured = has_same_person or has_replaceable_standalone

    if has_structured:
        free_text_columns = detection.select_free_text_columns(df, exclude)
    else:
        free_text_columns = []
        text_like = detection.select_free_text_columns(df, exclude)
        if text_like:
            logger.user.warning(
                "[PII Replacement] No structured PII columns detected, so free-text columns "
                f"will not be scanned or replaced: {', '.join(text_like)}. "
                "Add name/email or standalone entity columns, or a hand-written plan; "
                "free text only propagates values replaced from structured columns "
                "(no NER in this release)."
            )
        else:
            logger.runtime.info(
                "[PII Replacement] No structured PII columns detected; skipping free-text scan "
                "(nothing to propagate in heuristics mode)"
            )

    discovery.free_text_columns = free_text_columns

    identified = discovery.identified_not_replaced
    if identified:
        logger.runtime.info(
            f"[PII Replacement] Columns identified (excluded from replacement plan): {', '.join(identified)}"
        )
    if discovery.standalone_columns:
        standalone_desc = ", ".join(f"{e.column} ({e.entity})" for e in discovery.standalone_columns)
        logger.runtime.info(f"[PII Replacement] Standalone columns: {standalone_desc}")

    return discovery


def _entity_from_label(label: str | None) -> EntityType | None:
    if not label:
        return None
    return entities.entity_type_for_label(label)


def _mapped_entity_or_warn(label: str | None, *, column: str | None) -> EntityType | None:
    """Map a detector label to ``EntityType``, warning when unknown."""
    entity = _entity_from_label(label)
    if label and entity is None:
        where = f"column {column!r}" if column else "a detected column"
        logger.runtime.warning(
            f"[PII Replacement] Skipping {where}: detected entity label {label!r} is not a "
            "known EntityType; values will not be replaced. Review the plan or update the "
            "entity vocabulary if this label should be supported."
        )
    return entity


def _plan_pattern(entity: EntityType, pattern: str | None) -> str | None:
    """Keep pattern only when the contract allows it for this entity type."""
    if pattern is None:
        return None
    if ENTITY_BY_TYPE[entity].pattern_kind is PatternKind.none:
        return None
    return pattern


def _conditioner(
    column_name: str,
    entity_type: EntityType,
    *,
    replace_columns: set[str],
) -> ConditioningColumn:
    """Build a depends_on edge; omit entity_type when the conditioner is also replaced."""
    if column_name in replace_columns:
        return ConditioningColumn(column_name=column_name, entity_type=None)
    return ConditioningColumn(column_name=column_name, entity_type=entity_type)


def _depends_on_for_target(
    *,
    target_type: EntityType,
    bundle: SamePersonBundle,
    replace_columns: set[str],
) -> list[ConditioningColumn]:
    """Attach conditioners for one replaceable field using discovery priorities."""
    target_label = target_type.value
    allowed = ALLOWED_DEPENDS_ON.get(target_type)
    if not allowed:
        return []

    edges: list[ConditioningColumn] = []

    def _add(column: str, entity: EntityType) -> None:
        if entity not in allowed:
            return
        # At most one edge per conditioner entity_type.
        if any(
            (e.entity_type is entity)
            or (
                e.entity_type is None
                and entity.value in bundle.fields
                and bundle.fields[entity.value].column == e.column_name
            )
            for e in edges
        ):
            return
        for existing in edges:
            for lbl, field in bundle.fields.items():
                if field.column == existing.column_name and lbl == entity.value:
                    return
            for demo_attr, demo_col in bundle.demographics.items():
                if demo_col == existing.column_name and demo_attr == entity.value:
                    return
        edges.append(_conditioner(column, entity, replace_columns=replace_columns))

    # --- name parts: full_name else demographics ---
    if target_label in {"first_name", "middle_name", "last_name"}:
        full = bundle.fields.get("full_name")
        if full is not None:
            _add(full.column, EntityType.full_name)
        else:
            if target_label != "last_name" and "gender" in bundle.demographics:
                _add(bundle.demographics["gender"], EntityType.gender)
            if "ethnic_background" in bundle.demographics:
                _add(bundle.demographics["ethnic_background"], EntityType.ethnic_background)
        return edges

    # --- email: name parts preferred over full_name ---
    if target_label == "email":
        part_labels = ("first_name", "middle_name", "last_name")
        has_parts = False
        for lbl in part_labels:
            field = bundle.fields.get(lbl)
            if field is not None:
                _add(field.column, EntityType(lbl))
                has_parts = True
        if not has_parts:
            full = bundle.fields.get("full_name")
            if full is not None:
                _add(full.column, EntityType.full_name)
        return edges

    # --- full_name: demographics only ---
    if target_label == "full_name":
        if "gender" in bundle.demographics:
            _add(bundle.demographics["gender"], EntityType.gender)
        if "ethnic_background" in bundle.demographics:
            _add(bundle.demographics["ethnic_background"], EntityType.ethnic_background)
        return edges

    # street_address: geo depends_on deferred to a later PR
    return edges


def _columns_by_entity(detected: DiscoveryResult) -> dict[str, list[str]]:
    """Inventory of replaceable + conditioning columns keyed by entity label."""
    by_entity: dict[str, list[str]] = defaultdict(list)
    for bundle in detected.same_person_bundles:
        for label, field_info in bundle.fields.items():
            by_entity[label].append(field_info.column)
        for attr, col in bundle.demographics.items():
            by_entity[attr].append(col)
    for ent in detected.standalone_columns:
        if ent.column and ent.entity:
            by_entity[ent.entity].append(ent.column)
    for attr, col in detected.conditioning_demographics.items():
        if col not in by_entity[attr]:
            by_entity[attr].append(col)
    return dict(by_entity)


def _fmt_column_choice(columns: Sequence[str]) -> str:
    if len(columns) == 1:
        return columns[0]
    return "one of [" + ", ".join(columns) + "]"


def build_depends_on_hints(detected: DiscoveryResult) -> list[str]:
    """Suggest ``depends_on`` edits from discovery priorities when linking is ambiguous.

    Mirrors ``_depends_on_for_target`` priority order, but lists every candidate
    column when an entity type appears more than once. Returns YAML comment body
    lines (no leading ``#``).
    """
    if not detected.person_link_ambiguous:
        return []

    by_entity = _columns_by_entity(detected)
    duplicates = {label: cols for label, cols in sorted(by_entity.items()) if len(cols) > 1}
    lines: list[str] = [
        "depends_on omitted: multiple columns share the same person-related entity type.",
        "Heuristic discovery cannot link them to one subject. Suggested edits "
        "(pick one column per edge; exclusivity rules still apply):",
    ]
    if duplicates:
        lines.append(
            "duplicate entity types: "
            + "; ".join(f"{label}={', '.join(cols)}" for label, cols in duplicates.items())
        )

    def _hint_targets(target_label: str, conditioner_cols: Sequence[str], conditioner_label: str) -> None:
        targets = by_entity.get(target_label) or []
        if not targets or not conditioner_cols:
            return
        choice = _fmt_column_choice(conditioner_cols)
        for target in targets:
            lines.append(f"  - {target} ({target_label}) can depends_on {choice} ({conditioner_label})")

    full_names = by_entity.get("full_name") or []
    genders = by_entity.get("gender") or []
    ethnics = by_entity.get("ethnic_background") or []

    # name parts: full_name else demographics
    for part in ("first_name", "middle_name", "last_name"):
        if full_names:
            _hint_targets(part, full_names, "full_name")
        else:
            if part != "last_name":
                _hint_targets(part, genders, "gender")
            _hint_targets(part, ethnics, "ethnic_background")

    # email: name parts preferred over full_name
    email_cols = by_entity.get("email") or []
    if email_cols:
        part_cols = [c for lbl in ("first_name", "middle_name", "last_name") for c in (by_entity.get(lbl) or [])]
        if part_cols:
            # One hint line naming all available parts (same as multi-edge email depends_on).
            choice = _fmt_column_choice(part_cols)
            for email_col in email_cols:
                lines.append(f"  - {email_col} (email) can depends_on {choice} (name parts)")
        elif full_names:
            _hint_targets("email", full_names, "full_name")

    # full_name: demographics
    _hint_targets("full_name", genders, "gender")
    _hint_targets("full_name", ethnics, "ethnic_background")

    # Drop the intro-only case when nothing actionable beyond the duplicate list.
    actionable = [ln for ln in lines if ln.startswith("  - ")]
    if not actionable:
        return lines[:2] + (lines[2:3] if duplicates else [])
    return lines


def _detected_to_plan(
    detected: DiscoveryResult,
    *,
    scope: PiiReplacementScope,
) -> tuple[PiiReplacementPlan, list[str]]:
    """Convert structured detection into a flat ``PiiReplacementPlan`` plus YAML hints."""
    columns: list[PiiColumnPlan] = []
    link_ambiguous = detected.person_link_ambiguous

    replace_columns: set[str] = set()
    pending: list[tuple[str, EntityType, str | None, SamePersonBundle | None]] = []

    for bundle in detected.same_person_bundles:
        if not bundle.fields:
            continue  # demographics-only bundles are dropped
        for label, field_info in bundle.fields.items():
            entity = _mapped_entity_or_warn(label, column=field_info.column)
            if entity is None:
                continue
            if not is_columns_to_replace_type(entity):
                continue
            replace_columns.add(field_info.column)
            pending.append((field_info.column, entity, field_info.pattern, None if link_ambiguous else bundle))

    for ent in detected.standalone_columns:
        entity = _mapped_entity_or_warn(ent.entity, column=ent.column)
        if ent.column and entity is not None and is_columns_to_replace_type(entity):
            replace_columns.add(ent.column)
            pending.append((ent.column, entity, ent.pattern, None))

    for col in detected.free_text_columns:
        replace_columns.add(col)
        pending.append((col, EntityType.free_text, None, None))

    for column_name, entity, pattern, bundle in pending:
        depends_on: list[ConditioningColumn] = []
        if bundle is not None and not link_ambiguous:
            depends_on = _depends_on_for_target(
                target_type=entity,
                bundle=bundle,
                replace_columns=replace_columns,
            )
        columns.append(
            PiiColumnPlan(
                column_name=column_name,
                entity_type=entity,
                pattern=_plan_pattern(entity, pattern),
                depends_on=depends_on,
            )
        )

    plan = PiiReplacementPlan(scope=scope, columns_to_replace=columns)
    hints = build_depends_on_hints(detected) if link_ambiguous else []
    return plan, hints


def discover_plan(
    df: pd.DataFrame,
    group_key: str | None,
    cfg: entities.Config,
    config: ReplacePiiConfig,
) -> PiiReplacementPlan:
    """Run heuristic auto-discovery and emit a typed replacement plan.

    Args:
        df: Input dataframe whose columns are classified.
        group_key: Training group column name, or ``None`` for dataframe scope.
        cfg: Engine configuration for detection thresholds and sampler backend.
        config: User-facing PII replacement configuration.

    Returns:
        Validated ``PiiReplacementPlan`` with scope inferred from ``group_key``.
        When persona entity types are duplicated, ``depends_on`` is empty; callers
        that persist YAML should use ``discover_plan_with_hints`` for comment hints.
    """
    plan, _hints = discover_plan_with_hints(df, group_key, cfg, config)
    return plan


def discover_plan_with_hints(
    df: pd.DataFrame,
    group_key: str | None,
    cfg: entities.Config,
    config: ReplacePiiConfig,
) -> tuple[PiiReplacementPlan, list[str]]:
    """Like ``discover_plan``, also returning YAML ``depends_on`` hint comment lines."""
    # llm_enhancement is refused by ReplacePiiConfig; discovery is heuristics-only.
    _ = config
    discovery = _detect_full_dataframe(df, cfg, group_key=group_key)

    if group_key and group_key in df.columns:
        scope = PiiReplacementScope.group
    else:
        if group_key:
            logger.user.warning(
                f"[PII Replacement] group_training_examples_by={group_key!r} is not a dataframe "
                "column; discovering with dataframe scope instead of group."
            )
        scope = PiiReplacementScope.dataframe
    return _detected_to_plan(discovery, scope=scope)
