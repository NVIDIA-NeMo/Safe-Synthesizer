# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Programmatic PII column discovery."""

from __future__ import annotations

import pandas as pd

from ...config.replace_pii import (
    PersonaColumnSet,
    PersonaMatchColumn,
    PiiColumnPlan,
    PiiEntity,
    PiiReplacementPlan,
    PiiReplacementScope,
    PiiReplacerConfig,
)
from ...observability import get_logger
from .. import detection, entities, patterns
from ..llm import PiiDiscoveryEnhancer, select_discovery_enhancer
from ..models import DiscoveryResult

logger = get_logger(__name__)


def _discovery_exclude_columns(discovery: DiscoveryResult) -> set[str]:
    """Columns already assigned to a persona, or read to match one."""
    exclude = set()
    for persona_set in discovery.personas:
        exclude |= {field.column for field in persona_set.fields.values()}
        exclude |= {matcher.column_name for matcher in persona_set.match_persona_by if matcher.column_name}
    for ent in discovery.standalone_columns:
        if ent.column:
            exclude.add(ent.column)
    return exclude


def _detect_full_dataframe(
    df: pd.DataFrame,
    cfg: entities.Config,
    *,
    group_key: str | None = None,
    llm_enhancement: bool = False,
    enhancer: PiiDiscoveryEnhancer | None = None,
) -> DiscoveryResult:
    # When a group key is set, cardinality of per-group (group-constant) columns
    # must be measured against the number of groups, not rows. Otherwise a
    # per-group attribute that repeats across every row of its group looks
    # low-variety and can be mistaken for free text. ``scoped_column_stats``
    # recomputes ``unique_ratio`` per group and tags structural ``grain``.
    stats = detection.scoped_column_stats(df, group_key, cfg.group_constancy_threshold)
    discovery = detection.detect_structured_columns(df, stats, cfg)

    patterns.attach_persona_patterns(df, discovery.personas, cfg)
    patterns.attach_value_patterns(df, discovery.standalone_columns, cfg)

    # Heuristics first; LLM re-judges structured detection with that context
    # (same seam as the former enrich_structured_detection), before free-text
    # eligibility / plan emission.
    llm = select_discovery_enhancer(llm_enhancement=llm_enhancement, enhancer=enhancer)
    discovery = llm.review_discovery(df, discovery, cfg)

    exclude = _discovery_exclude_columns(discovery)
    exclude |= set(discovery.identified_not_replaced)

    # Without LLM enhancement, free-text columns are only modified by propagating
    # values replaced from structured entity columns (persona-backed or standalone).
    # If nothing structured was found, there is nothing to propagate, so skip the
    # free-text plan entry in heuristics mode. LLM mode detects entities directly,
    # so it always scans (once an enhancer's review_discovery is implemented).
    has_persona_columns = any(persona_set.fields for persona_set in discovery.personas)
    has_replaceable_standalone = any(
        e.entity and not entities.is_identify_only(e.entity) for e in discovery.standalone_columns
    )
    has_structured = has_persona_columns or has_replaceable_standalone

    if llm_enhancement or has_structured:
        free_text_columns = detection.select_free_text_columns(df, exclude)
    else:
        free_text_columns = []
        text_like = detection.select_free_text_columns(df, exclude)
        if text_like:
            logger.user.warning(
                "[PII Replacement] No structured PII columns detected, so free-text columns "
                f"will not be scanned or replaced: {', '.join(text_like)}. "
                "Add persona-backed or standalone entity columns, or a hand-written plan; "
                "free text only propagates values replaced from structured columns "
                "(no NER in this release)."
            )
        else:
            logger.runtime.info(
                "[PII Replacement] No structured PII columns detected; skipping free-text scan "
                "(nothing to propagate in non-LLM mode)"
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


def _entity_from_label(label: str | None) -> PiiEntity | None:
    if not label:
        return None
    try:
        return PiiEntity(label)
    except ValueError:
        return None


def _mapped_entity_or_warn(label: str | None, *, column: str | None) -> PiiEntity | None:
    """Map a detector label to ``PiiEntity``, warning when a non-empty label is unknown.

    Logs only the column name and entity label — never cell values — so a detector /
    vocabulary drift is visible without writing PII into the log.
    """
    entity = _entity_from_label(label)
    if label and entity is None:
        where = f"column {column!r}" if column else "a detected column"
        logger.runtime.warning(
            f"[PII Replacement] Skipping {where}: detected entity label {label!r} is not a "
            "known PiiEntity; values will not be replaced. Review the plan or update the "
            "entity vocabulary if this label should be supported."
        )
    return entity


def _detected_to_plan(
    detected: DiscoveryResult,
    *,
    scope: PiiReplacementScope,
) -> PiiReplacementPlan:
    """Convert structured detection into a typed ``PiiReplacementPlan``."""
    discovery = detected
    persona_backed: list[PersonaColumnSet] = []
    for persona_set in discovery.personas:
        cols: list[PiiColumnPlan] = []
        for label, field_info in persona_set.fields.items():
            entity = _mapped_entity_or_warn(label, column=field_info.column)
            if entity is not None:
                cols.append(
                    PiiColumnPlan(
                        column_name=field_info.column,
                        entity_type=entity,
                        patterns=list(field_info.patterns),
                    )
                )
        matchers = [
            PersonaMatchColumn(
                persona_attribute=matcher.persona_attribute,
                column_name=matcher.column_name,
            )
            for matcher in persona_set.match_persona_by
            if matcher.persona_attribute and matcher.column_name
        ]
        # Demographics alone do not justify a persona set: match_persona_by is only
        # useful when there are person-related columns to replace.
        if cols:
            persona_backed.append(
                PersonaColumnSet(
                    persona=persona_set.persona,
                    columns_to_replace=cols,
                    match_persona_by=matchers,
                )
            )

    standalone: list[PiiColumnPlan] = []
    for ent in discovery.standalone_columns:
        entity = _mapped_entity_or_warn(ent.entity, column=ent.column)
        if ent.column and entity is not None:
            standalone.append(PiiColumnPlan(column_name=ent.column, entity_type=entity, patterns=list(ent.patterns)))
    for col in discovery.free_text_columns:
        standalone.append(PiiColumnPlan(column_name=col, entity_type=PiiEntity.free_text))

    return PiiReplacementPlan(
        scope=scope,
        persona_backed_columns=persona_backed,
        standalone_columns_to_replace=standalone,
    )


def discover_plan(
    df: pd.DataFrame,
    group_key: str | None,
    cfg: entities.Config,
    config: PiiReplacerConfig,
    *,
    enhancer: PiiDiscoveryEnhancer | None = None,
) -> PiiReplacementPlan:
    """Run auto-discovery and emit a typed replacement plan.

    Args:
        df: Input dataframe whose columns are classified.
        group_key: Training group column name, or ``None`` for dataframe scope.
        cfg: Engine configuration for detection thresholds and persona backend.
        config: User-facing PII replacement configuration.
        enhancer: Optional discovery enhancer injected for ``review_discovery``.

    Returns:
        Validated ``PiiReplacementPlan`` with scope inferred from ``group_key``.
    """
    discovery = _detect_full_dataframe(
        df,
        cfg,
        group_key=group_key,
        llm_enhancement=config.llm_enhancement,
        enhancer=enhancer,
    )

    # Only emit group scope when the group key is actually present. Aligns
    # with GroupbyColumnCheck / validate_plan so auto plans are runnable on
    # non-pipeline entry points.
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
