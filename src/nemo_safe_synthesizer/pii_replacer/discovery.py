# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MVP PII column discovery."""

from __future__ import annotations

import pandas as pd

from ..artifacts.analyzers.field_features import describe_field
from ..artifacts.base.fields import FieldType

from ..config.pii_replacement import (
    PiiColumnPlan,
    PiiEntity,
    PiiReplacementPlan,
    ReplacePiiConfig,
)
from ..observability import get_logger
from . import core
from .plan import runtime_plan_to_pii_plan
from .runtime_config import RuntimeConfig

logger = get_logger(__name__)


def _discovery_exclude_columns(detected: dict) -> set[str]:
    """Columns already assigned to structured PII roles or demographics."""
    exclude = set()
    for role in detected.get("roles", []):
        exclude |= set((role.get("fields") or {}).values())
        exclude |= {v for v in (role.get("demographics") or {}).values() if v}
    for ent in detected.get("non_person", []):
        col = ent.get("column")
        if not col:
            continue
        exclude.add(col)
    return exclude


_FREE_TEXT_ELIGIBLE_FIELD_TYPES = frozenset({FieldType.TEXT, FieldType.OTHER})
_NON_FREE_TEXT_FIELD_TYPES = frozenset(
    {
        FieldType.BINARY,
        FieldType.CATEGORICAL,
        FieldType.NUMERIC,
        FieldType.EMPTY,
    }
)


def _free_text_eligibility(col: str, series: pd.Series) -> tuple[bool, str]:
    """Return whether a column may be free text and a short reason string."""
    from pandas.api.types import is_object_dtype, is_string_dtype

    if not (is_object_dtype(series.dtype) or is_string_dtype(series.dtype)):
        return False, f"dtype={series.dtype}"
    field_type = describe_field(col, series).type
    if field_type in _NON_FREE_TEXT_FIELD_TYPES:
        return False, f"field_type={field_type.value}"
    if field_type in _FREE_TEXT_ELIGIBLE_FIELD_TYPES:
        return True, f"field_type={field_type.value}"
    return False, f"field_type={field_type.value}"


def _nss_free_text_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """Free-text columns via dtype gate + the shared NSS field classifier."""
    text_fields: list[str] = []
    for col in df.columns:
        if col in exclude:
            logger.runtime.debug(
                f"[PII Replacement] Column {col!r} not scanned as free text for PII detection: "
                "already handled as a structured column"
            )
            continue
        eligible, reason = _free_text_eligibility(col, df[col])
        if eligible:
            text_fields.append(col)
            logger.runtime.info(f"[PII Replacement] Column {col!r} scanned as free text for PII detection ({reason})")
        else:
            logger.runtime.info(
                f"[PII Replacement] Column {col!r} not scanned as free text for PII detection ({reason})"
            )
    return text_fields


def _core_config(runtime: RuntimeConfig) -> core.Config:
    return core.Config(
        locale=runtime.locale,
        random_seed=runtime.random_seed,
        replace_group_key=runtime.replace_group_key,
        persona_backend=runtime.persona_backend,
        sdg_pgms_src=runtime.sdg_pgms_src,
        managed_assets_path=runtime.managed_assets_path,
        low_card_max=runtime.low_card_max,
        dominant_pattern_min_coverage=runtime.dominant_pattern_min_coverage,
        value_match_threshold=runtime.value_match_threshold,
        id_unique_ratio=runtime.id_unique_ratio,
        name_fuzzy_threshold=runtime.name_fuzzy_threshold,
        infer_value_patterns=runtime.infer_value_patterns,
        pattern_class_max=runtime.pattern_class_max,
        pattern_rare_char_frac=runtime.pattern_rare_char_frac,
        pattern_sample_cap=runtime.pattern_sample_cap,
    )


def _detect_full_dataframe(
    df: pd.DataFrame,
    cfg: core.Config,
    *,
    group_key: str | None = None,
    llm_enhancement: bool = False,
) -> dict:
    # When a group key is set, cardinality of per-group (group-constant) columns
    # must be measured against the number of groups, not rows. Otherwise a
    # per-group identifier (e.g. one MRN per patient) that repeats across every
    # row of its group looks low-cardinality and fails the unique_identifier
    # gate. ``scoped_column_stats`` recomputes ``unique_ratio`` per group.
    stats = core.scoped_column_stats(df, group_key, cfg.group_constancy_threshold)
    out = core._detect_subset_mvp(df, stats, cfg)

    roles = []
    for role in out["roles"]:
        roles.append(
            {
                "role": role["role"],
                "fields": role["fields"],
                "field_meta": role.get("field_meta", {}),
                "demographics": role["demographics"],
            }
        )

    non_person = list(out["non_person"])
    core._attach_value_patterns(df, non_person, cfg)

    exclude = _discovery_exclude_columns(out)
    exclude |= set(out.get("identified_not_replaced", []))

    # In MVP mode, free-text columns are only ever modified by propagating
    # synthetic values from structured detections (person fields and replaced
    # non-person columns). Temporal "identify-only" entities are never replaced,
    # so they don't count. When nothing structured is replaceable there is
    # nothing to propagate, so scanning free text is pointless. LLM mode detects
    # entities directly, so it always scans.
    has_person = any((role.get("fields") or {}) for role in roles)
    has_replaceable_non_person = any(
        e.get("entity") not in core.IDENTIFIED_NOT_REPLACED_ENTITIES for e in non_person
    )
    has_structured = has_person or has_replaceable_non_person

    if llm_enhancement or has_structured:
        free_text_columns = _nss_free_text_columns(df, exclude)
    else:
        free_text_columns = []
        logger.runtime.info(
            "[PII Replacement] No structured PII columns detected; skipping free-text scan "
            "(nothing to propagate in non-LLM mode)"
        )

    identified = out.get("identified_not_replaced", [])
    if identified:
        logger.runtime.info(
            f"[PII Replacement] Temporal columns identified (excluded from replacement plan): {', '.join(identified)}"
        )
    if non_person:
        non_person_desc = ", ".join(f"{e['column']} ({e['entity']})" for e in non_person)
        logger.runtime.info(f"[PII Replacement] Structured non-person columns: {non_person_desc}")
    if free_text_columns:
        logger.runtime.info(
            f"[PII Replacement] Columns scanned as free text for PII detection: {', '.join(free_text_columns)}"
        )

    return {
        "roles": roles,
        "non_person": non_person,
        "free_text_columns": free_text_columns,
    }


def discover_plan(
    df: pd.DataFrame,
    group_key: str | None,
    runtime: RuntimeConfig,
    config: ReplacePiiConfig,
) -> PiiReplacementPlan:
    cfg = _core_config(runtime)
    detected = _detect_full_dataframe(df, cfg, group_key=group_key, llm_enhancement=config.llm_enhancement)

    plan = runtime_plan_to_pii_plan(detected, group_key=group_key)

    if config.discovery.replace_group_key and group_key and group_key in df.columns:
        # The group key is replaced as a (non-person) unique_identifier; its
        # replacement never depends on a persona, so it is a plain column with no
        # ``persona`` reference rather than a persona-associated field.
        pat = None
        cov = None
        for ent in detected["non_person"]:
            if ent.get("column") == group_key:
                pat = ent.get("pattern")
                cov = ent.get("dominant_pattern_coverage")
                break
        existing = plan.columns.get(group_key)
        if existing is None or existing.entity_type != PiiEntity.unique_identifier:
            plan.columns[group_key] = PiiColumnPlan(
                entity_type=PiiEntity.unique_identifier,
                pattern=pat,
                dominant_pattern_coverage=cov,
            )
        elif pat and existing.pattern is None:
            plan.columns[group_key] = existing.model_copy(
                update={"pattern": pat, "dominant_pattern_coverage": cov}
            )

    return plan
