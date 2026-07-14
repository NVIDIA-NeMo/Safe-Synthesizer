# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan resolution, validation, and conversion to the runtime engine shape."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..config.data import DataParameters
from ..config.pii_replacement import (
    AssociatedColumnSet,
    PiiEntity,
    PiiReplacementPlan,
    ReplacePiiConfig,
)
from ..errors import ParameterError
from .runtime_config import RuntimeConfig


def entity_to_engine_label(entity: PiiEntity | str | None) -> str | None:
    """Return the engine string label for a plan entity (or ``None``)."""
    if entity is None:
        return None
    return entity.value if isinstance(entity, PiiEntity) else entity


def entity_from_engine_label(label: str | None) -> PiiEntity | None:
    """Return the ``PiiEntity`` for an engine label, or ``None`` if unknown."""
    if not label:
        return None
    try:
        return PiiEntity(label)
    except ValueError:
        return None


def _replacement_columns(plan: PiiReplacementPlan) -> list[str]:
    cols: list[str] = []
    for col_set in plan.associated_column_sets.values():
        cols.extend(col_set.columns_to_replace)
    cols.extend(plan.unassociated_columns_to_replace)
    return cols


def validate_plan(
    df: pd.DataFrame,
    plan: PiiReplacementPlan,
    *,
    data_config: DataParameters | None = None,
    runtime: RuntimeConfig | None = None,
    discovered: bool = False,
) -> None:
    """Validate a resolved plan against the input dataframe."""
    df_cols = set(df.columns)
    if plan.group_key and plan.group_key not in df_cols:
        raise ParameterError(f"plan.group_key column {plan.group_key!r} not found in dataframe")

    if data_config is not None:
        expected_group = data_config.group_training_examples_by
        if plan.group_key != expected_group:
            raise ParameterError(
                f"plan.group_key {plan.group_key!r} must match data.group_training_examples_by {expected_group!r}"
            )

    seen: set[str] = set()
    for col in _replacement_columns(plan):
        if col not in df_cols:
            raise ParameterError(f"replacement plan references missing column {col!r}")
        if col in seen:
            raise ParameterError(f"column {col!r} appears more than once in replacement plan")
        seen.add(col)

    for col_set in plan.associated_column_sets.values():
        cond = col_set.conditioning_columns
        if cond is None:
            continue
        for attr in ("gender", "ethnic_background"):
            col = getattr(cond, attr)
            if col and col not in df_cols:
                raise ParameterError(f"conditioning column {col!r} not found in dataframe")

    if discovered and runtime is not None:
        for col, spec in plan.unassociated_columns_to_replace.items():
            if spec.entity_type == PiiEntity.unique_identifier:
                nun = int(df[col].dropna().nunique())
                ratio = nun / max(len(df), 1)
                if ratio < runtime.id_unique_ratio:
                    raise ParameterError(
                        f"auto-discovered unique_id column {col!r} failed id_unique_ratio gate "
                        f"({ratio:.4f} < {runtime.id_unique_ratio})"
                    )


def unique_id_advisories(df: pd.DataFrame, plan: PiiReplacementPlan, runtime: RuntimeConfig) -> list[str]:
    warnings: list[str] = []
    for col, spec in plan.unassociated_columns_to_replace.items():
        if spec.entity_type != PiiEntity.unique_identifier or col not in df.columns:
            continue
        nun = int(df[col].dropna().nunique())
        ratio = nun / max(len(df), 1)
        if ratio < runtime.id_unique_ratio:
            warnings.append(
                f"unique_id column {col!r} has unique_ratio {ratio:.4f} "
                f"(below advisory threshold {runtime.id_unique_ratio})"
            )
    return warnings


def load_plan_from_path(path: str) -> PiiReplacementPlan:
    p = Path(path)
    text = p.read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ParameterError(f"plan file {path!r} must contain a mapping")
    return PiiReplacementPlan.model_validate(data)


def plan_to_runtime(plan: PiiReplacementPlan) -> dict[str, Any]:
    """Convert declarative plan to the runtime dict consumed by the engine."""
    roles: list[dict[str, Any]] = []
    for role_name, col_set in plan.associated_column_sets.items():
        fields: dict[str, str] = {}
        field_meta: dict[str, dict[str, Any]] = {}
        for col, spec in col_set.columns_to_replace.items():
            if spec.entity_type is None or spec.entity_type == PiiEntity.free_text:
                continue
            label = entity_to_engine_label(spec.entity_type)
            if label:
                fields[label] = col
                if spec.pattern is not None or spec.dominant_pattern_coverage is not None:
                    field_meta[label] = {
                        "pattern": spec.pattern,
                        "dominant_pattern_coverage": spec.dominant_pattern_coverage,
                    }
        demo: dict[str, str | None] = {"sex": None, "race": None}
        if col_set.conditioning_columns is not None:
            cond = col_set.conditioning_columns
            if cond.gender:
                demo["sex"] = cond.gender
            if cond.ethnic_background:
                demo["race"] = cond.ethnic_background
        if fields or any(demo.values()):
            roles.append(
                {
                    "role": role_name,
                    "fields": fields,
                    "field_meta": field_meta,
                    "demographics": demo,
                }
            )

    non_person: list[dict[str, Any]] = []
    free_text_columns: list[str] = []
    for col, spec in plan.unassociated_columns_to_replace.items():
        if spec.entity_type == PiiEntity.free_text:
            free_text_columns.append(col)
            continue
        entity = entity_to_engine_label(spec.entity_type)
        if entity:
            ent: dict[str, Any] = {"column": col, "entity": entity}
            if spec.pattern:
                ent["pattern"] = spec.pattern
            if spec.dominant_pattern_coverage is not None:
                ent["dominant_pattern_coverage"] = spec.dominant_pattern_coverage
            non_person.append(ent)

    for col_set in plan.associated_column_sets.values():
        for col, spec in col_set.columns_to_replace.items():
            if spec.entity_type == PiiEntity.free_text:
                free_text_columns.append(col)

    return {
        "group_key": plan.group_key,
        "roles": roles,
        "non_person": non_person,
        "free_text_columns": list(dict.fromkeys(free_text_columns)),
    }


def runtime_plan_to_pii_plan(
    runtime_plan: dict[str, Any],
    *,
    group_key: str | None,
) -> PiiReplacementPlan:
    associated: dict[str, AssociatedColumnSet] = {}
    for role in runtime_plan.get("roles", []):
        from ..config.pii_replacement import PiiColumnPlan, PiiConditioningColumns

        cols: dict[str, PiiColumnPlan] = {}
        field_meta = role.get("field_meta") or {}
        for label, col in (role.get("fields") or {}).items():
            entity = entity_from_engine_label(label)
            if entity is not None:
                meta = field_meta.get(label) or {}
                cols[col] = PiiColumnPlan(
                    entity_type=entity,
                    pattern=meta.get("pattern"),
                    dominant_pattern_coverage=meta.get("dominant_pattern_coverage"),
                )
        demo = role.get("demographics") or {}
        conditioning = None
        if demo.get("sex") or demo.get("race"):
            conditioning = PiiConditioningColumns(
                gender=demo.get("sex"),
                ethnic_background=demo.get("race"),
            )
        if cols or conditioning is not None:
            associated[role["role"]] = AssociatedColumnSet(
                columns_to_replace=cols,
                conditioning_columns=conditioning,
            )

    from ..config.pii_replacement import PiiColumnPlan

    unassociated: dict[str, PiiColumnPlan] = {}
    for ent in runtime_plan.get("non_person", []):
        col = ent.get("column")
        entity = entity_from_engine_label(ent.get("entity"))
        if col and entity is not None:
            unassociated[col] = PiiColumnPlan(
                entity_type=entity,
                pattern=ent.get("pattern"),
                dominant_pattern_coverage=ent.get("dominant_pattern_coverage"),
            )
    for col in runtime_plan.get("free_text_columns", []):
        unassociated[col] = PiiColumnPlan(entity_type=PiiEntity.free_text)

    return PiiReplacementPlan(
        group_key=group_key,
        associated_column_sets=associated,
        unassociated_columns_to_replace=unassociated,
    )


def resolve_plan(
    config: ReplacePiiConfig,
    df: pd.DataFrame,
    *,
    data_config: DataParameters,
    runtime: RuntimeConfig,
) -> PiiReplacementPlan:
    if config.llm_enhancement:
        raise NotImplementedError("llm_enhancement=True is not implemented in this release")

    group_key = data_config.group_training_examples_by

    if config.is_auto_discovery:
        from .discovery import discover_plan

        plan = discover_plan(df, group_key=group_key, runtime=runtime, config=config)
        validate_plan(df, plan, data_config=data_config, runtime=runtime, discovered=True)
        return plan

    if config.plan_path:
        plan = load_plan_from_path(config.plan_path)
        if plan.group_key is None:
            plan = plan.model_copy(update={"group_key": group_key})
        validate_plan(df, plan, data_config=data_config, runtime=runtime, discovered=False)
        return plan

    inline = config.inline_plan
    if inline is None:
        raise ParameterError("replacement_plan must be auto_discovery, a path, or an inline plan")
    plan = inline
    if plan.group_key is None:
        plan = plan.model_copy(update={"group_key": group_key})
    validate_plan(df, plan, data_config=data_config, runtime=runtime, discovered=False)
    return plan
