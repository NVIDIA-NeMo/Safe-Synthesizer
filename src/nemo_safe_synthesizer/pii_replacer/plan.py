# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan resolution, validation, and conversion to the runtime engine shape."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..config.data import DataParameters
from ..config.pii_replacement import (
    PiiColumnPlan,
    PiiEntity,
    PiiPersona,
    PiiReplacementPlan,
    ReplacePiiConfig,
    is_person_entity,
)
from ..errors import ParameterError
from . import core
from .runtime_config import RuntimeConfig

# Implicit role for person-sourced columns that name no persona. It never carries
# demographic conditioning; it only groups such columns for consistent replacement.
_PERSONALESS_ROLE = "_personaless"


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


def _unique_ratios(df: pd.DataFrame, plan: PiiReplacementPlan) -> dict[str, float]:
    """Per-column ``unique_ratio`` used for unique_identifier gating.

    When ``plan.group_key`` is set the cardinality of a per-group (group-constant)
    column is measured against the number of groups rather than rows, matching the
    detector's ``scoped_column_stats``. Otherwise a per-group identifier repeating
    across every row of its group would appear low-cardinality.
    """
    stats = core.scoped_column_stats(df, plan.group_key)
    return {c: s.get("unique_ratio", 0.0) for c, s in stats.items()}


def _replacement_columns(plan: PiiReplacementPlan) -> list[str]:
    return list(plan.columns)


def _field_meta(spec: PiiColumnPlan) -> dict[str, Any] | None:
    if spec.pattern is None and spec.dominant_pattern_coverage is None:
        return None
    return {"pattern": spec.pattern, "dominant_pattern_coverage": spec.dominant_pattern_coverage}


def _build_role(
    role_name: str,
    columns: dict[str, PiiColumnPlan],
    *,
    persona: PiiPersona | None,
) -> dict[str, Any] | None:
    """Assemble one engine role from a persona's person-sourced columns.

    Only person-sourced fields (names, email, phone, ssn, address) belong here;
    entity-driven columns are routed to the non-person path before this is called.
    """
    fields: dict[str, str] = {}
    field_meta: dict[str, dict[str, Any]] = {}
    for col, spec in columns.items():
        if spec.entity_type is None or spec.entity_type == PiiEntity.free_text:
            continue
        if not is_person_entity(spec.entity_type):
            continue
        label = entity_to_engine_label(spec.entity_type)
        if not label:
            continue
        fields[label] = col
        meta = _field_meta(spec)
        if meta:
            field_meta[label] = meta

    demo: dict[str, str | None] = {"sex": None, "race": None}
    if persona is not None:
        if persona.gender:
            demo["sex"] = persona.gender
        if persona.ethnic_background:
            demo["race"] = persona.ethnic_background

    if not fields and not any(demo.values()):
        return None
    return {
        "role": role_name,
        "fields": fields,
        "field_meta": field_meta,
        "demographics": demo,
    }


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

    for col, spec in plan.columns.items():
        if spec.persona and spec.persona not in plan.identified_personas:
            raise ParameterError(
                f"column {col!r} references unknown persona {spec.persona!r}; add it to identified_personas"
            )

    for persona_name, persona in plan.identified_personas.items():
        if persona is None:
            continue
        for attr in ("gender", "ethnic_background"):
            col = getattr(persona, attr)
            if col and col not in df_cols:
                raise ParameterError(
                    f"persona {persona_name!r} conditioning column {col!r} not found in dataframe"
                )

    if discovered and runtime is not None:
        ratios = _unique_ratios(df, plan)
        for col, spec in plan.columns.items():
            if spec.entity_type == PiiEntity.unique_identifier:
                ratio = ratios.get(col, 0.0)
                if ratio < runtime.id_unique_ratio:
                    raise ParameterError(
                        f"auto-discovered unique_id column {col!r} failed id_unique_ratio gate "
                        f"({ratio:.4f} < {runtime.id_unique_ratio})"
                    )


def unique_id_advisories(df: pd.DataFrame, plan: PiiReplacementPlan, runtime: RuntimeConfig) -> list[str]:
    warnings: list[str] = []
    ratios = _unique_ratios(df, plan)
    for col, spec in plan.columns.items():
        if spec.entity_type != PiiEntity.unique_identifier or col not in df.columns:
            continue
        ratio = ratios.get(col, 0.0)
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


PII_REPLACEMENT_PLAN_FILENAME = "pii_replacement_plan.yaml"


def save_plan_to_path(plan: PiiReplacementPlan, path: str | Path) -> Path:
    """Write a replacement plan as YAML, omitting fields with null values."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(plan.model_dump_json(exclude_none=True))
    with out.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out


# Entity-driven ("self-sourced") entities: their synthetic value is derived from
# the original value via pattern/perturbation, independent of any persona. They
# are always replaced through the non-person path regardless of where the plan
# places them, so associating one with a persona never changes the output.
ENTITY_DRIVEN_ENTITIES = frozenset(core.NON_PERSON_ENTITIES) | {PiiEntity.date_of_birth.value}


def _non_person_entry(col: str, spec: PiiColumnPlan) -> dict[str, Any]:
    ent: dict[str, Any] = {"column": col, "entity": entity_to_engine_label(spec.entity_type)}
    if spec.pattern:
        ent["pattern"] = spec.pattern
    if spec.dominant_pattern_coverage is not None:
        ent["dominant_pattern_coverage"] = spec.dominant_pattern_coverage
    return ent


def plan_to_runtime(plan: PiiReplacementPlan) -> dict[str, Any]:
    """Convert declarative plan to the runtime dict consumed by the engine."""
    roles: list[dict[str, Any]] = []
    non_person: list[dict[str, Any]] = []
    free_text_columns: list[str] = []

    persona_columns: dict[str, dict[str, PiiColumnPlan]] = {}
    personaless_person: dict[str, PiiColumnPlan] = {}

    for col, spec in plan.columns.items():
        if spec.entity_type is None:
            continue
        if spec.entity_type == PiiEntity.free_text:
            free_text_columns.append(col)
            continue
        label = entity_to_engine_label(spec.entity_type)
        if not label:
            continue
        # Entity-driven columns (unique_identifier, date_of_birth, ...) and
        # identify-only temporal columns are never persona-sourced, so they always
        # take the non-person path even when a persona is named. This makes persona
        # association irrelevant to their replacement result.
        if label in ENTITY_DRIVEN_ENTITIES or not is_person_entity(spec.entity_type):
            non_person.append(_non_person_entry(col, spec))
            continue
        if spec.persona:
            persona_columns.setdefault(spec.persona, {})[col] = spec
        else:
            personaless_person[col] = spec

    # Declared personas first (conditioning applies even before columns are seen),
    # then personas referenced only by a column, then the implicit personaless role.
    for persona_name in plan.identified_personas:
        role = _build_role(
            persona_name,
            persona_columns.get(persona_name, {}),
            persona=plan.identified_personas.get(persona_name),
        )
        if role:
            roles.append(role)
    for persona_name, cols in persona_columns.items():
        if persona_name in plan.identified_personas:
            continue
        role = _build_role(persona_name, cols, persona=None)
        if role:
            roles.append(role)
    if personaless_person:
        role = _build_role(_PERSONALESS_ROLE, personaless_person, persona=None)
        if role:
            roles.append(role)

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
    identified_personas: dict[str, PiiPersona | None] = {}
    columns: dict[str, PiiColumnPlan] = {}

    for role in runtime_plan.get("roles", []):
        role_name = role["role"]
        field_meta = role.get("field_meta") or {}

        if role_name == _PERSONALESS_ROLE:
            for label, col in (role.get("fields") or {}).items():
                entity = entity_from_engine_label(label)
                if entity is None:
                    continue
                meta = field_meta.get(label) or {}
                columns[col] = PiiColumnPlan(
                    entity_type=entity,
                    pattern=meta.get("pattern"),
                    dominant_pattern_coverage=meta.get("dominant_pattern_coverage"),
                )
            continue

        demo = role.get("demographics") or {}
        if demo.get("sex") or demo.get("race"):
            identified_personas[role_name] = PiiPersona(
                gender=demo.get("sex"),
                ethnic_background=demo.get("race"),
            )
        else:
            identified_personas.setdefault(role_name, None)

        for label, col in (role.get("fields") or {}).items():
            entity = entity_from_engine_label(label)
            if entity is None:
                continue
            meta = field_meta.get(label) or {}
            columns[col] = PiiColumnPlan(
                entity_type=entity,
                persona=role_name,
                pattern=meta.get("pattern"),
                dominant_pattern_coverage=meta.get("dominant_pattern_coverage"),
            )

    for ent in runtime_plan.get("non_person", []):
        col = ent.get("column")
        entity = entity_from_engine_label(ent.get("entity"))
        if col and entity is not None:
            columns[col] = PiiColumnPlan(
                entity_type=entity,
                pattern=ent.get("pattern"),
                dominant_pattern_coverage=ent.get("dominant_pattern_coverage"),
            )

    for col in runtime_plan.get("free_text_columns", []):
        columns[col] = PiiColumnPlan(entity_type=PiiEntity.free_text)

    return PiiReplacementPlan(
        group_key=group_key,
        identified_personas=identified_personas,
        columns=columns,
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
