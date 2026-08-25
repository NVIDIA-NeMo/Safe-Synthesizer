# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-aware multi-table PII plan discovery."""

from __future__ import annotations

import pandas as pd

from ...config.replace_pii import (
    KeyDomain,
    PersonaColumnSet,
    PersonaMatchColumn,
    PiiColumnPlan,
    PiiEntity,
    PiiReplacementPlan,
    PiiReplacementScope,
    ReplacePiiConfig,
    PolymorphicForeignKeyPlan,
    PolymorphicFkTarget,
    TableReplacementPlan,
)
from ...observability import get_logger
from .. import entities
from ..llm import PiiDiscoveryEnhancer
from ..log_context import discovery_table
from ..planning.discovery import discover_plan
from .domains import (
    build_key_domains_from_schema,
    bundle_overlap_domains,
    propose_person_reference,
)
from .schema import DatabaseSchema, qualify

logger = get_logger(__name__)

__all__ = ["discover_database_plan", "polymorphic_plans_from_schema", "qualify_table_plan"]


def qualify_table_plan(table_name: str, plan: PiiReplacementPlan) -> TableReplacementPlan:
    """Convert a single-table discovered plan into a qualified table body."""

    def _q(col: str) -> str:
        return col if "." in col else qualify(table_name, col)

    personas: list[PersonaColumnSet] = []
    for col_set in plan.persona_backed_columns:
        personas.append(
            PersonaColumnSet(
                persona=col_set.persona,
                columns_to_replace=[
                    PiiColumnPlan(
                        column_name=_q(spec.column_name),
                        entity_type=spec.entity_type,
                        patterns=list(spec.patterns),
                    )
                    for spec in col_set.columns_to_replace
                ],
                match_persona_by=[
                    PersonaMatchColumn(
                        persona_attribute=m.persona_attribute,
                        column_name=_q(m.column_name),
                    )
                    for m in col_set.match_persona_by
                ],
                person_key_domain=col_set.person_key_domain,
            )
        )
    standalone = [
        PiiColumnPlan(
            column_name=_q(spec.column_name),
            entity_type=spec.entity_type,
            patterns=list(spec.patterns),
        )
        for spec in plan.standalone_columns_to_replace
    ]
    return TableReplacementPlan(
        persona_backed_columns=personas,
        standalone_columns_to_replace=standalone,
    )


def _attach_person_key_domains(
    tables: dict[str, TableReplacementPlan],
    domains: list[KeyDomain],
    schema: DatabaseSchema,
) -> None:
    """Default ``person_key_domain`` on persona sets to the table PK domain when applicable."""
    pk_domain_by_table: dict[str, str] = {}
    for d in domains:
        if not d.person_reference:
            continue
        for qualified in d.columns:
            if "." not in qualified:
                continue
            table, col = qualified.split(".", 1)
            if table in schema.tables and schema.tables[table].primary_key == [col]:
                pk_domain_by_table[table] = d.id

    for table_name, table_plan in tables.items():
        default_domain = pk_domain_by_table.get(table_name)
        if not default_domain:
            continue
        for i, col_set in enumerate(table_plan.persona_backed_columns):
            if col_set.person_key_domain is None:
                table_plan.persona_backed_columns[i] = col_set.model_copy(
                    update={"person_key_domain": default_domain}
                )


def _domain_id_for_parent(domains: list[KeyDomain], parent_table: str, parent_cols: list[str]) -> str:
    """Resolve the key-domain id for a parent PK (prefer ``Table.pk``)."""
    preferred = qualify(parent_table, parent_cols[0]) if len(parent_cols) == 1 else None
    if preferred and any(d.id == preferred for d in domains):
        return preferred
    for d in domains:
        for col in d.columns:
            if col.startswith(f"{parent_table}."):
                bare = col.split(".", 1)[1]
                if bare in parent_cols:
                    return d.id
    return preferred or qualify(parent_table, parent_cols[0])


def polymorphic_plans_from_schema(
    schema: DatabaseSchema,
    domains: list[KeyDomain],
) -> list[PolymorphicForeignKeyPlan] | None:
    """Emit plan routers for schema polymorphic FKs, or ``None`` when none exist."""
    plans: list[PolymorphicForeignKeyPlan] = []
    for table_name, fk in schema.polymorphic_fks():
        if not fk.type_column or len(fk.columns) != 1:
            # v1: single-column polymorphic Ids only
            logger.user.warning(
                f"[PII Replacement] Skipping polymorphic FK on {table_name!r}: "
                "v1 supports a single Id column plus type_column."
            )
            continue
        targets = [
            PolymorphicFkTarget(
                type_value=str(tgt.type_value),
                domain=_domain_id_for_parent(domains, tgt.table, list(tgt.columns)),
            )
            for tgt in fk.targets()
        ]
        plans.append(
            PolymorphicForeignKeyPlan(
                column=qualify(table_name, fk.columns[0]),
                type_column=qualify(table_name, fk.type_column),
                targets=targets,
            )
        )
    return plans or None


def _strip_type_discriminator_columns(
    tables: dict[str, TableReplacementPlan],
    schema: DatabaseSchema,
) -> None:
    """Remove type discriminator columns from plan bodies (routing only, not PII)."""
    banned = schema.type_discriminator_columns()
    if not banned:
        return
    for table_name, table_plan in tables.items():
        before = len(table_plan.standalone_columns_to_replace)
        table_plan.standalone_columns_to_replace = [
            spec
            for spec in table_plan.standalone_columns_to_replace
            if spec.column_name not in banned
        ]
        for col_set in table_plan.persona_backed_columns:
            col_set.columns_to_replace = [
                spec for spec in col_set.columns_to_replace if spec.column_name not in banned
            ]
        removed = before - len(table_plan.standalone_columns_to_replace)
        if removed:
            logger.runtime.info(
                f"[PII Replacement] Omitting {removed} type-discriminator column(s) from "
                f"{table_name!r} plan (routing only)."
            )


def _ensure_polymorphic_id_standalone(
    tables: dict[str, TableReplacementPlan],
    poly: list[PolymorphicForeignKeyPlan] | None,
) -> None:
    """Ensure polymorphic Id columns are listed as standalone unique_identifier."""
    if not poly:
        return
    for entry in poly:
        table, bare = entry.column.split(".", 1)
        table_plan = tables.get(table)
        if table_plan is None:
            continue
        existing = {spec.column_name for spec in table_plan.standalone_columns_to_replace}
        if entry.column in existing:
            continue
        table_plan.standalone_columns_to_replace.append(
            PiiColumnPlan(column_name=entry.column, entity_type=PiiEntity.unique_identifier)
        )


def discover_database_plan(
    frames: dict[str, pd.DataFrame],
    schema: DatabaseSchema,
    cfg: entities.Config,
    config: ReplacePiiConfig,
    *,
    enhancer: PiiDiscoveryEnhancer | None = None,
) -> PiiReplacementPlan:
    """Discover one database-scope plan over a folder of tables + schema."""
    tables: dict[str, TableReplacementPlan] = {}
    for table_name in schema.table_order_names():
        df = frames[table_name]
        with discovery_table(table_name):
            single = discover_plan(df, group_key=None, cfg=cfg, config=config, enhancer=enhancer)
        tables[table_name] = qualify_table_plan(table_name, single)

    schema_domains = build_key_domains_from_schema(schema)
    domains = propose_person_reference(schema_domains, tables, schema)
    domains = bundle_overlap_domains(frames, domains, tables, schema)
    domains = propose_person_reference(domains, tables, schema)
    _attach_person_key_domains(tables, domains, schema)

    poly = polymorphic_plans_from_schema(schema, domains)
    _strip_type_discriminator_columns(tables, schema)
    _ensure_polymorphic_id_standalone(tables, poly)

    logger.user.info(
        "Discovered database PII replacement plan",
        extra={
            "tables": len(tables),
            "key_domains": len(domains),
            "person_reference_domains": sum(1 for d in domains if d.person_reference),
            "polymorphic_foreign_keys": 0 if poly is None else len(poly),
        },
    )
    return PiiReplacementPlan(
        scope=PiiReplacementScope.database,
        key_domains=domains,
        polymorphic_foreign_keys=poly,
        tables=tables,
    )
