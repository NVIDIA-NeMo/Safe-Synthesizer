# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project a database-scope plan table into a single-table engine plan."""

from __future__ import annotations

from ...config.replace_pii import (
    KeyDomain,
    PersonaColumnSet,
    PersonaMatchColumn,
    PiiColumnPlan,
    PiiReplacementPlan,
    PiiReplacementScope,
    PolymorphicForeignKeyPlan,
    TableReplacementPlan,
)
from ...errors import ParameterError
from .schema import split_qualified
from .store import PolymorphicColumnRoute, SharedRuntimeStore, TableRunContext

__all__ = ["build_table_context", "project_table_plan", "strip_table_prefix"]


def strip_table_prefix(qualified: str, table_name: str) -> str:
    """Strip ``table_name.`` from a qualified column, or return unchanged if bare."""
    prefix = f"{table_name}."
    if qualified.startswith(prefix):
        return qualified[len(prefix) :]
    if "." in qualified:
        other, _col = split_qualified(qualified)
        if other != table_name:
            raise ParameterError(
                f"column {qualified!r} does not belong to table {table_name!r}"
            )
    return qualified


def _project_column_plan(spec: PiiColumnPlan, table_name: str) -> PiiColumnPlan:
    return PiiColumnPlan(
        column_name=strip_table_prefix(spec.column_name, table_name),
        entity_type=spec.entity_type,
        patterns=list(spec.patterns),
    )


def _project_persona_set(col_set: PersonaColumnSet, table_name: str) -> PersonaColumnSet:
    return PersonaColumnSet(
        persona=col_set.persona,
        columns_to_replace=[_project_column_plan(s, table_name) for s in col_set.columns_to_replace],
        match_persona_by=[
            PersonaMatchColumn(
                persona_attribute=m.persona_attribute,
                column_name=strip_table_prefix(m.column_name, table_name),
            )
            for m in col_set.match_persona_by
        ],
        person_key_domain=col_set.person_key_domain,
    )


def project_table_plan(table_name: str, table_plan: TableReplacementPlan) -> PiiReplacementPlan:
    """Return a single-table ``dataframe``-scope plan with bare column names."""
    return PiiReplacementPlan(
        scope=PiiReplacementScope.dataframe,
        persona_backed_columns=[_project_persona_set(p, table_name) for p in table_plan.persona_backed_columns],
        standalone_columns_to_replace=[
            _project_column_plan(s, table_name) for s in table_plan.standalone_columns_to_replace
        ],
    )


def _resolve_key_column_for_domain(
    table_name: str,
    domain_id: str,
    store: SharedRuntimeStore,
) -> str | None:
    """Pick the bare column on ``table_name`` that holds keys for ``domain_id``.

    Prefers the domain's home PK (``Contact.Id`` → ``Id``) over other members of
    the same domain that happen to live on this table (e.g. ``ReportsToId``).
    """
    domain = store.domains.get(domain_id)
    if domain is None:
        return None
    table_cols: list[str] = []
    for qualified in domain.columns:
        try:
            t, bare = split_qualified(qualified)
        except ParameterError:
            continue
        if t == table_name:
            table_cols.append(bare)
    if not table_cols:
        return None
    if "." in domain_id:
        home_table, home_col = split_qualified(domain_id)
        if home_table == table_name and home_col in table_cols:
            return home_col
    return table_cols[0]


def build_table_context(
    table_name: str,
    table_plan: TableReplacementPlan,
    key_domains: list[KeyDomain],
    store: SharedRuntimeStore,
    *,
    polymorphic_foreign_keys: list[PolymorphicForeignKeyPlan] | None = None,
) -> TableRunContext:
    """Build engine context linking bare columns to shared domains / person refs."""
    column_domains: dict[str, str] = {}
    person_ref_columns: dict[str, str] = {}
    for kd in key_domains:
        for qualified in kd.columns:
            try:
                t, bare = split_qualified(qualified)
            except ParameterError:
                continue
            if t != table_name:
                continue
            column_domains[bare] = kd.id
            if kd.person_reference or (store.domains.get(kd.id) and store.domains[kd.id].person_reference):
                person_ref_columns[bare] = kd.id

    polymorphic_routes: dict[str, PolymorphicColumnRoute] = {}
    for poly in polymorphic_foreign_keys or []:
        try:
            t, bare = split_qualified(poly.column)
            _tt, type_bare = split_qualified(poly.type_column)
        except ParameterError:
            continue
        if t != table_name:
            continue
        route = PolymorphicColumnRoute(
            bare_column=bare,
            type_column=type_bare,
            targets={tgt.type_value: tgt.domain for tgt in poly.targets},
        )
        polymorphic_routes[bare] = route
        # Person free-text: polymorphic col can pull pairs from whichever domain it routes to.
        # Mark as a dynamic person ref; apply resolves domain per row.
        for domain_id in route.targets.values():
            state = store.domains.get(domain_id)
            if state and state.person_reference:
                person_ref_columns.setdefault(bare, domain_id)

    # Default domain when a persona set omits person_key_domain: this table's PK domain
    # marked person_reference (same heuristic as discovery).
    default_domain: str | None = None
    default_key_col: str | None = None
    for kd in key_domains:
        if not kd.person_reference:
            continue
        for qualified in kd.columns:
            try:
                t, bare = split_qualified(qualified)
            except ParameterError:
                continue
            if t == table_name and kd.id.endswith(f".{bare}"):
                default_domain = kd.id
                default_key_col = bare
                break
        if default_domain:
            break

    persona_key_bindings: dict[str, tuple[str, str]] = {}
    for col_set in table_plan.persona_backed_columns:
        domain_id = col_set.person_key_domain or default_domain
        if not domain_id:
            continue
        key_col = _resolve_key_column_for_domain(table_name, domain_id, store)
        if key_col is None and domain_id == default_domain:
            key_col = default_key_col
        if key_col is None:
            continue
        persona_key_bindings[col_set.persona] = (domain_id, key_col)

    return TableRunContext(
        table_name=table_name,
        column_domains=column_domains,
        person_ref_columns=person_ref_columns,
        persona_key_bindings=persona_key_bindings,
        polymorphic_routes=polymorphic_routes,
    )
