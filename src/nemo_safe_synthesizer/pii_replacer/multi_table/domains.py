# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build key domains from schema PK/FK equivalence classes + overlap bundling."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from ...config.replace_pii import KeyDomain, TableReplacementPlan
from ...observability import get_logger
from .overlap import (
    OVERLAP_ELIGIBLE_ENTITIES,
    column_value_set,
    should_bundle_by_overlap,
    warn_if_schema_domain_disjoint,
)
from .schema import DatabaseSchema, qualify, split_qualified

logger = get_logger(__name__)

__all__ = ["build_key_domains_from_schema", "bundle_overlap_domains"]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            groups[self.find(item)].append(item)
        return dict(groups)


def build_key_domains_from_schema(schema: DatabaseSchema) -> list[KeyDomain]:
    """Build key domains from PK columns and ordinary FK links (schema first).

    Polymorphic FKs do **not** merge parent domains; each parent PK stays its own
    domain. Polymorphic Id columns are routed per row at apply time.
    """
    uf = _UnionFind()
    for table_name, table in schema.tables.items():
        for col in table.primary_key:
            uf.add(qualify(table_name, col))
    for child, child_cols, parent, parent_cols in schema.ordinary_fk_links():
        for c_col, p_col in zip(child_cols, parent_cols, strict=True):
            uf.union(qualify(child, c_col), qualify(parent, p_col))

    domains: list[KeyDomain] = []
    for _root, members in uf.components().items():
        members_sorted = sorted(members)
        # Prefer PK table.column as domain id: choose the member that is a PK.
        domain_id = members_sorted[0]
        for qualified in members_sorted:
            table, col = split_qualified(qualified)
            if col in schema.tables[table].primary_key and len(schema.tables[table].primary_key) == 1:
                domain_id = qualify(table, col)
                break
        domains.append(KeyDomain(id=domain_id, person_reference=False, columns=members_sorted))
    # Stable order by domain id
    domains.sort(key=lambda d: d.id)
    return domains


def _standalone_entity_columns(
    tables: dict[str, TableReplacementPlan],
) -> list[tuple[str, str, str]]:
    """Return ``(qualified, entity, table)`` for overlap-eligible standalone cols."""
    out: list[tuple[str, str, str]] = []
    for table_name, table_plan in tables.items():
        for spec in table_plan.standalone_columns_to_replace:
            if spec.entity_type is None:
                continue
            entity = spec.entity_type.value
            if entity not in OVERLAP_ELIGIBLE_ENTITIES:
                continue
            # Specs may already be qualified from discovery.
            col = spec.column_name
            if "." not in col:
                col = qualify(table_name, col)
            out.append((col, entity, table_name))
    return out


def bundle_overlap_domains(
    frames: dict[str, pd.DataFrame],
    schema_domains: list[KeyDomain],
    tables: dict[str, TableReplacementPlan],
    schema: DatabaseSchema,
) -> list[KeyDomain]:
    """Extend schema domains with value-overlap bundles; warn on disjoint schema links.

    Schema PK/FK domains are authoritative: overlap may pull an orphan column into
    a domain (a missing FK), but never merges two schema domains. Polymorphic Id
    columns are excluded because they legitimately carry values from several
    parents, which would otherwise chain every parent domain together.
    """
    column_domain = {c: d.id for d in schema_domains for c in d.columns}
    value_sets: dict[str, set[str]] = {}
    for d in schema_domains:
        for qualified in d.columns:
            table, col = split_qualified(qualified)
            if table in frames and col in frames[table].columns:
                value_sets[qualified] = column_value_set(frames[table][col])
            else:
                value_sets[qualified] = set()
        warn_if_schema_domain_disjoint(d.id, d.columns, value_sets)

    routed_columns = schema.polymorphic_id_columns() | schema.type_discriminator_columns()
    by_entity: dict[str, list[str]] = defaultdict(list)
    for qualified, entity, table_name in _standalone_entity_columns(tables):
        table, col = split_qualified(qualified) if "." in qualified else (table_name, qualified)
        q = qualify(table, col)
        if q in routed_columns:
            continue
        if table in frames and col in frames[table].columns:
            value_sets[q] = column_value_set(frames[table][col])
        by_entity[entity].append(q)

    uf = _UnionFind()
    for d in schema_domains:
        for c in d.columns:
            uf.add(c)
            uf.union(c, d.columns[0])

    # Schema domain id owning each component, so overlap cannot merge two of them.
    schema_id_by_root: dict[str, str] = {}
    for d in schema_domains:
        if d.columns:
            schema_id_by_root[uf.find(d.columns[0])] = d.id

    def _bundle(left: str, right: str) -> None:
        root_left, root_right = uf.find(left), uf.find(right)
        if root_left == root_right:
            return
        schema_left = schema_id_by_root.get(root_left)
        schema_right = schema_id_by_root.get(root_right)
        if schema_left and schema_right:
            logger.user.warning(
                f"[PII Replacement] Columns {left!r} and {right!r} share values but belong to "
                f"key domains {schema_left!r} and {schema_right!r}; keeping the schema domains "
                "separate."
            )
            return
        uf.union(left, right)
        merged_root = uf.find(left)
        owner = schema_left or schema_right
        schema_id_by_root.pop(root_left, None)
        schema_id_by_root.pop(root_right, None)
        if owner:
            schema_id_by_root[merged_root] = owner

    for cols in by_entity.values():
        unique_cols = list(dict.fromkeys(cols))
        for i, left in enumerate(unique_cols):
            uf.add(left)
            for right in unique_cols[i + 1 :]:
                uf.add(right)
                if column_domain.get(left) and column_domain.get(left) == column_domain.get(right):
                    continue
                a = value_sets.get(left) or set()
                b = value_sets.get(right) or set()
                if should_bundle_by_overlap(a, b):
                    _bundle(left, right)

    # Rebuild domains from union-find, preserving person_reference and preferred ids.
    id_by_member = {c: d.id for d in schema_domains for c in d.columns}
    person_ref = {d.id: d.person_reference for d in schema_domains}
    out: list[KeyDomain] = []
    for members in uf.components().values():
        members_sorted = sorted(set(members))
        if len(members_sorted) == 1 and members_sorted[0] not in id_by_member:
            # Singleton orphan with no schema membership and no overlap partner: skip as domain
            # (still replaced per-column via standalone maps).
            continue
        preferred = None
        for m in members_sorted:
            if m in id_by_member:
                preferred = id_by_member[m]
                break
        domain_id = preferred or members_sorted[0]
        ref = False
        for m in members_sorted:
            mid = id_by_member.get(m)
            if mid and person_ref.get(mid):
                ref = True
                break
        out.append(KeyDomain(id=domain_id, person_reference=ref, columns=members_sorted))

    # A schema domain must never be dropped, but re-adding one whose columns already
    # landed in another domain would put those columns in two domains at once.
    covered = {c for d in out for c in d.columns}
    for d in schema_domains:
        if not any(c in covered for c in d.columns):
            out.append(d)

    out.sort(key=lambda d: d.id)
    return out


def propose_person_reference(
    domains: list[KeyDomain],
    tables: dict[str, TableReplacementPlan],
    schema: DatabaseSchema,
) -> list[KeyDomain]:
    """Set ``person_reference`` when persona columns attach to a domain's home table."""
    updated: list[KeyDomain] = []
    for d in domains:
        person_ref = d.person_reference
        home_table = split_qualified(d.id)[0] if "." in d.id else None
        if home_table and home_table in tables:
            table_plan = tables[home_table]
            if table_plan.persona_backed_columns:
                # Domain should include this table's PK when it's the home key.
                pk_cols = set(schema.tables[home_table].primary_key) if home_table in schema.tables else set()
                domain_bares = {split_qualified(c)[1] for c in d.columns if split_qualified(c)[0] == home_table}
                if pk_cols & domain_bares or any(
                    split_qualified(c)[0] == home_table and c == d.id for c in d.columns
                ):
                    person_ref = True
        updated.append(KeyDomain(id=d.id, person_reference=person_ref, columns=list(d.columns)))
    return updated
