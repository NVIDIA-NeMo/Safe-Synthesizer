# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural multi-table schema (PK/FK only; no person_reference)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import Field, field_validator, model_validator

from ...config.base import NSSBaseModel
from ...errors import ParameterError

__all__ = [
    "DatabaseSchema",
    "ForeignKey",
    "ForeignKeyTarget",
    "TableSchema",
    "load_schema",
    "qualify",
    "split_qualified",
]


def qualify(table: str, column: str) -> str:
    """Return ``table.column``."""
    return f"{table}.{column}"


def split_qualified(ref: str) -> tuple[str, str]:
    """Split ``table.column`` into ``(table, column)``."""
    if "." not in ref:
        raise ParameterError(f"expected table-qualified column (table.column), got {ref!r}")
    table, column = ref.split(".", 1)
    if not table or not column or "." in column:
        raise ParameterError(f"expected table-qualified column (table.column), got {ref!r}")
    return table, column


class ForeignKeyTarget(NSSBaseModel):
    """One parent target of a (possibly polymorphic) foreign key."""

    table: str = Field(description="Referenced parent table name.")
    columns: list[str] = Field(description="Referenced parent column(s), usually the PK.")
    type_value: str | None = Field(
        default=None,
        description="Discriminator value when this FK is polymorphic (e.g. Contact).",
    )

    @field_validator("columns")
    @classmethod
    def _non_empty_columns(cls, value: list[str]) -> list[str]:
        if not value:
            raise ParameterError("foreign_keys.references.columns must be non-empty")
        return value


class ForeignKey(NSSBaseModel):
    """FK from local ``columns`` to one parent (ordinary) or several (polymorphic).

    Ordinary::

        columns: [ContactId]
        references: Contact.Id

    Polymorphic::

        columns: [WhoId]
        type_column: WhoType
        references:
          - { table: Contact, columns: [Id], type_value: Contact }
          - { table: Lead, columns: [Id], type_value: Lead }
    """

    columns: list[str] = Field(description="Local (bare) column names that form the foreign key.")
    references: str | dict[str, Any] | list[Any] = Field(
        description=(
            "Ordinary: 'Table.column' or {table, columns}. "
            "Polymorphic: list of {table, columns, type_value}."
        ),
    )
    type_column: str | None = Field(
        default=None,
        description="Bare type-discriminator column for polymorphic FKs (e.g. WhoType).",
    )

    @field_validator("columns")
    @classmethod
    def _non_empty_columns(cls, value: list[str]) -> list[str]:
        if not value:
            raise ParameterError("foreign_keys.columns must be non-empty")
        return value

    def is_polymorphic(self) -> bool:
        return self.type_column is not None or isinstance(self.references, list)

    def targets(self) -> list[ForeignKeyTarget]:
        """Normalize ``references`` to a list of ``ForeignKeyTarget``."""
        ref = self.references
        if isinstance(ref, list):
            if not ref:
                raise ParameterError("foreign_keys.references list must be non-empty")
            out: list[ForeignKeyTarget] = []
            for item in ref:
                if isinstance(item, ForeignKeyTarget):
                    out.append(item)
                elif isinstance(item, dict):
                    out.append(ForeignKeyTarget.model_validate(item))
                elif isinstance(item, str):
                    table, column = split_qualified(item)
                    out.append(ForeignKeyTarget(table=table, columns=[column]))
                else:
                    raise ParameterError(
                        f"foreign_keys.references list entries must be mappings or 'Table.col', got {type(item).__name__}"
                    )
            return out

        if isinstance(ref, dict):
            return [ForeignKeyTarget.model_validate(ref)]

        if not isinstance(ref, str) or not ref.strip():
            raise ParameterError("foreign_keys.references must be a string, mapping, or list")
        parts = [p.strip() for p in ref.split(",") if p.strip()]
        tables: list[str] = []
        columns: list[str] = []
        for part in parts:
            table, column = split_qualified(part)
            tables.append(table)
            columns.append(column)
        if len(set(tables)) != 1:
            raise ParameterError(f"foreign_keys.references must name a single table, got {ref!r}")
        return [ForeignKeyTarget(table=tables[0], columns=columns)]

    def referenced_table_columns(self) -> tuple[str, list[str]]:
        """Ordinary-FK helper: single ``(table, columns)``. Raises if polymorphic."""
        tgts = self.targets()
        if len(tgts) != 1 or self.type_column:
            raise ParameterError(
                "referenced_table_columns() is only valid for ordinary (single-parent) foreign keys"
            )
        return tgts[0].table, list(tgts[0].columns)

    @model_validator(mode="after")
    def _validate_shape(self) -> Self:
        tgts = self.targets()
        poly = self.type_column is not None or len(tgts) > 1
        if self.type_column is not None and len(tgts) < 1:
            raise ParameterError("polymorphic foreign key requires at least one references target")
        if isinstance(self.references, list) and self.type_column is None:
            raise ParameterError(
                "polymorphic foreign_keys.references (list) requires type_column"
            )
        if poly:
            for tgt in tgts:
                if not tgt.type_value:
                    raise ParameterError(
                        "polymorphic foreign key targets must include type_value"
                    )
                if len(tgt.columns) != len(self.columns):
                    raise ParameterError(
                        f"foreign key arity mismatch: local columns {self.columns!r} vs "
                        f"referenced {tgt.table}.{tgt.columns!r}"
                    )
            type_values = [t.type_value for t in tgts]
            if len(set(type_values)) != len(type_values):
                raise ParameterError("polymorphic foreign key type_value values must be unique")
        else:
            if len(tgts[0].columns) != len(self.columns):
                raise ParameterError(
                    f"foreign key arity mismatch: local columns {self.columns!r} vs "
                    f"referenced {tgts[0].columns!r}"
                )
        return self


class TableSchema(NSSBaseModel):
    """One table's primary key and foreign keys."""

    primary_key: list[str] = Field(
        default_factory=list,
        description="Bare primary-key column name(s).",
    )
    foreign_keys: list[ForeignKey] = Field(
        default_factory=list,
        description="Foreign keys from this table to others (ordinary or polymorphic).",
    )

    @field_validator("primary_key")
    @classmethod
    def _non_empty_pk(cls, value: list[str]) -> list[str]:
        if not value:
            raise ParameterError("tables.*.primary_key must be non-empty")
        return value


class DatabaseSchema(NSSBaseModel):
    """Structural schema for a folder of CSV tables."""

    tables: dict[str, TableSchema] = Field(description="Table name -> schema.")

    @model_validator(mode="after")
    def _validate_fk_targets(self) -> Self:
        for table_name, table in self.tables.items():
            for fk in table.foreign_keys:
                for tgt in fk.targets():
                    if tgt.table not in self.tables:
                        raise ParameterError(
                            f"foreign key from {table_name!r} references unknown table {tgt.table!r}"
                        )
                    pk = self.tables[tgt.table].primary_key
                    if list(tgt.columns) != list(pk):
                        if set(tgt.columns) != set(pk) or len(tgt.columns) != len(pk):
                            raise ParameterError(
                                f"foreign key from {table_name!r} must reference {tgt.table} primary key "
                                f"{pk!r}, got {tgt.columns!r}"
                            )
        return self

    def table_order_names(self) -> list[str]:
        """Schema listing order (dict insertion order)."""
        return list(self.tables.keys())

    def pk_qualified(self, table: str) -> list[str]:
        return [qualify(table, col) for col in self.tables[table].primary_key]

    def ordinary_fk_links(self) -> list[tuple[str, list[str], str, list[str]]]:
        """Return ordinary (non-polymorphic) ``(child, child_cols, parent, parent_cols)``."""
        links: list[tuple[str, list[str], str, list[str]]] = []
        for table_name, table in self.tables.items():
            for fk in table.foreign_keys:
                if fk.is_polymorphic():
                    continue
                parent, parent_cols = fk.referenced_table_columns()
                links.append((table_name, list(fk.columns), parent, list(parent_cols)))
        return links

    def fk_links(self) -> list[tuple[str, list[str], str, list[str]]]:
        """All parent edges for topo order: ordinary + each polymorphic target.

        Polymorphic FKs contribute one edge per declared parent table.
        """
        links = list(self.ordinary_fk_links())
        for table_name, table in self.tables.items():
            for fk in table.foreign_keys:
                if not fk.is_polymorphic():
                    continue
                for tgt in fk.targets():
                    links.append((table_name, list(fk.columns), tgt.table, list(tgt.columns)))
        return links

    def polymorphic_fks(self) -> list[tuple[str, ForeignKey]]:
        """Return ``(child_table, fk)`` for every polymorphic foreign key."""
        out: list[tuple[str, ForeignKey]] = []
        for table_name, table in self.tables.items():
            for fk in table.foreign_keys:
                if fk.is_polymorphic():
                    out.append((table_name, fk))
        return out

    def type_discriminator_columns(self) -> set[str]:
        """Qualified type-discriminator columns that must not be replaced."""
        cols: set[str] = set()
        for table_name, fk in self.polymorphic_fks():
            if fk.type_column:
                cols.add(qualify(table_name, fk.type_column))
        return cols

    def polymorphic_id_columns(self) -> set[str]:
        """Qualified polymorphic Id columns, which are routed per row.

        These hold values from several parents, so they never join a key domain.
        """
        return {qualify(table_name, col) for table_name, fk in self.polymorphic_fks() for col in fk.columns}


def load_schema(path: Path | str) -> DatabaseSchema:
    """Load and validate a schema YAML file."""
    p = Path(path)
    try:
        text = p.read_text()
    except OSError as exc:
        raise ParameterError(f"Could not read schema file {path!r}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ParameterError(f"Invalid YAML in schema file {path!r}: {exc}") from exc
    if not isinstance(data, dict):
        raise ParameterError(f"schema file {path!r} must contain a mapping")
    try:
        return DatabaseSchema.model_validate(data)
    except Exception as exc:
        raise ParameterError(f"Invalid schema in {path!r}: {exc}") from exc
