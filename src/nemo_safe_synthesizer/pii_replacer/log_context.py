# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context for PII discovery/replacement log messages."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

__all__ = ["column_log_label", "discovery_table"]

_discovery_table: ContextVar[str | None] = ContextVar("pii_discovery_table", default=None)


@contextmanager
def discovery_table(table_name: str | None) -> Iterator[None]:
    """Qualify bare column names in discovery logs as ``Table.column``.

    Used by database-scope discovery so per-table skip warnings are unambiguous
    when many tables share headers like ``Name`` or ``Phone``.
    """
    token = _discovery_table.set(table_name)
    try:
        yield
    finally:
        _discovery_table.reset(token)


def column_log_label(col: str) -> str:
    """Return ``col``, or ``{table}.{col}`` when a discovery table is active."""
    table = _discovery_table.get()
    if table:
        return f"{table}.{col}"
    return col
